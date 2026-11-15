"""Source-only Colab preparation/export helpers; no GPU or dependency changes."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def extract_archive(source, directory):
    source, directory = Path(source), Path(directory).resolve()
    with zipfile.ZipFile(source) as archive:
        for entry in archive.infolist():
            target = (directory/entry.filename).resolve()
            if ('\\' in entry.filename or ':' in entry.filename
                    or not target.is_relative_to(directory)
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError('Unsafe archive entry: '+entry.filename)
        archive.extractall(directory)


def stage_split(source, scratch, *, kind, limit=None):
    """Copy selected input only; answers never enter the inference split."""
    if kind not in {'train', 'public'}:
        raise ValueError('kind must be train or public')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer or None')
    if kind == 'public' and limit is not None:
        raise ValueError('Public submission must use all questions; use TRAIN for smoke tests')
    source, scratch = Path(source).resolve(), Path(scratch).resolve()
    if source.is_file():
        if source.suffix.lower() != '.zip':
            raise ValueError('Provide a split directory or ZIP')
        archive_root = scratch/'archives'/sha256(source)
        marker = archive_root/'.complete'
        if not marker.is_file():
            extract_archive(source, archive_root)
            marker.write_text('ok\n', encoding='utf-8')
        source = archive_root
    candidates = [p.parent for p in source.rglob('questions.jsonl')
                  if all((p.parent/name).is_file() for name in ('rulebook.md', 'calendar.csv'))
                  and (p.parent/'data').is_dir()]
    if len(candidates) != 1:
        raise ValueError('Point to exactly one split (not a parent containing train and public)')
    root = candidates[0]
    questions = [json.loads(line) for line in (root/'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    identifiers = [q['id'] for q in questions]
    if not questions or len(set(identifiers)) != len(questions):
        raise ValueError('Questions must have unique IDs')
    questions = questions[:limit] if limit else questions
    businesses = sorted({q['business_id'] for q in questions})
    files = [root/'rulebook.md', root/'calendar.csv']
    for business in businesses:
        directory = (root/'data'/business).resolve()
        if not directory.is_relative_to((root/'data').resolve()) or not directory.is_dir():
            raise ValueError('Invalid business path')
        files.extend(sorted(p for p in directory.rglob('*') if p.is_file()))
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Input contains a linked file outside the split')
    hashes = {p.relative_to(root).as_posix(): sha256(p) for p in files}
    question_bytes = ''.join(json.dumps(q, ensure_ascii=False)+'\n' for q in questions).encode('utf-8')
    hashes['questions.jsonl'] = hashlib.sha256(question_bytes).hexdigest()
    identity = {'kind': kind, 'questions': len(questions), 'businesses': len(businesses), 'files': hashes}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = scratch/'splits'/fingerprint
    destination.mkdir(parents=True, exist_ok=True)
    for relative, expected in hashes.items():
        target = destination/relative
        if target.is_file() and sha256(target) == expected:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == 'questions.jsonl':
            target.write_bytes(question_bytes)
        else:
            shutil.copy2(root/relative, target)
        if sha256(target) != expected:
            raise ValueError('Input copy hash mismatch: '+relative)
    return destination, {**identity, 'fingerprint': fingerprint, 'original_root': str(root)}


def export_submission(split, result, destination):
    from tools.verify_submission import verify_result
    split, result, destination = Path(split), Path(result), Path(destination)
    validation = verify_result(split, result)
    questions = [json.loads(line) for line in (split/'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    rows = [json.loads(line) for line in (result/'submission.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {row['id']: row for row in rows}
    exported = [{key: value for key, value in by_id[q['id']].items() if key in {'id', 'type', 'answer'}} for q in questions]
    text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in exported)
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise ValueError('Export exists with different answers; choose a new destination')
    destination.write_text(text, encoding='utf-8', newline='\n')
    report = {**validation, 'types': dict(Counter(row['type'] for row in exported)), 'sha256': sha256(destination)}
    write_json(destination.with_suffix('.validation.json'), report)
    return report


def diagnostics(result):
    logs = [json.loads(p.read_text(encoding='utf-8')) for p in (Path(result)/'logs').glob('*.json')]
    reasons, types = Counter(), Counter()
    for log in logs:
        answer = log['submission']
        types[answer['type']] += 1
        if answer['type'] != 'abstain':
            continue
        reason = answer.get('reason', '')
        category = ('context_overflow' if 'Context' in reason else
                    'repeated_action' if 'action/kết quả' in reason else
                    'steps_or_time' if reason.startswith('Hết ngân sách') else
                    'invalid_action' if 'dừng retry' in reason else 'other')
        reasons[category] += 1
    return {'questions_logged': len(logs), 'types': dict(types), 'abstain_reasons': dict(reasons),
            'quality_evaluated': False}
