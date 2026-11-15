"""Create a small reproducible, label-free benchmark from the participant package."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import zipfile


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def prepare(participant, output, *, size=24, businesses=6, seed=20261112):
    participant, output = Path(participant).resolve(), Path(output).resolve()
    dataset = participant / 'dataset'
    if output == participant or output.is_relative_to(participant):
        raise ValueError('Output must be outside the original participant package')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new/empty output directory; frozen datasets are not overwritten')
    archive_path = output.parent / (output.name + '.zip')
    if archive_path.exists():
        raise ValueError('Choose a new archive filename: ' + str(archive_path))
    if size < businesses or businesses < 2 or size % businesses:
        raise ValueError('size must be divisible by businesses, with at least two businesses')
    source_manifest = json.loads((participant / 'benchmark_manifest.json').read_text(encoding='utf-8-sig'))
    questions = [json.loads(line) for line in (dataset / 'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if digest(dataset / 'questions.jsonl') != source_manifest['questions_sha256']:
        raise ValueError('Participant question list differs from its original manifest')
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    by_business = defaultdict(list)
    for question in questions:
        if set(question) != {'id', 'business_id', 'question', 'as_of_date'}:
            raise ValueError('Only four input fields are allowed; labels/predictions must stay separate')
        key = question['business_id']
        if Path(key).name != key or key in {'.', '..'} or '\\' in key:
            raise ValueError('Invalid business ID')
        by_business[key].append(question)
    per_business = size // businesses
    # Reserve two extension questions per shop and two pilot questions for the
    # smallest/largest metadata strata. Eligibility is identical for all shops.
    required = per_business + 4
    candidates = []
    for key, rows in sorted(by_business.items()):
        if len(rows) < required:
            continue
        folder = dataset / 'data' / key
        files = [p for p in folder.rglob('*') if p.is_file()]
        readme = (folder / 'README.md').read_text(encoding='utf-8-sig')
        table_count = sum(p.suffix.lower() in {'.csv', '.xlsx', '.db', '.sqlite'} for p in files)
        candidates.append({'business_id': key, 'available_questions': len(rows), 'readme_chars': len(readme),
                           'table_files': table_count, 'data_bytes': sum(p.stat().st_size for p in files)})
    candidates.sort(key=lambda r: (r['table_files'], r['readme_chars'], r['data_bytes'], r['business_id']))
    if len(candidates) < businesses:
        raise ValueError('Not enough businesses with sufficient disjoint questions')
    chosen = [candidates[round(i * (len(candidates)-1) / (businesses-1))] for i in range(businesses)]
    main, pilot, reserve = [], [], []
    rng = random.Random(seed)
    for index, item in enumerate(chosen):
        rows = sorted(by_business[item['business_id']], key=lambda q: q['id'])
        rng.shuffle(rows)
        main.extend(rows[:per_business])
        reserve.extend(rows[per_business:per_business+2])
        if index in {0, businesses-1}:
            pilot.extend(rows[per_business+2:per_business+4])
    all_rows = sorted(main + pilot + reserve, key=lambda q: (q['business_id'], q['id']))
    if len({q['id'] for q in all_rows}) != len(all_rows):
        raise ValueError('Benchmark arms must be disjoint')
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'dataset'
    target.mkdir()
    copied = {}
    for name in ('rulebook.md', 'calendar.csv'):
        if digest(dataset/name) != source_manifest['files'].get('dataset/'+name):
            raise ValueError('Shared input differs from source manifest: '+name)
        shutil.copy2(dataset / name, target / name)
        copied['dataset/' + name] = digest(target / name)
    for item in chosen:
        source = dataset / 'data' / item['business_id']
        for path in sorted(source.rglob('*')):
            if path.is_symlink():
                raise ValueError('Symlink in source dataset')
            if not path.is_file():
                continue
            if path.suffix.lower() not in {'.csv', '.xlsx', '.db', '.sqlite'} and path.name != 'README.md':
                continue  # Never copy participant predictions, gold, or arbitrary output files.
            relative = path.relative_to(dataset)
            expected = source_manifest['files'].get('dataset/' + relative.as_posix())
            if not expected:
                raise ValueError('Data file missing from original manifest: ' + str(relative))
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            actual = digest(destination)
            if actual != expected:
                raise ValueError('Data differs from original manifest: ' + str(relative))
            copied['dataset/' + relative.as_posix()] = actual
    (target / 'questions.jsonl').write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in all_rows), encoding='utf-8')
    copied['dataset/questions.jsonl'] = digest(target / 'questions.jsonl')
    for name, rows in (('main_ids.json', main), ('pilot_ids.json', pilot), ('reserve_ids.json', reserve)):
        ids = [q['id'] for q in sorted(rows, key=lambda q: (q['business_id'], q['id']))]
        save(output / name, ids)
        copied[name] = digest(output / name)
    report = {'benchmark_id': f'kaggle-runtime-{size}-seed{seed}', 'seed': seed,
              'source_benchmark': source_manifest['benchmark_id'], 'source_questions_sha256': digest(dataset/'questions.jsonl'),
              'main_size': len(main), 'pilot_size': len(pilot), 'reserve_size': len(reserve),
              'references_used_for_selection': False, 'sampling': 'metadata-stratified repeated-business diagnostic subset',
              'selection_method': 'eligible shops sorted by (table_files,readme_chars,data_bytes,id); equally spaced rank strata; seeded sample of sorted IDs',
              'population_warning': 'Not a uniform sample of 200 or 3000; not a general competition accuracy estimate',
              'eligibility': f'at least {required} questions per business', 'businesses': chosen,
              'questions_total': len(all_rows), 'files': copied}
    save(output / 'benchmark_manifest.json', report)
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    save(Path(str(archive_path) + '.manifest.json'), {'sha256': digest(archive_path), 'benchmark': report['benchmark_id']})
    return report


def freeze_baseline(project, reference, output):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Baseline archive already exists')
    commit = subprocess.run(['git', 'rev-parse', reference + '^{commit}'], cwd=project,
                            check=True, capture_output=True, text=True).stdout.strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project))
    from tools.package_colab import source_path_allowed
    payload = subprocess.run(['git', 'archive', '--format=zip', commit], cwd=project,
                             check=True, capture_output=True).stdout
    hashes = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as snapshot, zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for entry in snapshot.infolist():
            if entry.is_dir() or not source_path_allowed(entry.filename):
                continue
            content = snapshot.read(entry)
            archive.writestr(entry.filename, content)
            hashes[entry.filename] = hashlib.sha256(content).hexdigest()
        archive.comment = json.dumps({'git_commit': commit, 'source_policy': 'shared-source-only'}).encode()
    save(Path(str(output) + '.manifest.json'), {'archive': output.name, 'commit': commit,
         'sha256': digest(output), 'source_only': True, 'file_count': len(hashes), 'files': hashes})
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=24)
    parser.add_argument('--businesses', type=int, default=6)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--baseline-ref')
    parser.add_argument('--baseline-output', type=Path)
    args = parser.parse_args()
    report = prepare(args.participant, args.output, size=args.size, businesses=args.businesses, seed=args.seed)
    if args.baseline_ref:
        if args.baseline_output is None:
            parser.error('--baseline-output is required with --baseline-ref')
        report['baseline_commit'] = freeze_baseline(Path(__file__).resolve().parents[1], args.baseline_ref, args.baseline_output)
    print(json.dumps({k: report[k] for k in ('benchmark_id','main_size','pilot_size','reserve_size','businesses')}, indent=2))


if __name__ == '__main__':
    main()
