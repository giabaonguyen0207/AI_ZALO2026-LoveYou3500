"""Integrity checks for the versioned Qwen SQL SFT release (no GPU imports)."""
from __future__ import annotations
import json
from pathlib import Path
import stat
import zipfile

from ..agent.protocol import parse_action
from ..data.split import read_jsonl
from ..domain import DataMateError
from ..models.artifacts import sha256
from ..models.registry import get_model


def verify_sql_dataset(directory, *, verify_source=True):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='verified_sql_sft' or manifest.get('source_role')!='competition_training':
        raise DataMateError('Not a TRAIN SQL SFT release')
    if manifest.get('target_model')!=get_model('qwen35-9b').to_dict():
        raise DataMateError('SQL dataset checkpoint/revision differs from pinned Qwen')
    train_b=set(manifest['train_businesses']);val_b=set(manifest['validation_businesses'])
    if train_b & val_b:raise DataMateError('Overlapping business partitions')
    for name,digest in manifest['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path)!=digest:
            raise DataMateError('SQL dataset file hash mismatch: '+name)
    if verify_source:
        root=Path(__file__).resolve().parents[1]
        for name,digest in manifest['runtime_source_hashes'].items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha256(path)!=digest:
                raise DataMateError('SQL dataset/source release mismatch: '+name)
    seen=set()
    for partition,allowed in [('train',train_b),('validation',val_b)]:
        rows=read_jsonl(directory/(partition+'.jsonl'))
        if len(rows)!=manifest[partition+'_examples']:raise DataMateError('SQL dataset row count mismatch')
        for row in rows:
            if row['question_id'] in seen or row['business_id'] not in allowed:raise DataMateError('Duplicate ID or partition leak')
            seen.add(row['question_id'])
            messages=row['messages']
            if [m.get('role') for m in messages[:2]]!=['system','user']:raise DataMateError('SQL dataset has invalid prompt roles')
            request=json.loads(messages[1]['content'])['request']
            if set(request)!={'question','as_of_date'}:raise DataMateError('Unexpected identifiers/labels in request')
            actions=[parse_action(m['content']) for m in messages if m['role']=='assistant']
            if not actions or any('answer' in a for a in actions):raise DataMateError('Literal final answer in SQL target')
            contains_sql=any(a.get('action')=='sql' or a.get('query',{}).get('sql') for a in actions)
            if not contains_sql and actions[-1].get('type')!='unanswerable':raise DataMateError('Answerable sample lacks SQL target')
    return manifest


def unpack_sql_archive(archive,destination,release):
    archive=Path(archive);destination=Path(destination);release=json.loads(Path(release).read_text(encoding='utf-8'))
    if sha256(archive)!=release['dataset_archive_sha256']:raise DataMateError('SQL dataset ZIP checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(set(names))!=len(names) or 'manifest.json' not in names:raise DataMateError('Duplicate/missing ZIP entries')
        if sum(i.file_size for i in infos)>1024**3:raise DataMateError('Dataset ZIP exceeds 1 GiB unpack limit')
        for item in infos:
            name=item.filename
            if name!=Path(name).name or ':' in name or '\\' in name or stat.S_ISLNK(item.external_attr>>16):
                raise DataMateError('Unsafe SQL dataset ZIP path')
        manifest=json.loads(z.read('manifest.json'))
        if set(names)!=set(manifest['files'])|{'manifest.json'}:raise DataMateError('SQL dataset ZIP inventory mismatch')
        import hashlib
        if hashlib.sha256(z.read('manifest.json')).hexdigest()!=release['dataset_manifest_sha256']:
            raise DataMateError('SQL dataset manifest does not match release')
        for name,digest in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise DataMateError('SQL ZIP member hash mismatch')
        if destination.exists() and any(destination.iterdir()):
            verify_sql_dataset(destination)
            if sha256(destination/'manifest.json')!=release['dataset_manifest_sha256']:raise DataMateError('Existing dataset differs; use new directory')
        else:
            destination.mkdir(parents=True,exist_ok=True);z.extractall(destination)
    return verify_sql_dataset(destination)
