"""Assemble explicit, hashed training evidence. Never downloads or trains models.

The input spec is local (usually under outputs); source paths are not published.
Only declared files are copied. Keep frozen sources and original reports intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / 'src')]
from datamate.models.artifacts import sha256

REQUIRED_ROLES = {'source', 'data_generation', 'train', 'validation', 'logs',
                  'configuration', 'checkpoints', 'dependencies', 'reproduce', 'documentation'}


def relative(value):
    if not isinstance(value, str) or '\\' in value or ':' in value:
        raise ValueError('Evidence path must be a relative POSIX path')
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or '..' in path.parts:
        raise ValueError('Evidence path escapes its root')
    return path.as_posix()


def _validate(manifest):
    files = manifest.get('files', {})
    if manifest.get('format_version') != 1 or not isinstance(files, dict):
        raise ValueError('Invalid training evidence manifest')
    roles = {v.get('role') for v in files.values()}
    if REQUIRED_ROLES - roles:
        raise ValueError('Missing training evidence roles: ' + str(sorted(REQUIRED_ROLES - roles)))
    for name, info in files.items():
        relative(name)
        if not re.fullmatch('[0-9a-f]{64}', info.get('sha256', '')) or type(info.get('bytes')) is not int:
            raise ValueError('Invalid evidence file hash/size: ' + name)
    for stage in manifest.get('training_stages', []):
        for split in ('train', 'validation'):
            name = stage.get(split)
            if name not in files or files[name]['role'] != split or files[name].get('records', 0) < 1:
                raise ValueError('Training stage must reference local nonempty ' + split + ' data')
    if not manifest.get('training_stages'):
        raise ValueError('Declare training stages and data splits')
    for base in manifest.get('training_only_models', []):
        revision = base.get('revision', '')
        if (not re.fullmatch('[0-9a-f]{40}', revision) or not base.get('license')
                or not base.get('revision_date') or base['revision_date'][:10] >= '2026-10-01'
                or not base.get('files')):
            raise ValueError('Training-only base requires eligible fixed revision, license and file hashes')
        for name, info in base['files'].items():
            relative(name)
            if not re.fullmatch('[0-9a-f]{64}', info.get('sha256', '')) or ('/resolve/' + revision + '/') not in info.get('url', ''):
                raise ValueError('Training-only base files need revision-pinned URLs and SHA256')


def _records(path):
    count = 0
    with path.open(encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                json.loads(line)
                count += 1
    return count


def assemble(spec, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Existing evidence is not overwritten; choose a new directory')
    declared = {}
    for entry in spec.get('entries', []):
        source = Path(entry['source'])
        destination = relative(entry['path'])
        if source.is_symlink() or not source.exists():
            raise ValueError('Missing or symlink evidence source: ' + str(source))
        paths = sorted(source.rglob('*')) if source.is_dir() else [source]
        for file in paths:
            if file.is_symlink():
                raise ValueError('Evidence symlinks are not supported')
            if not file.is_file() or '__pycache__' in file.parts:
                continue
            name = relative(destination + '/' + file.relative_to(source).as_posix()) if source.is_dir() else destination
            if name == 'data_manifest.json' or name in declared:
                raise ValueError('Duplicate/reserved evidence path: ' + name)
            digest = sha256(file)
            if entry.get('sha256') and (source.is_dir() or entry['sha256'] != digest):
                raise ValueError('Expected evidence hash differs: ' + name)
            info = {'role':entry['role'], 'bytes':file.stat().st_size, 'sha256':digest}
            if file.suffix == '.jsonl':
                info['records'] = _records(file)
            declared[name] = (file, info)
    manifest = {'format_version':1, 'files':{n:v[1] for n,v in declared.items()},
                'training_stages':spec.get('training_stages', []),
                'training_only_models':spec.get('training_only_models', []),
                'inference_artifacts':spec.get('inference_artifacts', {}),
                'notes':spec.get('notes', [])}
    _validate(manifest)
    for name, (source, _) in declared.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (output / 'data_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return verify(output)


def verify(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / 'data_manifest.json').read_text(encoding='utf-8'))
    _validate(manifest)
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Evidence contains a symlink')
        if path.is_file() and path != root / 'data_manifest.json':
            actual.add(path.relative_to(root).as_posix())
    if actual != set(manifest['files']):
        raise ValueError('Training evidence files differ from manifest')
    for name, info in manifest['files'].items():
        path = root / relative(name)
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError('Training evidence hash/size differs: ' + name)
        if 'records' in info and _records(path) != info['records']:
            raise ValueError('Training evidence record count differs: ' + name)
    return manifest


def with_dependencies(source, wheelhouse, output):
    """Create a new sealed copy after target-Linux offline wheels are prepared."""
    from tools.runtime_artifacts import verify_wheels
    source, wheelhouse = Path(source).resolve(), Path(wheelhouse).resolve()
    manifest = verify(source)
    wheels = verify_wheels(wheelhouse)
    if wheels['identity']['profile']['system'] != 'linux':
        raise ValueError('Prepare training wheels on target Linux Python')
    entries = [{'source':str(source/name), 'path':name, 'role':info['role'], 'sha256':info['sha256']}
               for name, info in manifest['files'].items()]
    entries.append({'source':str(wheelhouse), 'path':'wheelhouse', 'role':'dependencies'})
    spec = {**manifest, 'entries':entries,
            'notes':[*manifest.get('notes', []), 'Target-Linux offline training wheels attached; actual retraining not performed.']}
    return assemble(spec,output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--source', type=Path, help='Existing source/data evidence to copy with target-Linux wheels')
    parser.add_argument('--with-wheelhouse', type=Path)
    args = parser.parse_args()
    if args.source and args.with_wheelhouse and args.output and not args.verify and not args.spec:
        manifest = with_dependencies(args.source,args.with_wheelhouse,args.output)
    elif args.verify and not args.spec and not args.output and not args.source:
        manifest = verify(args.verify)
    elif args.spec and args.output and not args.verify:
        manifest = assemble(json.loads(args.spec.read_text(encoding='utf-8-sig')), args.output)
    else:
        parser.error('Choose --spec SPEC --output NEW_DIR, --verify DIR, or --source DIR --with-wheelhouse WHEELS --output NEW_DIR')
    print(json.dumps({'files':len(manifest['files']), 'stages':manifest['training_stages']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
