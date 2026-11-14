"""Online staging and offline reuse. Never called by the inference agent loop."""
from __future__ import annotations

from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import uuid
import zipfile

from datamate.domain import DataMateError
from datamate.models.artifacts import sha256, verify_artifact
from datamate.models.registry import get_model, inventory_from_settings, model_paths, relocate_models, inference_extras


def _run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _fresh_target(path):
    """Keep existing corrupt/incomplete/different artifacts intact."""
    path = Path(path)
    return path.with_name(path.name + '-repair-' + uuid.uuid4().hex[:12]) if path.exists() and any(path.iterdir()) else path


def python_profile(python):
    code = ('import json,sys,sysconfig,platform; print(json.dumps(dict('
            'python=list(sys.version_info[:2]),system=sys.platform,platform=sysconfig.get_platform(),'
            'machine=platform.machine(),implementation=platform.python_implementation())))')
    return json.loads(_run([str(python), '-c', code], capture_output=True, text=True).stdout)


def _ensure_pip(python):
    code = 'import importlib.util; print(int(importlib.util.find_spec("pip") is not None))'
    if _run([str(python), '-c', code], capture_output=True, text=True).stdout.strip() != '1':
        # Uses Python's bundled pip wheels; never downloads from ensurepip.
        _run([str(python), '-m', 'ensurepip', '--upgrade'])


def _requirements_hashes(path, seen=None):
    seen = {} if seen is None else seen
    path = Path(path).resolve()
    if str(path) in seen:
        return seen
    seen[str(path)] = sha256(path)
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        match = re.match(r'^\s*(?:-r\s*|--requirement[=\s]+|-c\s*|--constraint[=\s]+)(.+?)\s*(?:\s+#.*)?$', line)
        if match:
            _requirements_hashes(path.parent / match[1].strip('"\''), seen)
    return seen


def _canonical(name):
    return re.sub('[-_.]+', '-', name).lower()


def seal_wheels(directory, identity):
    directory = Path(directory).resolve()
    wheels, packages = {}, {}
    for path in sorted(directory.glob('*.whl')):
        if path.is_symlink():
            raise DataMateError('Wheel symlink không được phép')
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(names) != 1 or archive.getinfo(names[0]).file_size > 16_000_000:
                raise DataMateError(f'METADATA wheel không hợp lệ: {path.name}')
            metadata = BytesParser().parsebytes(archive.read(names[0]))
        name, version = metadata.get('Name'), metadata.get('Version')
        if (not name or not version or not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?', name)
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.+!_-]*', version)):
            raise DataMateError(f'Name/Version wheel không hợp lệ: {path.name}')
        key = _canonical(name)
        if key in packages:
            raise DataMateError('Wheelhouse có nhiều wheel cho cùng package: ' + name)
        digest = sha256(path)
        packages[key] = {'name': name, 'version': version, 'wheel': path.name, 'sha256': digest}
        wheels[path.name] = {'sha256': digest, 'bytes': path.stat().st_size}
    if not wheels:
        raise DataMateError('Wheelhouse rỗng')
    lock = directory / 'requirements.resolved.txt'
    lock.write_text(''.join(f"{p['name']}=={p['version']} --hash=sha256:{p['sha256']}\n"
                            for _, p in sorted(packages.items())), encoding='utf-8')
    manifest = {'format_version': 1, 'identity': identity, 'wheels': wheels, 'packages': packages,
                'requirements_sha256': sha256(lock)}
    _save(directory / 'wheelhouse_manifest.json', manifest)
    return manifest


def verify_wheels(directory, expected_identity=None):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'wheelhouse_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1 or expected_identity is not None and manifest.get('identity') != expected_identity:
        raise DataMateError('Wheelhouse identity khác requirements/Python/platform/backend')
    if set(p.name for p in directory.glob('*.whl')) != set(manifest.get('wheels', {})):
        raise DataMateError('Danh sách wheels khác manifest')
    for name, info in manifest['wheels'].items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or not path.resolve().is_relative_to(directory) or sha256(path) != info['sha256']:
            raise DataMateError('Wheel hash/path sai: ' + name)
    if sha256(directory / 'requirements.resolved.txt') != manifest.get('requirements_sha256'):
        raise DataMateError('Resolved requirements đã bị sửa')
    return manifest


def prepare_wheels(root, project, extra, python, *, requirements=None, index_url=None, extra_index_urls=()):
    project = Path(project).resolve()
    profile = python_profile(python)
    if 'vllm' in extra.split(',') and profile['system'] != 'linux':
        raise DataMateError('Chuẩn bị vLLM wheels trên Linux target environment; không tải GPU stack trên laptop Windows')
    metadata = tomllib.loads((project / 'pyproject.toml').read_text(encoding='utf-8'))
    direct = [*metadata['project']['dependencies'],
              *(requirement for name in extra.split(',') for requirement in metadata['project']['optional-dependencies'][name]),
              *metadata['build-system']['requires'], 'wheel']
    direct = list(dict.fromkeys(direct))
    identity = {'extra': extra, 'profile': profile,
                'project_metadata_sha256': sha256(project / 'pyproject.toml'),
                'requirements': _requirements_hashes(requirements) if requirements else direct,
                # Never persist index URLs, which can contain credentials.
                'index_options_sha256': hashlib.sha256(json.dumps([index_url, list(extra_index_urls),
                    os.environ.get('PIP_INDEX_URL'), os.environ.get('PIP_EXTRA_INDEX_URL')]).encode()).hexdigest()}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = Path(root).resolve() / 'wheelhouses' / key
    try:
        manifest = verify_wheels(target, identity)
        return target, manifest, 'hit'
    except (OSError, ValueError, KeyError, TypeError, DataMateError):
        target = _fresh_target(target)
    target.mkdir(parents=True, exist_ok=True)
    _ensure_pip(python)
    command = [str(python), '-m', 'pip', 'download', '--only-binary=:all:', '--dest', str(target)]
    command += direct
    if requirements:
        command += ['--requirement', str(Path(requirements).resolve())]
    if index_url:
        command += ['--index-url', index_url]
    for url in extra_index_urls:
        command += ['--extra-index-url', url]
    _run(command)
    return target, seal_wheels(target, identity), 'downloaded'


def install_libraries(directory, python, *, check_only=False):
    """Install missing/different versions from verified wheels, with no network."""
    directory = Path(directory).resolve()
    manifest = verify_wheels(directory)
    if python_profile(python) != manifest['identity']['profile']:
        raise DataMateError('Python/platform đích khác wheelhouse; stage lại trên đúng environment')
    expected = {p['name']: p['version'] for p in manifest['packages'].values()}
    code = ('import json,sys; from importlib.metadata import version,PackageNotFoundError; result={}; '
            '\nfor name in json.loads(sys.argv[1]):\n'
            ' try: result[name]=version(name)\n except PackageNotFoundError: result[name]=None\n'
            'print(json.dumps(result))')
    actual = json.loads(_run([str(python), '-c', code, json.dumps(list(expected))], capture_output=True, text=True).stdout)
    changed = {name: {'installed': actual.get(name), 'required': version}
               for name, version in expected.items() if actual.get(name) != version}
    if check_only:
        if changed:
            raise DataMateError('Thư viện thiếu/sai version: '+json.dumps(changed))
        return {'status': 'hit', 'changed': {}, 'check_only': True}
    if changed:
        _ensure_pip(python)
        _run([str(python), '-m', 'pip', 'install', '--no-index', '--find-links', str(directory),
              '--require-hashes', '--requirement', str(directory / 'requirements.resolved.txt')])
        after = json.loads(_run([str(python), '-c', code, json.dumps(list(expected))], capture_output=True, text=True).stdout)
        if after != expected:
            raise DataMateError('Thư viện sau cài chưa khớp resolved wheelhouse')
    _ensure_pip(python)
    _run([str(python), '-m', 'pip', 'check'])
    return {'status': 'installed' if changed else 'hit', 'changed': changed}


def prepare_model(root, project, python, key, candidate):
    root = Path(root).resolve()
    spec = get_model(key)
    parent = root / 'models' / key
    if not parent.resolve().is_relative_to(root):
        raise DataMateError('Model key không phải đường dẫn cache hợp lệ')
    source = Path(candidate).resolve() if candidate else None
    verified = None
    if source and source.is_dir() and any(source.iterdir()) and not (source / 'artifact_manifest.json').is_file():
        raise DataMateError('Checkpoint source không rỗng nhưng thiếu artifact_manifest.json; seal/verify trước, không thay bằng base')
    if source and (source / 'artifact_manifest.json').is_file():
        try:
            verified = verify_artifact(source, key)
        except (OSError, ValueError, KeyError, TypeError, DataMateError):
            # Never replace a damaged fine-tuned artifact with base weights.
            try:
                manifest = json.loads((source / 'artifact_manifest.json').read_text(encoding='utf-8'))
            except (ValueError, OSError):
                manifest = {}
            if not isinstance(manifest, dict):
                manifest = {}
            if manifest.get('lineage'):
                raise DataMateError('Artifact fine-tune bị lỗi: cung cấp bản đúng; không tự thay bằng base')
            own_base_cache = (source.parent == parent and
                (source.name == spec.revision or source.name.startswith(spec.revision + '-repair-')))
            if not own_base_cache:
                raise DataMateError('Checkpoint source manifest/hash không hợp lệ; không tự thay bằng base')
    target = parent / (sha256(source / 'artifact_manifest.json') if verified else spec.revision)
    if verified and source.is_relative_to(root):
        return source, 'hit'
    try:
        verify_artifact(target, key)
        if verified and sha256(target / 'artifact_manifest.json') != sha256(source / 'artifact_manifest.json'):
            raise DataMateError('Cache model khác artifact được yêu cầu')
        return target, 'hit'
    except (OSError, ValueError, KeyError, TypeError, DataMateError):
        target = _fresh_target(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if verified:
        if target == source or target.is_relative_to(source):
            raise DataMateError('Không copy artifact vào bên trong chính nó')
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('.cache'))
        status = 'copied'
    else:
        # Fresh interpreter: library changes made during preparation are visible.
        env = {**os.environ, 'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0',
               'PYTHONPATH': str(Path(project).resolve() / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')}
        _run([str(python), '-m', 'datamate', 'stage-model', '--model', key, '--output', str(target)], env=env)
        status = 'downloaded'
    verify_artifact(target, key)
    return target, status


def verify_bundle(path):
    path = Path(path).resolve()
    bundle = json.loads(path.read_text(encoding='utf-8'))
    if bundle.get('format_version') != 1:
        raise DataMateError('Bundle format không hỗ trợ')
    def local(relative):
        target = (path.parent / relative).resolve()
        if not target.is_relative_to(path.parent):
            raise DataMateError('Bundle path trỏ ra ngoài artifact root')
        return target
    wheels = local(bundle['wheelhouse'])
    wheel_manifest = verify_wheels(wheels)
    if sha256(wheels / 'wheelhouse_manifest.json') != bundle['wheelhouse_manifest_sha256']:
        raise DataMateError('Bundle wheelhouse manifest hash sai')
    for key, item in bundle['models'].items():
        folder = local(item['path'])
        verify_artifact(folder, key)
        if sha256(folder / 'artifact_manifest.json') != item['manifest_sha256']:
            raise DataMateError('Bundle model manifest hash sai: ' + key)
    config = local(bundle['config'])
    if sha256(config) != bundle['config_sha256']:
        raise DataMateError('Bundle config hash sai')
    from datamate.config import Settings
    settings = Settings.load(config, use_env=False)
    expected_inventory = inventory_from_settings(settings)
    if bundle.get('inventory') != expected_inventory or set(bundle['models']) != {m['key'] for m in expected_inventory['models']}:
        raise DataMateError('Bundle thiếu/thừa model hoặc inventory không khớp config')
    if wheel_manifest['identity']['extra'] != inference_extras(settings):
        raise DataMateError('Backend config không khớp dependency bundle')
    return bundle


def prepare_bundle(root, project, settings, python, *, requirements=None, install=False,
                   index_url=None, extra_index_urls=()):
    from datamate.data.split import ensure_outside_dataset
    root = ensure_outside_dataset(root, settings.split_root)
    settings.validate()
    specs = inventory_from_settings(settings)
    extra = inference_extras(settings)
    wheelhouse, _, wheel_status = prepare_wheels(root, project, extra, python,
        requirements=requirements, index_url=index_url, extra_index_urls=extra_index_urls)
    libraries = install_libraries(wheelhouse, python) if install else {'status': 'not_installed'}
    models, statuses, sources, staged = {}, {}, model_paths(settings), {}
    for spec in specs['models']:
        key = spec['key']
        candidate = sources[key]
        folder, statuses[key] = prepare_model(root, project, python, key, candidate)
        models[key] = {'path': folder.relative_to(root).as_posix(),
                       'manifest_sha256': sha256(folder / 'artifact_manifest.json')}
        staged[key] = str(folder)
    relocate_models(settings, staged)
    config_name = 'runtime-' + hashlib.sha256(json.dumps(settings.to_dict(), sort_keys=True).encode()).hexdigest()[:16] + '.json'
    _save(root / config_name, settings.to_dict())
    bundle = {'format_version': 1, 'inventory': specs, 'models': models,
              'wheelhouse': wheelhouse.relative_to(root).as_posix(),
              'wheelhouse_manifest_sha256': sha256(wheelhouse / 'wheelhouse_manifest.json'),
              'config': config_name, 'config_sha256': sha256(root / config_name),
              'preparation': {'models': statuses, 'wheelhouse': wheel_status, 'libraries': libraries},
              'gpu_smoke_verified': False, 'docker_verified': False}
    _save(root / 'bundle.json', bundle)
    verify_bundle(root / 'bundle.json')
    return bundle
