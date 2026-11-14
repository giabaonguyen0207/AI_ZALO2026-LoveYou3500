"""Build-time installer: local wheels only; never called by predict.sh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig


def profile():
    return {'python': list(sys.version_info[:2]), 'system': sys.platform,
            'platform': sysconfig.get_platform(), 'machine': platform.machine(),
            'implementation': platform.python_implementation()}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def install(wheelhouse, manifest_path, extra='auto', project='.'):
    wheelhouse = Path(wheelhouse).resolve()
    solution = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    extra = solution.get('inference_extra', 'llm') if extra == 'auto' else extra
    selected = set(extra.split(','))
    if not selected <= {'llm', 'vllm', 'rag'} or len(selected & {'llm', 'vllm'}) != 1:
        raise ValueError('Unsupported inference extra')
    lock = wheelhouse / 'requirements.resolved.txt'
    sealed = wheelhouse / 'wheelhouse_manifest.json'
    if sealed.exists():
        manifest = json.loads(sealed.read_text(encoding='utf-8'))
        if manifest['identity']['extra'] != extra or manifest['identity']['profile'] != profile():
            raise ValueError('Wheelhouse backend/Python/platform differs from base image')
        if set(p.name for p in wheelhouse.glob('*.whl')) != set(manifest['wheels']):
            raise ValueError('Wheelhouse file list differs from manifest')
        for name, item in manifest['wheels'].items():
            path = wheelhouse / name
            if Path(name).name != name or path.is_symlink() or not path.resolve().is_relative_to(wheelhouse) or digest(path) != item['sha256']:
                raise ValueError('Invalid wheel hash/path: ' + name)
        if digest(lock) != manifest['requirements_sha256']:
            raise ValueError('Resolved wheel lock hash changed')
    command = [sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(wheelhouse)]
    if lock.exists():
        subprocess.run([*command, '--require-hashes', '--requirement', str(lock)], check=True)
    # Build dependencies are explicit even for a legacy wheelhouse without lock.
    subprocess.run([*command, 'setuptools', 'wheel'], check=True)
    subprocess.run([*command, '--no-build-isolation', str(Path(project).resolve()) + '[' + extra + ']'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
    return {'extra': extra, 'wheelhouse_verified': sealed.exists()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', default=os.environ.get('DATAMATE_WHEELHOUSE', '/wheelhouse'))
    parser.add_argument('--manifest', default=os.environ.get('DATAMATE_SOLUTION_MANIFEST', 'solution_manifest.json'))
    parser.add_argument('--extra', default=os.environ.get('INFERENCE_EXTRA', 'auto'))
    parser.add_argument('--project', default='.')
    args = parser.parse_args()
    print(json.dumps(install(args.wheelhouse, args.manifest, args.extra, args.project)))


if __name__ == '__main__':
    main()
