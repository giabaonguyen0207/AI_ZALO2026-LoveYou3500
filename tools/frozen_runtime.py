"""Validate an existing pip freeze without importing GPU libraries or installing."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import tomllib


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def read_freeze(path):
    pins = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)', line)
        if not match:
            raise ValueError('Freeze must contain exact name==version pins: ' + line)
        name, version = canonical(match[1]), match[2]
        if name in pins and pins[name] != version:
            raise ValueError('Conflicting freeze versions: ' + name)
        pins[name] = version
    if not pins:
        raise ValueError('Empty runtime freeze')
    return pins


def prepare_requirements(freeze, project, destination):
    project = Path(project)
    pins = read_freeze(freeze)
    required = read_freeze(project/'configs/docker-runtime-constraints.txt')
    for name, version in required.items():
        if pins.get(name) != version:
            raise ValueError('Freeze differs from existing Docker pin: ' + name)
    metadata = tomllib.loads((project/'pyproject.toml').read_text(encoding='utf-8'))['project']
    own = canonical(metadata['name'])
    if own in pins and pins[own] != metadata['version']:
        raise ValueError('Source project version differs from freeze')
    # datamate-agent is built from the current working tree by install_runtime;
    # never ask the package index for the team's private project distribution.
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(''.join(f'{name}=={version}\n' for name,version in sorted(pins.items()) if name != own),
                           encoding='utf-8')
    return destination


def check_runtime(freeze, *, skip_project=False, version_lookup=None):
    lookup = version_lookup or importlib.metadata.version
    expected = read_freeze(freeze)
    differences = {}
    for name, version in expected.items():
        if skip_project and name == 'datamate-agent':
            continue
        try:
            actual = lookup(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            differences[name] = {'expected': version, 'actual': actual}
    if differences:
        raise ValueError('Runtime differs from freeze: ' + json.dumps(differences, sort_keys=True))
    return {'checked_packages': len(expected) - int(skip_project and 'datamate-agent' in expected),
            'versions_match': True, 'gpu_tested': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_runtime(args.freeze), indent=2))
