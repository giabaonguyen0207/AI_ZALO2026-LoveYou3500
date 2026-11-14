import json
from pathlib import Path

import pytest

from tools.training_evidence import assemble, verify


def evidence_spec(tmp_path):
    source = tmp_path / 'inputs'
    source.mkdir()
    files = {
        'source/train.py': 'print("training source")\n',
        'generation/build.py': 'print("data generation")\n',
        'data/train.jsonl': '{"text":"Tiếng Việt"}\n',
        'data/validation.jsonl': '{"text":"holdout"}\n',
        'logs/training_report.json': '{"global_step":198}\n',
        'configs/training.json': '{}\n',
        'checkpoints/adapter.safetensors': 'fixture checkpoint',
        'requirements.txt': 'fixturelib==1.0\n',
        'train.sh': '#!/bin/sh\nset -eu\npython source/train.py\n',
        'README.md': 'Reproduction and frozen provenance.\n',
    }
    roles = {'source':'source', 'generation':'data_generation', 'logs':'logs',
             'configs':'configuration', 'checkpoints':'checkpoints',
             'requirements.txt':'dependencies', 'train.sh':'reproduce', 'README.md':'documentation'}
    entries = []
    for name, content in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8', newline='\n')
        role = ('train' if name.endswith('/train.jsonl') else 'validation') if name.startswith('data/') else roles[name.split('/')[0]]
        entries.append({'source':str(path), 'path':name, 'role':role})
    return {'format_version':1, 'entries':entries, 'training_only_models':[],
            'training_stages':[{'name':'fixture', 'train':'data/train.jsonl',
                                'validation':'data/validation.jsonl'}]}


def test_manifest_covers_real_local_data_counts_hashes_and_fails_on_tampering(tmp_path):
    spec = evidence_spec(tmp_path)
    root = tmp_path / 'evidence'
    manifest = assemble(spec, root)
    assert manifest['files']['data/train.jsonl']['records'] == 1
    assert manifest['files']['data/train.jsonl']['role'] == 'train'
    assert manifest['files']['data/train.jsonl']['bytes'] == len('{"text":"Tiếng Việt"}\n'.encode())
    assert verify(root) == manifest
    (root / 'data/train.jsonl').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='hash|size'):
        verify(root)


@pytest.mark.parametrize('problem', ['missing_role', 'escape', 'unlisted', 'unpinned_base'])
def test_rejects_incomplete_or_unsafe_evidence(tmp_path, problem):
    spec = evidence_spec(tmp_path)
    root = tmp_path / 'evidence'
    if problem == 'missing_role':
        spec['entries'] = [e for e in spec['entries'] if e['role'] != 'validation']
    elif problem == 'escape':
        spec['entries'][0]['path'] = '../escaped.py'
    elif problem == 'unpinned_base':
        spec['training_only_models'] = [{'repo_id':'Qwen/example', 'revision':'main'}]
    if problem != 'unlisted':
        with pytest.raises(ValueError):
            assemble(spec, root)
    else:
        assemble(spec, root)
        (root / 'secret.txt').write_text('not declared')
        with pytest.raises(ValueError, match='files'):
            verify(root)
    assert not (tmp_path / 'escaped.py').exists()


def test_expected_hash_mismatch_is_rejected_before_copy(tmp_path):
    spec = evidence_spec(tmp_path)
    spec['entries'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='hash'):
        assemble(spec, tmp_path / 'evidence')
