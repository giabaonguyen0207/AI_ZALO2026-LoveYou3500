"""Checks for source packaging and the maintained GPU entry point."""
import json
from pathlib import Path
import zipfile

import pytest

from tools.package_colab import package


def test_gpu_source_package_excludes_archives_data_models_and_history(tmp_path):
    root = tmp_path / 'project'
    files = ['pyproject.toml', '.dockerignore', 'requirements.txt', 'requirements-dev.lock.txt', 'requirements-colab.txt',
             'README.md', 'AGENTS.md', 'predict.sh', 'src/datamate/__init__.py', 'docs/run-gpu.md',
             'docs/history/old.md', 'docs/references/paper.pdf', 'archive/private.txt',
             'models/weights.bin', 'train/answers.jsonl', 'outputs/run/log.json', 'src/datamate/__pycache__/x.pyc']
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    output = tmp_path / 'source.zip'
    report = package(root, output)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert 'src/datamate/__init__.py' in names and 'docs/run-gpu.md' in names
    assert '.dockerignore' in names  # package_solution also runs from the GPU source ZIP.
    assert not any(n.startswith(('archive/', 'models/', 'train/', 'outputs/', 'docs/history/', 'docs/references/')) for n in names)
    assert not any('__pycache__' in n for n in names)
    assert json.loads(output.with_suffix('.zip.manifest.json').read_text())['sha256'] == report['sha256']


def test_maintained_inference_notebook_is_generated_and_compiles():
    from tools.create_inference_notebook import SETUP, INSTALL, PREPARE, RUN
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / 'notebooks/DataMate_Colab_Inference.ipynb').read_text(encoding='utf-8'))
    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
    assert [''.join(c['source']) for c in code] == [SETUP, INSTALL, PREPARE, RUN]
    for cell in code:
        assert cell['outputs'] == []
        compile(''.join(cell['source']), cell['id'], 'exec')


def test_cli_has_no_retired_web_command():
    from datamate.cli import parser
    with pytest.raises(SystemExit):
        parser().parse_args(['serve'])


def test_gpu_setup_separates_source_versions_and_rejects_stale_kernel(tmp_path, monkeypatch):
    import os
    import sys
    import types
    from tools.create_inference_notebook import SETUP
    packet = tmp_path / 'source.zip'
    data = tmp_path / 'data'
    data.mkdir()
    monkeypatch.setenv('DATAMATE_PLATFORM','gpu')
    code = SETUP
    replacements = {'/work/datamate': tmp_path / 'runtime', '/work/datamate-results': tmp_path / 'saved',
                    '/work/inputs/datamate-colab-source.zip': packet, '/work/data/train': data}
    for original, value in replacements.items():
        code = code.replace('Path(' + repr(original).replace("'", '"') + ')', 'Path(' + repr(str(value)) + ')')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.delitem(sys.modules, 'datamate', raising=False)
    for name in ('HF_HOME', 'TORCH_HOME', 'PIP_CACHE_DIR'):
        monkeypatch.setenv(name, os.environ.get(name, ''))
    with zipfile.ZipFile(packet, 'w') as archive:
        archive.writestr('src/datamate/__init__.py', '# version one')
    first = {}
    exec(compile(code, 'gpu-setup-fixture', 'exec'), first)
    old = first['WORK']
    monkeypatch.setitem(sys.modules, 'datamate', types.SimpleNamespace(__file__=str(old / 'src/datamate/__init__.py')))
    with zipfile.ZipFile(packet, 'w') as archive:
        archive.writestr('src/datamate/__init__.py', '# version two')
    with pytest.raises(RuntimeError, match='restart the kernel'):
        exec(compile(code, 'gpu-setup-fixture', 'exec'), {})
    monkeypatch.delitem(sys.modules, 'datamate')
    second = {}
    exec(compile(code, 'gpu-setup-fixture', 'exec'), second)
    assert second['WORK'] != old
    assert (old / 'src/datamate/__init__.py').read_text() == '# version one'
    assert (second['WORK'] / 'src/datamate/__init__.py').read_text() == '# version two'


@pytest.mark.parametrize('backend', ['transformers', 'vllm'])
def test_solution_staging_copies_only_declared_models_and_runtime_files(tmp_path, monkeypatch, backend):
    import struct
    from datamate.domain import json_text
    from datamate.models.registry import MODELS, ModelSpec
    from datamate.models.artifacts import seal_artifact
    from tools.package_solution import stage
    project = Path(__file__).resolve().parents[1]
    spec = ModelSpec('fixture', 'fixture/checkpoint', 'a' * 40, 3, '2026-01-01T00:00:00Z')
    monkeypatch.setitem(MODELS, 'fixture', spec)
    model = tmp_path / 'checkpoint'
    model.mkdir()
    header = json.dumps({'tensor': {'shape': [3], 'dtype': 'BF16', 'data_offsets': [0, 6]}}).encode()
    (model / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 6)
    seal_artifact(model, spec)
    wheelhouse = tmp_path / 'wheelhouse'
    wheelhouse.mkdir()
    (wheelhouse / 'fixture.whl').write_bytes(b'fixture, not an installable wheel')
    config = tmp_path / 'settings.json'
    config.write_text(json_text({'model_key': 'fixture', 'backend': backend}), encoding='utf-8')
    output = tmp_path / 'context'
    report = stage(project, config, output, wheelhouse, {'fixture': model}, base_image='team/base@sha256:fixture')
    assert report['inventory']['total_parameters'] == 3 and report['image_built'] is False
    assert report['inference_extra'] == ('vllm' if backend == 'vllm' else 'llm')
    assert report['base_image'] == 'team/base@sha256:fixture'
    assert report['base_image_policy'] == 'team_selected' and report['base_image_gpu_verified'] is False
    assert (output / 'src/datamate/training/finetune.py').exists()
    assert not any((output / name).exists() for name in ('archive', 'train', 'public_test', 'outputs', 'tests'))
    assert list((output / 'models').iterdir()) == [output / 'models/fixture']
    with pytest.raises(ValueError, match='empty'):
        stage(project, config, output, wheelhouse, {'fixture': model})
