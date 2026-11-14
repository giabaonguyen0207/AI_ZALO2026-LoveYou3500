import json
from pathlib import Path
import struct
import types

import pytest

from datamate.models.artifacts import seal_artifact, verify_artifact
from datamate.models.registry import MODELS, ModelSpec
from datamate.domain import DataMateError
from tools.colab_model_cache import prepare_local_model


@pytest.fixture
def model_source(tmp_path, monkeypatch):
    source = tmp_path / 'drive/model'
    source.mkdir(parents=True)
    spec = ModelSpec('fixture', 'fixture/model', 'a'*40, 3, '2026-01-01T00:00:00Z')
    monkeypatch.setitem(MODELS, 'fixture', spec)
    header = json.dumps({'a':{'shape':[3], 'dtype':'BF16', 'data_offsets':[0,6]}}).encode()
    (source/'model.safetensors').write_bytes(struct.pack('<Q', len(header))+header+b'\0'*6)
    (source/'config.json').write_text('{}')
    seal_artifact(source, spec)
    return source


def test_copy_reuses_cache_and_backend_still_rejects_same_size_corruption(model_source, tmp_path, monkeypatch):
    local = prepare_local_model(model_source, tmp_path/'local', 'fixture')
    assert local != model_source
    assert verify_artifact(local, 'fixture') == verify_artifact(model_source, 'fixture')
    original = (model_source/'model.safetensors').read_bytes()
    def forbidden_copy(*args, **kwargs):
        raise AssertionError('A cache hit must not recopy weights')
    monkeypatch.setattr('tools.colab_model_cache.shutil.copyfileobj', forbidden_copy)
    assert prepare_local_model(model_source, tmp_path/'local', 'fixture') == local
    (local/'model.safetensors').write_bytes(original[:-1]+b'x')
    assert prepare_local_model(model_source, tmp_path/'local', 'fixture') == local
    with pytest.raises(DataMateError, match='Hash'):
        verify_artifact(local, 'fixture')
    assert (model_source/'model.safetensors').read_bytes() == original


def test_interrupted_copy_can_resume_without_publishing_partial_artifact(model_source, tmp_path, monkeypatch):
    import shutil
    original = shutil.copyfileobj
    def interrupted(src, dst, **kwargs):
        dst.write(src.read(1))
        raise OSError('interrupted')
    monkeypatch.setattr('tools.colab_model_cache.shutil.copyfileobj', interrupted)
    with pytest.raises(OSError, match='interrupted'):
        prepare_local_model(model_source, tmp_path/'local', 'fixture')
    assert not list((tmp_path/'local').rglob('artifact_manifest.json'))
    monkeypatch.setattr('tools.colab_model_cache.shutil.copyfileobj', original)
    local = prepare_local_model(model_source, tmp_path/'local', 'fixture')
    verify_artifact(local, 'fixture')


def test_new_manifest_uses_separate_cache_and_insufficient_disk_stops(model_source, tmp_path, monkeypatch):
    local = prepare_local_model(model_source, tmp_path/'local', 'fixture')
    (model_source/'config.json').write_text('{"changed":true}')
    seal_artifact(model_source, MODELS['fixture'])
    updated = prepare_local_model(model_source, tmp_path/'local', 'fixture')
    assert updated != local
    verify_artifact(updated, 'fixture')
    monkeypatch.setattr('tools.colab_model_cache.shutil.disk_usage', lambda _: types.SimpleNamespace(free=0))
    with pytest.raises(OSError, match='không đủ chỗ'):
        prepare_local_model(model_source, tmp_path/'empty', 'fixture')


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'dir\\escape', 'C:/escape'])
def test_manifest_cannot_escape_cache(model_source, tmp_path, name):
    p=model_source/'artifact_manifest.json'
    manifest=json.loads(p.read_text())
    manifest['files'][name]={'bytes':0,'sha256':'0'*64}
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        prepare_local_model(model_source, tmp_path/'local', 'fixture')


def test_gpu_notebook_prepares_without_loading_then_runs_shared_batch_once(model_source, tmp_path, monkeypatch):
    import runpy
    import sys
    import datamate.application
    import datamate.execution
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/DataMate_Colab_Inference.ipynb').read_text(encoding='utf-8'))
    from tools.create_inference_notebook import PREPARE, RUN
    assert ''.join(next(c for c in notebook['cells'] if c.get('id')=='gpu-prepare')['source']) == PREPARE
    assert ''.join(next(c for c in notebook['cells'] if c.get('id')=='gpu-run')['source']) == RUN
    calls=[]
    def batch(settings, **kwargs):
        calls.append(settings)
        assert Path(settings.model_dir).is_relative_to(tmp_path/'local')
        verify_artifact(settings.model_dir, settings.model_key)
        return {'written': 1}
    monkeypatch.setattr(datamate.execution, 'run_isolated', batch)
    real_run_path=runpy.run_path
    monkeypatch.setattr(runpy, 'run_path', lambda path: real_run_path(path) if Path(path).name=='colab_runtime_check.py' else
                        {'prepare_local_model':lambda src, cache, key:prepare_local_model(src, tmp_path/'local', key)})
    context={'WORK':root,'MODEL_SOURCE_DIR':model_source,'MODEL_KEY':'fixture', 'QUANTIZATION':'none',
             'TRAIN_ROOT':tmp_path/'train','SAVE':tmp_path/'saved',
             'COPY_MODEL_TO_SCRATCH':True, 'CONFIG_FILE':root/'configs/qwen35-9b.json',
             'IDS_FILE':None, 'RUN_LIMIT':None, 'RESUME':False}
    exec(compile(PREPARE, 'gpu-prepare', 'exec'), context)
    assert not calls
    exec(compile(RUN, 'gpu-run', 'exec'), context)
    assert len(calls)==1
    settings=calls[0]
    assert settings.quantization=='none' and settings.max_context_tokens==16384
    assert settings.max_new_tokens==768 and settings.max_steps==6 and settings.question_timeout_seconds==90
    assert settings.schema_mode=='code_first' and settings.adaptive_preparation is False
    assert settings.require_plan and settings.auto_execute_plan
    assert settings.mode=='self_review'


def test_runtime_preflight_rejects_stale_source_or_imports_before_weights(tmp_path,monkeypatch):
    import datamate.agent.runner
    import datamate.data.normalization
    from tools.colab_runtime_check import verify_runtime
    root=Path(__file__).resolve().parents[1]
    assert verify_runtime(root)['pipeline_version']==6
    monkeypatch.setattr(datamate.agent.runner.Agent,'pipeline_version',5)
    with pytest.raises(RuntimeError,match='Module cũ'):verify_runtime(root)
    monkeypatch.setattr(datamate.agent.runner.Agent,'pipeline_version',6)
    monkeypatch.setattr(datamate.data.normalization,'semantic_version',lambda:'stale')
    with pytest.raises(RuntimeError,match='chuẩn hóa'):verify_runtime(root)
    source=tmp_path/'src/datamate';(source/'agent').mkdir(parents=True);(source/'models').mkdir()
    (source/'agent/runner.py').write_text('class Agent:\n    pipeline_version = 5\n')
    (source/'models/backends.py').write_text('class TransformersBackend:\n    protocol_version = 3\n')
    with pytest.raises(RuntimeError,match='Source ZIP cũ'):verify_runtime(tmp_path)
