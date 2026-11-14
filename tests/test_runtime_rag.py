from pathlib import Path
import json
import time

import pytest

from datamate.config import Settings
from datamate.domain import DataMateError, Question
from datamate.models.registry import MODELS, ModelSpec
from datamate.models.contracts import BackendUnavailableError
from tools import runtime_artifacts as artifacts
from tools.install_runtime import install_bundle
from tools.package_solution import stage
from test_runtime_artifacts import runtime, model, prepare


def enable_rag(runtime, tmp_path, monkeypatch):
    runtime.settings.rag.mode = 'dense_rerank'
    for key, name in [('vietnamese-embedding', 'embedding_dir'), ('vietnamese-reranker', 'reranker_dir')]:
        spec = ModelSpec(key, 'test/' + key, 'b' * 40, 3, '2025-01-01T00:00:00Z')
        monkeypatch.setitem(MODELS, key, spec)
        folder = tmp_path / key
        model(folder, spec)
        setattr(runtime.settings.rag, name, str(folder))


def test_nonempty_unsealed_checkpoint_never_downloads_a_base(runtime, tmp_path):
    source = tmp_path / 'trained-but-unsealed'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'checkpoint owned by user')
    runtime.settings.model_dir = str(source)
    with pytest.raises(DataMateError, match='manifest|seal'):
        prepare(runtime)
    assert not any('stage-model' in command for command, _ in runtime.calls)


def test_optional_lock_keeps_project_inference_and_build_requirements(runtime, tmp_path):
    lock = tmp_path / 'gpu-lock.txt'
    lock.write_text('fixturelib==1.0\n', encoding='utf-8')
    prepare(runtime, requirements=lock)
    command = next(command for command, _ in runtime.calls if 'download' in command)
    assert 'openpyxl==3.1.5' in command
    assert 'setuptools>=77' in command
    assert any('transformers>=' in value for value in command)
    assert '--requirement' in command


def test_bundle_stage_and_offline_relocate_include_all_rag_models(runtime, tmp_path, monkeypatch):
    enable_rag(runtime, tmp_path, monkeypatch)
    bundle = prepare(runtime)
    assert set(bundle['models']) == {runtime.spec.key, 'vietnamese-embedding', 'vietnamese-reranker'}
    wheelhouse = runtime.root / bundle['wheelhouse']
    assert artifacts.verify_wheels(wheelhouse)['identity']['extra'] == 'llm,rag'
    config = runtime.root / bundle['config']
    sources = {key: runtime.root / item['path'] for key, item in bundle['models'].items()}
    output = tmp_path / 'docker'
    result = stage(runtime.project, config, output, wheelhouse, sources)
    assert len(result['inventory']['models']) == 3 and result['inference_extra'] == 'llm,rag'
    settings = Settings.load(output / 'configs/submission.json', use_env=False)
    assert settings.rag.embedding_dir == '/app/models/vietnamese-embedding'
    assert settings.rag.reranker_dir == '/app/models/vietnamese-reranker'
    assert (output / 'models/vietnamese-reranker/artifact_manifest.json').is_file()
    relocated = tmp_path / 'installed-config.json'
    install_bundle(runtime.root / 'bundle.json', 'target-python', runtime.project, relocated)
    after = Settings.load(relocated, use_env=False)
    assert after.rag.embedding_dir == str(sources['vietnamese-embedding'])
    assert after.rag.reranker_dir == str(sources['vietnamese-reranker'])


def test_pinned_rag_payload_hash_is_enforced_without_restricting_generator(tmp_path):
    from datamate.models.artifacts import verify_checkpoint_weight_hash
    from datamate.models.registry import get_model
    (tmp_path / 'model.safetensors').write_bytes(b'untrusted payload')
    with pytest.raises(DataMateError, match='Hash|hash'):
        verify_checkpoint_weight_hash(tmp_path, get_model('vietnamese-embedding'))
    assert verify_checkpoint_weight_hash(tmp_path, get_model('qwen35-9b')) is None


def test_fatal_sequential_generation_keeps_question_id():
    from datamate.batching import run_questions

    class Agent:
        def run(self, *args):
            raise BackendUnavailableError('engine died')

    question = Question('failed-question', 'business', 'q', '2026-01-01')
    with pytest.raises(BackendUnavailableError) as caught:
        list(run_questions(Agent(), [question], 1, time.monotonic() + 1, {}))
    assert caught.value.question_id == 'failed-question'
