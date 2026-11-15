import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from datamate.cli import parser, load_settings
from datamate.domain import DataMateError


def test_backend_and_rag_are_independent_cli_choices(monkeypatch):
    monkeypatch.setenv('DATAMATE_BACKEND', 'vllm')
    monkeypatch.setenv('DATAMATE_RAG_MODE', 'none')
    args = parser().parse_args(['run', '--backend', 'torch', '--rag-mode', 'dense_rerank',
                               '--rag-device', 'cpu', '--rag-index-dir', 'index with spaces'])
    settings = load_settings(args)
    assert settings.backend == 'transformers' and settings.rag.mode == 'dense_rerank'
    assert settings.rag.device == 'cpu' and settings.rag.index_dir == 'index with spaces'
    settings = load_settings(parser().parse_args(['run', '--backend', 'vllm', '--rag-mode', 'none']))
    assert settings.backend == 'vllm' and settings.rag.mode == 'none'


def test_supervisor_terminates_an_unresponsive_process():
    from datamate.execution import run_checked_process
    started = time.monotonic()
    with pytest.raises(DataMateError, match='timeout|budget'):
        run_checked_process([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.15, grace_seconds=0.1)
    assert time.monotonic() - started < 5


def test_supervisor_enforces_model_startup_deadline(tmp_path):
    from datamate.execution import run_checked_process
    progress = tmp_path / 'startup_progress.json'
    code = ('import json,time; from pathlib import Path; '
            f'Path({str(progress)!r}).write_text(json.dumps(dict(phase="generator_init",monotonic_started=time.monotonic(),timeout_seconds=0.10))); '
            'time.sleep(10)')
    with pytest.raises(DataMateError, match='startup'):
        run_checked_process([sys.executable, '-c', code], timeout=10, startup_progress=progress, grace_seconds=0.1)


def test_supervised_demo_writes_real_shared_pipeline_results(split_root, tmp_path, monkeypatch):
    from datamate.execution import run_isolated
    from datamate.config import Settings
    monkeypatch.setenv('DATAMATE_BACKEND', 'vllm')  # Resolved CLI/config must win in child.
    settings = Settings(backend='demo', split_root=str(split_root), cache_dir=str(tmp_path / 'cache'),
                        result_dir=str(tmp_path / 'result')).validate()
    summary = run_isolated(settings)
    assert summary['written'] == 3 and summary['backend'] == 'demo'
    rows = [json.loads(line) for line in (tmp_path / 'result/submission.jsonl').read_text(encoding='utf-8').splitlines()]
    assert {row['id'] for row in rows} == {'question_0', 'question_1', 'question_2'}
    assert all(row['type'] == 'abstain' for row in rows)
    assert summary['process_wall_seconds'] >= summary['wall_time_seconds']


def test_common_precision_reaches_both_generator_loaders(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    import transformers
    from datamate.config import Settings
    from datamate.models import backends
    from datamate.models.vllm_backend import engine_options
    from test_vllm_backend import Tokenizer
    captured = []
    model = SimpleNamespace(eval=lambda: model,
        generation_config=SimpleNamespace(eos_token_id=1, pad_token_id=0),
        get_output_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(shape=(256, 1))))
    monkeypatch.setattr(backends, 'verify_artifact', lambda *args: {})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(transformers, 'set_seed', lambda value: None)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(backends, 'load_target', lambda *args, **kwargs: captured.append(kwargs) or model)
    settings = Settings(dtype='float32', cache_dir=str(tmp_path)).validate()
    backend = backends.TransformersBackend(settings)
    assert captured[0]['dtype'] == torch.float32
    settings.backend = 'vllm'
    assert engine_options(settings)['dtype'] == 'float32'
    backend.close()
