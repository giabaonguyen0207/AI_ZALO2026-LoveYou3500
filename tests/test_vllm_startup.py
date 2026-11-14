import time
from types import SimpleNamespace

import pytest

from datamate.config import Settings
from datamate.models.contracts import BackendUnavailableError
from datamate.models.vllm_backend import VLLMBackend
from test_vllm_backend import make_backend, Engine, Tokenizer


def test_runtime_separates_startup_phases_and_effective_dtype(make_backend):
    engine = Engine()
    engine.model_config = SimpleNamespace(dtype='float16')
    backend, _ = make_backend(engine)
    info = backend.runtime_info()
    assert set(info['startup_phases']) >= {'artifact_verification_seconds', 'library_import_seconds', 'tokenizer_seconds', 'engine_init_seconds'}
    assert all(value >= 0 for value in info['startup_phases'].values())
    assert info['resolved_dtype'] == 'float16'
    assert 'cuda_available' in info['hardware']


def test_verification_time_is_subtracted_before_starting_engine(tmp_path, monkeypatch):
    from datamate.models import vllm_backend
    monkeypatch.setattr(vllm_backend, 'verify_artifact', lambda *args: time.sleep(0.03))
    started = []
    settings = Settings(backend='vllm', cache_dir=str(tmp_path), vllm={'init_timeout_seconds': 0.01})
    backend = None
    try:
        with pytest.raises(BackendUnavailableError, match='deadline|budget|thời gian'):
            backend = VLLMBackend(settings, engine_factory=lambda options: started.append(True) or Engine(),
                sampling_factory=lambda **kwargs: SimpleNamespace(**kwargs), tokenizer=Tokenizer(),
                generation_config=SimpleNamespace(eos_token_id=1, pad_token_id=0))
        assert started == []
    finally:
        if backend is not None:
            backend.close()
