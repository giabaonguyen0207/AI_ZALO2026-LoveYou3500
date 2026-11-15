import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from datamate.config import Settings
from datamate.domain import DataMateError, Question
from datamate.models.contracts import BackendUnavailableError, GenerationTimeoutError
from datamate.models.vllm_backend import VLLMBackend


class Tokenizer:
    eos_token_id, pad_token_id = 2, 0
    all_special_ids = [0, 1, 2]
    chat_template = 'assistant <|im_end|>'

    def __len__(self): return 256
    def convert_tokens_to_ids(self, text): return 2 if text == '<|im_end|>' else None
    def convert_ids_to_tokens(self, token): return '<|im_end|>' if token == 2 else None
    def encode(self, text, **kwargs): return [2]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_dict=True, enable_thinking=False)
        return {'input_ids': [4, 5] + [6] * sum(len(m['content']) for m in messages)}

    def decode(self, tokens, **kwargs):
        assert kwargs == {'skip_special_tokens': False}
        return json.dumps({'action': 'final', 'type': 'abstain', 'reason': str(tokens[0])})


class Engine:
    def __init__(self, *, batch=0, delay=0.01, fail=False):
        self.batch, self.delay, self.fail = batch, delay, fail
        self.requests, self.active, self.aborted = [], set(), []
        self.peak, self.closed = 0, False
        self.ready = asyncio.Event()

    async def generate(self, prompt, params, *, request_id):
        self.requests.append((prompt, params, request_id))
        self.active.add(request_id)
        self.peak = max(self.peak, len(self.active))
        if len(self.active) >= self.batch:
            self.ready.set()
        try:
            await self.ready.wait()
            await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError('engine died / CUDA out of memory')
            yield SimpleNamespace(finished=True, outputs=[SimpleNamespace(
                token_ids=[len(prompt['prompt_token_ids']), 2], finish_reason='stop', stop_reason=2)])
        finally:
            self.active.remove(request_id)

    async def abort(self, request_id):
        self.aborted.append(request_id)

    def shutdown(self):
        self.closed = True


@pytest.fixture
def make_backend(tmp_path, monkeypatch):
    from datamate.models import vllm_backend
    verified, engines, backends = [], [], []
    monkeypatch.setattr(vllm_backend, 'verify_artifact', lambda path, key: verified.append((path, key)))

    def make(engine=None, **overrides):
        engine = engine or Engine()
        settings = Settings(backend='vllm', model_dir=str(tmp_path / 'custom checkpoint'),
                            cache_dir=str(tmp_path / 'cache'), **overrides)
        backend = VLLMBackend(settings, engine_factory=lambda options: engine,
            sampling_factory=lambda **kwargs: SimpleNamespace(**kwargs), tokenizer=Tokenizer(),
            generation_config=SimpleNamespace(eos_token_id=1, pad_token_id=0))
        engines.append(engine)
        backends.append(backend)
        return backend, engine
    yield make
    for backend in backends:
        backend.close()
    assert len(verified) == len(backends)


def test_shared_async_engine_actually_overlaps_requests_and_keeps_thread_local_logs(make_backend):
    backend, engine = make_backend(Engine(batch=4), question_workers=4)
    barrier = threading.Barrier(4)

    def generate(i):
        barrier.wait(timeout=3)
        messages = [{'role': 'user', 'content': 'x' * (i+1)}]
        barrier_tokens = backend.count_tokens(messages)
        raw = backend.generate(messages, 3, max_new_tokens=i+4)
        return barrier_tokens, json.loads(raw), dict(backend.last_generation)

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(generate, range(4)))
    assert engine.peak == backend.runtime_info()['active_requests_peak'] == 4
    assert len({request_id for _, _, request_id in engine.requests}) == 4
    assert not engine.active
    for i, (count, raw, meta) in enumerate(rows):
        assert count == meta['input_tokens'] == i+3
        assert meta['max_new_tokens'] == i+4 and meta['output_tokens'] == 2
        assert raw['reason'] == str(count)
        assert meta['effective_eos_token_ids'] == [1, 2]
    assert backend.last_generation is None  # Main thread did not generate.
    assert all(params.temperature == 0 and params.seed == backend.settings.seed and params.skip_special_tokens is False
               for _, params, _ in engine.requests)


def test_context_and_expired_budget_never_submit_to_engine(make_backend):
    backend, engine = make_backend()
    with pytest.raises(DataMateError, match='Context vượt'):
        backend.generate([{'role': 'user', 'content': 'x' * backend.settings.max_context_tokens}], 1)
    with pytest.raises(GenerationTimeoutError):
        backend.generate([], 0)
    assert engine.requests == []


def test_cache_hit_metadata_comes_from_request_output_and_can_be_unknown(make_backend):
    class CachedEngine(Engine):
        async def generate(self, *args, **kwargs):
            async for row in super().generate(*args, **kwargs):
                row.num_cached_tokens = 1
                yield row
    backend, _ = make_backend(CachedEngine())
    raw = backend.generate([{'role':'user','content':'x'}],2)
    assert json.loads(raw)['action']=='final' and backend.last_generation['cached_tokens']==1
    other, _ = make_backend()
    other.generate([{'role':'user','content':'x'}],2)
    assert other.last_generation['cached_tokens'] is None


def test_timeout_aborts_and_engine_can_serve_next_request(make_backend):
    backend, engine = make_backend(Engine(delay=0.1))
    with pytest.raises(GenerationTimeoutError):
        backend.generate([{'role': 'user', 'content': 'q'}], 0.02)
    assert len(engine.aborted) == 1 and not engine.active
    assert backend.last_generation['stop_reason'] == 'timeout_aborted'
    backend.generate([], 1)
    assert len(engine.requests) == 2


def test_abort_failure_is_fatal_and_not_retried(make_backend):
    engine = Engine(delay=0.1)
    async def bad_abort(request_id):
        raise RuntimeError('IPC broken')
    engine.abort = bad_abort
    backend, engine = make_backend(engine)
    with pytest.raises(BackendUnavailableError, match='abort'):
        backend.generate([], 0.02)
    with pytest.raises(BackendUnavailableError):
        backend.generate([], 1)
    assert len(engine.requests) == 1


def test_engine_death_propagates_through_agent_instead_of_becoming_abstain(make_backend, split_root):
    from datamate.agent.runner import Agent
    backend, engine = make_backend(Engine(fail=True), split_root=str(split_root))
    agent = Agent(backend.settings, backend)
    with pytest.raises(BackendUnavailableError, match='engine died'):
        agent.run(Question('arbitrary', 'shop_a', 'Tổng tiền?', '2026-06-02'))
    with pytest.raises(BackendUnavailableError):
        backend.generate([], 1)
    assert len(engine.requests) == 1


def test_close_aborts_live_requests_and_stops_loop_idempotently(make_backend):
    backend, engine = make_backend(Engine(delay=5))
    started = threading.Event()
    original = engine.generate
    async def generate(*args, **kwargs):
        started.set()
        async for item in original(*args, **kwargs):
            yield item
    engine.generate = generate
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(backend.generate, [], 10)
        assert started.wait(2)
        backend.close()
        with pytest.raises(GenerationTimeoutError):
            pending.result(timeout=2)
    backend.close()
    assert engine.closed and engine.aborted and not engine.active
    assert not backend._thread.is_alive()
    json.dumps(backend.runtime_info())


def test_engine_options_are_supplied_and_local_model_is_not_loaded_twice(make_backend):
    backend, engine = make_backend(vllm=dict(tensor_parallel_size=2, dtype='float16',
        quantization='bitsandbytes', load_format='bitsandbytes', gpu_memory_utilization=0.7,
        max_num_seqs=3, max_num_batched_tokens=2048, enable_prefix_caching=False,
        extra_engine_args={'disable_log_stats': True}))
    options = backend.options
    assert options['model'] == options['tokenizer'] == str(Path(backend.settings.model_dir).resolve())
    assert options['tensor_parallel_size'] == 2 and options['max_num_seqs'] == 3
    assert options['quantization'] == options['load_format'] == 'bitsandbytes'
    assert options['max_model_len'] == backend.settings.max_context_tokens
    assert options['max_num_batched_tokens'] == 2048 and options['enable_prefix_caching'] is False
    assert options['trust_remote_code'] is False and options['disable_log_stats'] is True
    assert options['language_model_only'] is True


def test_optional_sampling_arguments_reach_engine_and_logs(make_backend):
    backend, engine = make_backend(vllm={'extra_sampling_params': {'temperature': 0.2, 'top_p': 0.8}})
    backend.generate([], 1)
    assert engine.requests[0][1].temperature == 0.2 and engine.requests[0][1].top_p == 0.8
    assert backend.last_generation['effective_do_sample'] is True
    assert backend.last_generation['sampling_params']['top_p'] == 0.8


def test_fatal_errors_are_not_cached_as_schema_or_preparation_failures(make_backend, split_root, monkeypatch):
    from datamate.agent.preparation import BusinessPreparer
    from datamate.agent.schema_repair import SchemaRepair
    from datamate.data.ingest import Catalog
    from datamate.data.split import DatasetSplit
    import datamate.agent.schema_repair as schema_module
    import time
    backend, engine = make_backend(Engine(fail=True), split_root=str(split_root))
    split = DatasetSplit(split_root)
    preparer = BusinessPreparer(backend.settings, split)
    with pytest.raises(BackendUnavailableError):
        preparer._propose('test', {}, lambda value: value, None, backend,
                          time.monotonic()+2, 'test', [])
    monkeypatch.setattr(schema_module, 'repair_needs', lambda *args: {'policy': ['week_start']})
    repair = SchemaRepair(backend.settings, split)
    catalog = Catalog(Path(backend.settings.cache_dir) / 'fixture.sqlite', [], [], 'fixture',
                      {'bindings': [], 'profiles': {}, 'policy': {'week_start': None}})
    with pytest.raises(BackendUnavailableError):
        repair.repair(catalog, catalog, '', {}, backend, time.monotonic()+2)
    assert not list(repair.cache.directory.glob('*.json'))
