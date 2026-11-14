import asyncio
import json
from pathlib import Path

import pytest

from datamate.application import run_batch
from datamate.models.contracts import BackendUnavailableError
from test_vllm_backend import Engine, make_backend
from test_pipeline_v3 import business, plan


def install(monkeypatch, backend):
    import datamate.application as app
    path = Path(backend.settings.model_dir)
    path.mkdir(parents=True)
    (path / 'artifact_manifest.json').write_text('{}')
    monkeypatch.setattr(app, 'build_backend', lambda settings, **kwargs: backend)


def test_real_batch_controller_overlaps_agents_flushes_each_id_and_resumes_without_load(make_backend, split_root, tmp_path, monkeypatch):
    questions = [json.loads(line) for line in (split_root / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    for question in questions:
        question['business_id'] = 'shop_a'  # Same-shop requests must still overlap.
    (split_root / 'questions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in questions), encoding='utf-8')
    backend, engine = make_backend(Engine(batch=3), split_root=str(split_root),
        result_dir=str(tmp_path / 'result'), question_workers=3)
    install(monkeypatch, backend)
    summary = run_batch(backend.settings)
    output = Path(backend.settings.result_dir)
    rows = [json.loads(line) for line in (output / 'submission.jsonl').read_text().splitlines()]
    assert {r['id'] for r in rows} == {'question_0', 'question_1', 'question_2'}
    assert len(rows) == summary['written'] == 3
    assert summary['question_workers'] == engine.peak == 3
    assert engine.closed and not engine.active
    logs = [json.loads(p.read_text(encoding='utf-8')) for p in (output / 'logs').glob('*.json')]
    assert len(logs) == 3
    for log in logs:
        assert log['trace'][0]['generation']['output_tokens'] == 2
        assert log['trace'][0]['generation']['stop_reason'] == 'stop'
    resumed = run_batch(backend.settings, resume=True)
    assert resumed['written'] == 0 and resumed['previously_completed'] == 3


def test_fatal_engine_failure_preserves_other_finished_rows_and_stops_new_work(make_backend, split_root, tmp_path, monkeypatch):
    import datamate.application as app

    class PartialFailure(Engine):
        def __init__(self):
            super().__init__(batch=3)
            self.can_fail = asyncio.Event()

        async def generate(self, prompt, params, *, request_id):
            async for result in super().generate(prompt, params, request_id=request_id):
                index = next(i for i, (_, _, rid) in enumerate(self.requests) if rid == request_id)
                if index == 2:
                    await self.can_fail.wait()
                    raise RuntimeError('fatal after two saved results')
                yield result

    backend, engine = make_backend(PartialFailure(), split_root=str(split_root),
        result_dir=str(tmp_path / 'result'), question_workers=3)
    install(monkeypatch, backend)
    saved = []
    original = app.write_debug_log

    def write(path, payload):
        original(path, payload)
        saved.append(payload['id'])
        if len(saved) == 2:
            backend._loop.call_soon_threadsafe(engine.can_fail.set)
    monkeypatch.setattr(app, 'write_debug_log', write)
    with pytest.raises(BackendUnavailableError, match='two saved'):
        run_batch(backend.settings)
    output = Path(backend.settings.result_dir)
    rows = [json.loads(line) for line in (output / 'submission.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and {r['id'] for r in rows} == set(saved)
    summary = json.loads((output / 'run_summary.json').read_text(encoding='utf-8'))
    assert summary['written'] == 2 and 'two saved' in summary['error']
    error = json.loads((output / 'logs/runtime-error.json').read_text(encoding='utf-8'))
    assert error['id'] in {'question_0', 'question_1', 'question_2'} - set(saved)
    assert engine.closed and not engine.active


def test_wall_deadline_aborts_inflight_requests_and_closes_workers(make_backend, split_root, tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    import datamate.application as app
    import datamate.batching as batching
    import datamate.agent.runner as runner
    started = time.monotonic()
    active_at = []
    class WaitingEngine(Engine):
        async def generate(self,*args,**kwargs):
            if not active_at:
                active_at.append(time.monotonic())
            async for result in super().generate(*args,**kwargs):
                yield result
    # Freeze only application clocks during hashing/telemetry so host I/O
    # cannot consume the fixture deadline before it exercises an active request.
    clock = SimpleNamespace(monotonic=lambda: started+(time.monotonic()-active_at[0] if active_at else 0))
    for module in (app,batching,runner):
        monkeypatch.setattr(module,'time',clock)
    backend, engine = make_backend(WaitingEngine(delay=5), split_root=str(split_root),
        result_dir=str(tmp_path / 'result'), question_workers=2, wall_time_seconds=.2)
    install(monkeypatch, backend)
    summary = run_batch(backend.settings)
    assert summary['time_budget_exhausted'] is True
    assert engine.closed and not engine.active and not backend._thread.is_alive()
    assert 1 <= len(engine.requests) <= 2  # Exercise an active request, not a pre-load deadline.


def test_resume_rejects_changed_scheduler_or_engine_settings(make_backend, split_root, tmp_path, monkeypatch):
    backend, engine = make_backend(split_root=str(split_root), result_dir=str(tmp_path / 'result'))
    install(monkeypatch, backend)
    run_batch(backend.settings)
    backend.settings.vllm.enable_prefix_caching = False
    from datamate.domain import DataMateError
    with pytest.raises(DataMateError, match='resume'):
        run_batch(backend.settings, resume=True)


@pytest.mark.parametrize('mode,calls', [('single', 1), ('self_review', 2)])
def test_backend_selection_preserves_existing_prompt_plan_sql_and_agent_loop(business, tmp_path, scripted_backend, mode, calls):
    from datamate.agent.runner import Agent
    from datamate.config import Settings
    from datamate.domain import Question
    split, _ = business
    question = Question('arbitrary', 'shop_a', 'Doanh thu từ đầu năm', '2025-01-31')
    runs = []
    for name in ('transformers', 'vllm'):
        actions = [plan(question.question), {'action': 'final', 'type': 'number'}]
        backend = scripted_backend(actions)
        settings = Settings(backend=name, mode=mode, split_root=str(split.root), cache_dir=str(tmp_path / name))
        assert settings.vllm.cache_prompt_prefix is False
        runs.append(Agent(settings, backend).run(question))
    left, right = runs
    assert left.answer.answer == right.answer.answer == 400
    assert left.model_calls == right.model_calls == calls
    assert left.initial_messages == right.initial_messages
    assert [event.get('llm_action') for event in left.trace] == [event.get('llm_action') for event in right.trace]
    assert left.trace[0]['execution']['sql'] == right.trace[0]['execution']['sql']
    assert left.trace[0]['execution']['rows'] == right.trace[0]['execution']['rows']
