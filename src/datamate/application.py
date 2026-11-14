"""Application services shared by CLI and HTTP; no second inference service."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from .agent.runner import Agent
from .config import Settings
from .data.split import DatasetSplit, ensure_outside_dataset, read_jsonl
from .data.ingest import Ingestor
from .domain import Answer, DataMateError, Question, RunResult, json_text
from .debug_log import write_debug_log
from .models.backends import build_backend
from .models.registry import inventory_from_settings, model_paths
from .retrieval import build_retriever
from .models.artifacts import sha256
from .models.runtime import runtime_identity
from .batching import run_questions


def create_agent(settings: Settings, *, business_ids=None, remaining_seconds=None, progress=None) -> Agent:
    started = time.monotonic()
    settings.validate()
    deadline = started + (settings.wall_time_seconds if remaining_seconds is None else remaining_seconds)
    inventory_from_settings(settings)
    split = DatasetSplit(settings.split_root)
    # Check writable outputs before loading a large model.
    ensure_outside_dataset(settings.cache_dir, split.root)
    ensure_outside_dataset(settings.result_dir, split.root)
    if settings.rag.index_dir:
        ensure_outside_dataset(settings.rag.index_dir, split.root)
    retriever = build_retriever(settings, split)
    backend = None
    try:
        ingestor = Ingestor(split, settings.cache_dir)
        if settings.rag.mode != 'none':
            if progress:
                progress('retrieval_prepare', deadline-time.monotonic())
            selected = sorted(set(business_ids if business_ids is not None else [q.business_id for q in split.questions()]))
            retriever.prepare(selected, ingestor=ingestor, remaining_seconds=deadline-time.monotonic())
        if time.monotonic() >= deadline:
            raise DataMateError('Hết wall budget khi chuẩn bị RAG/model')
        if progress:
            progress('generator_init', min(deadline-time.monotonic(), settings.vllm.init_timeout_seconds)
                     if settings.backend == 'vllm' else deadline-time.monotonic())
        backend = (build_backend(settings, remaining_seconds=deadline-time.monotonic())
                   if settings.backend == 'vllm' else build_backend(settings))
        agent = Agent(settings, backend, split, retriever, ingestor=ingestor)
        if progress:
            progress('ready', deadline-time.monotonic())
        return agent
    except BaseException:
        try:
            if backend is not None:
                backend.close()
        finally:
            retriever.close()
        raise


def run_batch(settings: Settings, ids: set[str] | None = None, limit: int | None = None, resume: bool = False) -> dict:
    started = time.monotonic()  # Includes model load, hash verification and ingestion.
    settings.validate()
    split = DatasetSplit(settings.split_root)
    questions = split.questions()
    if ids is not None:
        available = {q.id for q in questions}
        if not ids <= available:
            raise DataMateError("Danh sách ID có câu không thuộc split")
        questions = [q for q in questions if q.id in ids]
    if limit is not None:
        if limit < 1:
            raise DataMateError("limit phải dương")
        questions = questions[:limit]
    result_dir = ensure_outside_dataset(settings.result_dir, split.root)
    result_dir.mkdir(parents=True, exist_ok=True)
    def progress(phase, timeout):
        path = result_dir / 'startup_progress.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json_text({'phase': phase, 'monotonic_started': time.monotonic(),
                                        'timeout_seconds': max(0, timeout)}), encoding='utf-8')
        temporary.replace(path)
    logs = result_dir / "logs"
    logs.mkdir(exist_ok=True)
    output = result_dir / "submission.jsonl"
    identity = {"settings": settings.to_dict(), "rulebook_sha256": split.rulebook_hash,
                "questions_sha256": hashlib.sha256((split.root / "questions.jsonl").read_bytes()).hexdigest(),
                "selected_ids_sha256": hashlib.sha256(json_text([q.id for q in questions]).encode()).hexdigest(),
                "inventory": inventory_from_settings(settings),
                "runtime": runtime_identity(settings)}
    package_root=Path(__file__).parent
    identity['source_hashes']={p.relative_to(package_root).as_posix():sha256(p) for p in sorted(package_root.rglob('*.py'))}
    input_files = [split.root / "calendar.csv"]
    for business in sorted({q.business_id for q in questions}):
        input_files.extend(sorted(p for p in split.business_path(business).rglob("*") if p.is_file()))
    identity["input_hashes"] = {p.relative_to(split.root).as_posix(): sha256(p) for p in input_files}
    if settings.backend != "demo":
        identity["artifact_manifests"] = {}
        for folder in model_paths(settings).values():
            if folder:
                manifest_path = Path(folder) / "artifact_manifest.json"
                if not manifest_path.exists():
                    raise DataMateError("Chưa chuẩn bị checkpoint local. Chạy stage-model trước.")
                identity["artifact_manifests"][str(Path(folder).resolve())] = sha256(manifest_path)
    import json
    identity_path = result_dir / "run_identity.json"
    if output.exists() and not resume:
        raise DataMateError("submission.jsonl đã có. Chọn result_dir mới hoặc --resume")
    completed = set()
    if resume and output.exists():
        if not identity_path.exists() or json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise DataMateError("Không resume: nguồn/cấu hình/danh sách câu hỏi đã thay đổi")
        rows = read_jsonl(output)
        completed = {row["id"] for row in rows}
        if len(completed) != len(rows) or not completed <= {q.id for q in questions}:
            raise DataMateError("Submission resume trùng ID hoặc chứa ID ngoài lượt chạy")
        for row in rows:
            Answer.from_dict(row)
    identity_path.write_text(json_text(identity, indent=2), encoding="utf-8")
    todo = [q for q in questions if q.id not in completed]
    if settings.backend == 'vllm' and settings.vllm.group_by_business:
        todo.sort(key=lambda q: q.business_id)  # IDs route data; never determine answers.
    print(f'Inference: {len(questions)} câu; đã có {len(completed)}, còn {len(todo)}. Kết quả: {output}', flush=True)
    deadline = started + settings.wall_time_seconds
    state = {'time_budget_exhausted': False}
    agent, loaded_seconds, written, failure = None, None, 0, None
    try:
        if time.monotonic() >= deadline:
            state['time_budget_exhausted'] = True
            todo = []
        if todo:
            print('Đang nạp model và chuẩn bị dữ liệu...', flush=True)
            agent = create_agent(settings, business_ids=sorted({q.business_id for q in todo}),
                                 remaining_seconds=deadline-time.monotonic(), progress=progress)
            print(f'Model đã sẵn sàng sau {time.monotonic()-started:.1f}s; bắt đầu trả lời.', flush=True)
        loaded_seconds = time.monotonic()-started
        if agent and callable(getattr(agent.backend, 'runtime_info', None)):
            (result_dir / 'runtime.json').write_text(json_text(agent.backend.runtime_info(), indent=2), encoding='utf-8')
        if agent and settings.rag.mode != 'none':
            (result_dir / 'retrieval_runtime.json').write_text(json_text(agent.retriever.runtime_info(), indent=2), encoding='utf-8')
        with output.open('a' if resume else 'w', encoding='utf-8', newline='\n') as handle:
            runs = run_questions(agent, todo, settings.effective_workers, deadline, state) if agent else iter(())
            try:
                for question, run in runs:
                    name = hashlib.sha256(question.id.encode()).hexdigest()[:24] + '.json'
                    payload = {'id': question.id, 'business_id': question.business_id,
                               **question.prompt_dict(), 'rulebook_sha256': split.rulebook_hash,
                               **run.to_dict(question.id)}
                    temporary = logs / (name + '.tmp')
                    temporary.write_text(json_text(payload, indent=2), encoding='utf-8')
                    temporary.replace(logs / name)
                    handle.write(json_text(run.answer.to_dict(question.id)) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    written += 1
                    write_debug_log((logs / name).with_suffix('.md'), payload)
                    error = any(item.get('error_kind') in {'runtime', 'generation'} for item in run.trace)
                    outcome = 'LỖI XỬ LÝ' if error else 'ĐÃ LƯU'
                    print(f'[{len(completed)+written}/{len(questions)}] {outcome} {question.id} | '
                          f'{run.answer.type} | {run.elapsed_seconds:.1f}s | {run.model_calls} model calls', flush=True)
            finally:
                close_runs = getattr(runs, 'close', None)
                if close_runs:
                    close_runs()
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        (logs / 'runtime-error.json').write_text(json_text({
            'id': getattr(exc, 'question_id', None), 'error': failure,
            'elapsed_seconds': time.monotonic()-started, 'submission': {'type': 'abstain'}}, indent=2), encoding='utf-8')
        raise
    finally:
        try:
            if agent:
                agent.close()
                if callable(getattr(agent.backend, 'runtime_info', None)):
                    (result_dir / 'runtime.json').write_text(json_text(agent.backend.runtime_info(), indent=2), encoding='utf-8')
        except BaseException as exc:
            cleanup_error = f'{type(exc).__name__}: {exc}'
            (logs / 'shutdown-error.json').write_text(json_text({'error': cleanup_error,
                'elapsed_seconds': time.monotonic()-started}, indent=2), encoding='utf-8')
            if failure is None:
                failure = cleanup_error
                raise
        finally:
            summary = {'backend': settings.backend, 'mode': settings.mode, 'model': settings.model_key,
                       'question_workers': settings.effective_workers,
                       'selected_questions': len(questions), 'previously_completed': len(completed), 'written': written,
                       'wall_time_seconds': round(time.monotonic()-started, 4),
                       'load_seconds': round(loaded_seconds, 4) if loaded_seconds is not None else None,
                       **state, 'error': failure, 'submission': str(output.resolve())}
            (result_dir / 'run_summary.json').write_text(json_text(summary, indent=2), encoding='utf-8')
    return summary
