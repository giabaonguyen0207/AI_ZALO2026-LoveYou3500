"""Application services shared by CLI and HTTP; no second inference service."""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from .agent.runner import Agent
from .config import Settings
from .data.split import DatasetSplit, ensure_outside_dataset, read_jsonl
from .domain import Answer, DataMateError, Question, RunResult, json_text
from .models.backends import build_backend
from .models.registry import inventory
from .models.artifacts import sha256


def create_agent(settings: Settings) -> Agent:
    settings.validate()
    inventory(settings.model_key, settings.assistant_key)
    split = DatasetSplit(settings.split_root)
    # Check writable outputs before loading a large model.
    ensure_outside_dataset(settings.cache_dir, split.root)
    ensure_outside_dataset(settings.result_dir, split.root)
    return Agent(settings, build_backend(settings), split)


def run_batch(settings: Settings, ids: set[str] | None = None, limit: int | None = None, resume: bool = False) -> dict:
    started = time.monotonic()  # Includes model load, hash verification and ingestion.
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
    logs = result_dir / "logs"
    logs.mkdir(exist_ok=True)
    output = result_dir / "submission.jsonl"
    identity = {"settings": settings.to_dict(), "rulebook_sha256": split.rulebook_hash,
                "questions_sha256": hashlib.sha256((split.root / "questions.jsonl").read_bytes()).hexdigest(),
                "selected_ids_sha256": hashlib.sha256(json_text([q.id for q in questions]).encode()).hexdigest(),
                "inventory": inventory(settings.model_key, settings.assistant_key)}
    input_files = [split.root / "calendar.csv"]
    for business in sorted({q.business_id for q in questions}):
        input_files.extend(sorted(p for p in split.business_path(business).rglob("*") if p.is_file()))
    identity["input_hashes"] = {p.relative_to(split.root).as_posix(): sha256(p) for p in input_files}
    if settings.backend != "demo":
        identity["artifact_manifests"] = {}
        for folder in (settings.model_dir, settings.assistant_dir):
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
    agent = create_agent(settings)
    loaded_seconds = time.monotonic() - started
    written, timed_out = 0, False
    with output.open("a" if resume else "w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            if question.id in completed:
                continue
            remaining = settings.wall_time_seconds - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                break  # Missing answers count as abstain; existing rows survive.
            run = agent.run(question, remaining)
            name = hashlib.sha256(question.id.encode()).hexdigest()[:24] + ".json"
            payload = {"id": question.id, "business_id": question.business_id,
                       "rulebook_sha256": split.rulebook_hash, **run.to_dict(question.id)}
            temporary = logs / (name + ".tmp")
            temporary.write_text(json_text(payload, indent=2), encoding="utf-8")
            temporary.replace(logs / name)
            handle.write(json_text(run.answer.to_dict(question.id)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1
    summary = {"backend": settings.backend, "mode": settings.mode, "model": settings.model_key,
               "selected_questions": len(questions), "previously_completed": len(completed), "written": written,
               "wall_time_seconds": round(time.monotonic()-started, 4), "load_seconds": round(loaded_seconds, 4),
               "time_budget_exhausted": timed_out, "submission": str(output.resolve())}
    (result_dir / "run_summary.json").write_text(json_text(summary, indent=2), encoding="utf-8")
    return summary
