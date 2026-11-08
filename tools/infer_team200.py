"""Run the frozen team benchmark with the loaded agent; gold is read only for review."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import time
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datamate.data.split import ensure_outside_dataset, read_jsonl
from datamate.debug_log import write_debug_log
from datamate.domain import Answer, DataMateError, json_text
from datamate.evaluation.scoring import score
from tools.diagnose_train import (TimedBackend, diagnostic_flags, disconnect_colab_after_report,
                                  question_profile, runtime_snapshot, save_json, source_hashes)

BENCHMARK_ID = "datamate-train200-seed4000930841"
QUESTIONS_SHA256 = "7ae2079219d0a79ca1dd66952610ceb091e36c0ecf65359b11dac8263c67b782"
REFERENCE_ZIP_SHA256 = "0bf2b5c1f44ea56f63ea77d61cb64aa0d53cb7006761d524d0bdadb9e092a3a3"
RUNNER_VERSION = 2
MIN_PIPELINE_VERSION = 6


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json_text(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_plan(agent, participant_root):
    participant_root = Path(participant_root).resolve()
    manifest = json.loads((participant_root / "benchmark_manifest.json").read_text(encoding="utf-8"))
    if (manifest["benchmark_id"] != BENCHMARK_ID or manifest["sample_size"] != 200
            or manifest["questions_sha256"] != QUESTIONS_SHA256 or manifest.get("protocol_version") != 2):
        raise DataMateError("Cần đúng participant v2 của benchmark 200 câu đã khóa")
    for name, digest in manifest["files"].items():
        path = (participant_root / name).resolve()
        if not path.is_relative_to(participant_root) or not path.is_file() or sha256(path) != digest:
            raise DataMateError(f"Participant thiếu/khác hash: {name}")
    if agent.split.root != (participant_root / "dataset").resolve():
        raise DataMateError("Agent phải đọc dataset participant không có nhãn, không phải TRAIN đầy đủ")
    if (agent.split.root / "answers.jsonl").exists():
        raise DataMateError("Dataset inference không được chứa answers.jsonl")
    plan = agent.split.questions()
    ids = json.loads((participant_root / "question_ids.json").read_text(encoding="utf-8"))
    if len(plan) != 200 or [q.id for q in plan] != ids or len(set(ids)) != 200:
        raise DataMateError("Danh sách/thứ tự 200 câu không khớp")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", q.id) for q in plan):
        raise DataMateError("Question ID không dùng được làm tên log")
    return plan, manifest


def load_references(reference_zip, expected_ids):
    """Called after the generation loop, including when producing a partial report."""
    if sha256(reference_zip) != REFERENCE_ZIP_SHA256:
        raise DataMateError("Gói đáp án không khớp coordinator-private.zip v2")
    with zipfile.ZipFile(reference_zip) as archive:
        if archive.testzip() is not None:
            raise DataMateError("Gói đáp án bị lỗi CRC")
        manifest = json.loads(archive.read("coordinator/benchmark_manifest.json"))
        content = archive.read("coordinator/answers.jsonl")
        if (manifest["benchmark_id"] != BENCHMARK_ID or manifest["questions_sha256"] != QUESTIONS_SHA256
                or hashlib.sha256(content).hexdigest() != manifest["files"]["answers.jsonl"]):
            raise DataMateError("Nhãn không thuộc bộ benchmark đã khóa")
    rows = [json.loads(line) for line in content.decode("utf-8-sig").splitlines() if line.strip()]
    if [row["id"] for row in rows] != expected_ids:
        raise DataMateError("Nhãn không khớp thứ tự 200 câu")
    return rows


def make_bundle(destination, name, files):
    bundle = destination / name
    temporary = bundle.with_suffix(".zip.tmp")
    hashes = {p.relative_to(destination).as_posix(): sha256(p) for p in files}
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(destination).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise DataMateError(f"ZIP lỗi: {name}")
        for member, digest in hashes.items():
            with archive.open(member) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                    raise DataMateError(f"ZIP khác nội dung đã lưu: {member}")
    temporary.replace(bundle)
    (destination / (name + ".sha256")).write_text(sha256(bundle) + "  " + name + "\n", encoding="utf-8")
    return bundle


def write_outputs(destination, plan, records, runtime, ledger, reference_zip, participant, units_used):
    """Always write the two submission files, even if optional review grading fails."""
    ordered = [records[q.id] for q in plan if q.id in records]
    submissions = [r["submission"] for r in ordered]
    save_jsonl(destination / "submission.jsonl", submissions)
    incomplete_timing = any(not entry.get("closed", False) for entry in ledger["batches"])
    load_seconds = round(sum(ledger["loads"].values()), 4)
    batch_seconds = None if incomplete_timing else round(sum(e["elapsed_seconds"] for e in ledger["batches"]), 4)
    gpu = "; ".join(f"{item['name']} ({item['total_vram_gib']:.1f} GiB)" for item in runtime.get("gpus", [])) or "CPU"
    settings = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))["settings"]
    model = json.loads((Path(settings["model_dir"]) / "artifact_manifest.json").read_text(encoding="utf-8"))
    source = model.get("source", {})
    model_description = source.get("repo_id", source.get("model_id", settings["model_key"]))
    revision = source.get("revision")
    metadata = {"participant": participant, "models": [model_description + ("@" + revision if revision else "")],
                "gpu": gpu, "load_seconds": load_seconds, "inference_seconds": batch_seconds,
                "compute_units_used": units_used}
    save_json(destination / "run_metadata.json", metadata)
    times = sorted(r["elapsed_seconds"] for r in ordered)
    summary = {"benchmark_id": BENCHMARK_ID, "last_invocation_id": ledger["batches"][-1]["id"], "planned_questions": len(plan),
               "completed_questions": len(ordered), "remaining_questions": len(plan) - len(ordered),
               "status": "complete" if len(ordered) == len(plan) else "partial",
               "mean_seconds": sum(times) / len(times) if times else None,
               "median_seconds": statistics.median(times) if times else None,
               "p95_seconds": times[max(0, math.ceil(.95 * len(times)) - 1)] if times else None,
               "model_calls": sum(r["model_calls"] for r in ordered),
               "total_model_calls": sum(r["model_calls"] for r in ordered),
               "load_seconds": load_seconds, "inference_wall_seconds": batch_seconds,
               "timing_incomplete_after_hard_runtime_loss": incomplete_timing,
               "predicted_type_counts": dict(Counter(r["submission"]["type"] for r in ordered)),
               "local_evaluator": True, "pa_denominator": len(plan), "review_error": None}
    references = None
    try:
        references = load_references(reference_zip, [q.id for q in plan])
        metrics = score(submissions, references, {q.id for q in plan})
        details = {row["id"]: row for row in metrics.pop("details")}
        summary.update(metrics)
    except Exception as exc:
        summary["review_error"] = f"{type(exc).__name__}: {exc}"
        details = {}
    ref_index = {row["id"]: row for row in references} if references is not None else {}
    review = []
    for q in plan:
        record = records.get(q.id, {})
        reference = ref_index.get(q.id)
        review.append({"id": q.id, "business_id": q.business_id, **q.prompt_dict(), **record,
                       "missing": q.id not in records, "reference": reference,
                       "outcome": details.get(q.id, {}).get("result"),
                       "points": details.get(q.id, {}).get("points"),
                       "question_profile": question_profile(q.question),
                       "flags": diagnostic_flags(record, reference) if record and reference else ["not_run"] if not record else []})
    save_json(destination / "summary.json", summary)
    save_json(destination / "runtime.json", runtime)
    save_jsonl(destination / "review.jsonl", review)
    with (destination / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["id", "business_id", "question", "as_of_date", "missing", "outcome", "points",
                  "predicted_type", "predicted_answer", "reference_type", "reference_answer", "elapsed_seconds", "model_calls", "flags"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in review:
            predicted, reference = row.get("submission", {}), row.get("reference") or {}
            writer.writerow({**{key: row.get(key) for key in fields if key not in {"predicted_type", "predicted_answer", "reference_type", "reference_answer", "flags"}},
                             "predicted_type": predicted.get("type"), "predicted_answer": json_text(predicted.get("answer")),
                             "reference_type": reference.get("type"), "reference_answer": json_text(reference.get("answer")),
                             "flags": ",".join(row["flags"])})
    lines = ["# Review benchmark 200 câu", "", f"Trạng thái: {summary['status']}; đã chạy {len(ordered)}/{len(plan)} câu.",
             "", "PA tính trên toàn bộ 200 câu, thiếu/abstain = 0. Đây là benchmark nội bộ TRAIN, không phải holdout độc lập.",
             "", "```json", json_text(summary, indent=2), "```", "",
             "Chi tiết đáp án/nhãn ở review.csv và review.jsonl; raw output, SQL/tool và prompt ở logs/*.json và logs/*.md.",
             "Chỉ gửi group_submission.zip cho nhóm; diagnostic_bundle.zip có nhãn chuẩn, giữ để phân tích sau khi khóa kết quả."]
    (destination / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    group = make_bundle(destination, "group_submission.zip", [destination / "submission.jsonl", destination / "run_metadata.json"])
    files = [p for p in sorted(destination.rglob("*")) if p.is_file()
             and not p.name.endswith((".tmp", ".zip", ".sha256"))]
    diagnostic = make_bundle(destination, "diagnostic_bundle.zip", files)
    return summary, group, diagnostic


def run_team200(agent, participant_root, output_dir, reference_zip, *, participant,
                load_seconds, load_session_id, source_archive_sha256,
                limit=None, wall_budget_seconds=10800, units_used=None, invocation_id=None):
    if getattr(agent.backend, "counts_as_model", True) is False:
        raise DataMateError("Cần model thật; không dùng demo để chạy benchmark")
    if getattr(agent, "pipeline_version", 0) < MIN_PIPELINE_VERSION:
        raise DataMateError("Agent cũ trong RAM; cần pipeline >=6. Restart kernel và nạp kit/notebook mới")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 200):
        raise DataMateError("limit phải là None hoặc số nguyên trong 1..200")
    if load_seconds < 0 or wall_budget_seconds <= 0 or not participant.strip():
        raise DataMateError("Tên và ngân sách/thời gian không hợp lệ")
    if units_used is not None and (type(units_used) not in (int, float) or not math.isfinite(units_used) or units_used < 0):
        raise DataMateError("Unit phải là số không âm hoặc None; không suy unit thực từ mức unit/giờ")
    plan, benchmark = load_plan(agent, participant_root)
    destination = ensure_outside_dataset(output_dir, agent.split.root)
    destination.mkdir(parents=True, exist_ok=True)
    original = agent.backend
    runtime = runtime_snapshot(original)
    model_manifest = Path(agent.settings.model_dir) / "artifact_manifest.json"
    identity = {"runner_version": RUNNER_VERSION, "pipeline_version": agent.pipeline_version, "benchmark_id": BENCHMARK_ID,
                "participant": participant, "question_ids": [q.id for q in plan],
                "questions_sha256": QUESTIONS_SHA256, "settings": agent.settings.to_dict(),
                "source_hashes": source_hashes(), "runner_sha256": sha256(__file__),
                "source_archive_sha256": source_archive_sha256, "model_manifest_sha256": sha256(model_manifest),
                "runtime_identity": {k: runtime.get(k) for k in ("packages", "gpus", "dtype", "generation_tokens", "effective_generation")}}
    manifest_path = destination / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != identity:
            raise DataMateError("Không resume: code/model/config/data/runtime khác lượt cũ. Đổi RUN_NAME hoặc khôi phục đúng cấu hình.")
    elif any(destination.iterdir()):
        raise DataMateError("Output có file nhưng thiếu manifest; dùng RUN_NAME mới")
    save_json(manifest_path, identity)
    logs = destination / "logs"
    logs.mkdir(exist_ok=True)
    records = {}
    allowed = {q.id for q in plan}
    for path in sorted(logs.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        qid = row.get("id")
        if qid not in allowed or qid in records or row.get("submission", {}).get("id") != qid:
            raise DataMateError("Log resume trùng/sai ID")
        Answer.from_dict(row["submission"])
        records[qid] = row
    existing = destination / "submission.jsonl"
    canonical = [records[q.id]["submission"] for q in plan if q.id in records]
    if existing.exists():
        submitted = read_jsonl(existing)
        if submitted != canonical[:len(submitted)]:
            raise DataMateError("Submission khác log; không tự ghi đè một kết quả đã bị sửa")
    save_jsonl(existing, canonical)
    for qid, record in records.items():
        if not (logs / (qid + ".md")).exists():
            write_debug_log(logs / (qid + ".md"), record)
    save_jsonl(destination / "selected_questions.jsonl", [{"id": q.id, "business_id": q.business_id, **q.prompt_dict()} for q in plan])
    ledger_path = destination / "session_history.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"loads": {}, "batches": []}
    if load_session_id in ledger["loads"] and ledger["loads"][load_session_id] != load_seconds:
        raise DataMateError("Cùng load_session_id nhưng thời gian load khác; nạp lại model hoặc dùng giá trị gốc")
    ledger["loads"][load_session_id] = load_seconds
    entry = {"id": invocation_id or uuid.uuid4().hex, "started_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": 0, "closed": False}
    if any(previous["id"] == entry["id"] for previous in ledger["batches"]):
        raise DataMateError("Invocation ID bị dùng lại; chạy lại ô batch để tạo lượt mới")
    ledger["batches"].append(entry)
    save_json(ledger_path, ledger)
    target = len(plan) if limit is None else limit
    timer = TimedBackend(original)
    started = time.monotonic()
    summary = group = diagnostic = None
    try:
        agent.backend = timer
        for index, q in enumerate(plan[:target], 1):
            if q.id in records:
                print(f"[{index}/{target}] resume: {q.id} đã có", flush=True)
                continue
            remaining = wall_budget_seconds - (time.monotonic() - started)
            if remaining <= 0:
                print("Hết ngân sách phiên; sẽ lưu bộ kết quả partial để resume.", flush=True)
                break
            timer.calls = []
            print(f"[{index}/{target}] {q.business_id} | {q.question}", flush=True)
            result = agent.run(q, remaining_seconds=remaining)
            row = {"id": q.id, "business_id": q.business_id, **q.prompt_dict(),
                   "rulebook_sha256": agent.split.rulebook_hash, **result.to_dict(q.id),
                   "generation_calls": list(timer.calls)}
            save_json(logs / (q.id + ".json"), row)
            records[q.id] = row
            write_debug_log(logs / (q.id + ".md"), row)
            save_jsonl(existing, [records[item.id]["submission"] for item in plan if item.id in records])
            entry["elapsed_seconds"] = round(time.monotonic() - started, 4)
            save_json(ledger_path, ledger)
            print(f"  -> {result.answer.type} | {result.elapsed_seconds:.2f}s | {result.model_calls} calls", flush=True)
    finally:
        agent.backend = original
        entry["elapsed_seconds"] = round(time.monotonic() - started, 4)
        entry["closed"] = True
        save_json(ledger_path, ledger)
        summary, group, diagnostic = write_outputs(destination, plan, records, runtime, ledger,
                                                   reference_zip, participant, units_used)
        save_json(destination / "report_receipt.json", {"invocation_id": entry["id"],
                  "group_sha256": sha256(group), "diagnostic_sha256": sha256(diagnostic)})
        print(json_text(summary, indent=2), flush=True)
        print("NỘP NHÓM:", group, flush=True)
        print("REVIEW + LOG:", diagnostic, flush=True)
    return summary, group, diagnostic


def finish_and_disconnect(output_dir, *, enabled=True):
    """Only release the VM after both verified bundles are persisted on Drive."""
    output_dir = Path(output_dir)
    for name in ("group_submission.zip", "diagnostic_bundle.zip"):
        path = output_dir / name
        digest_path = output_dir / (name + ".sha256")
        if not path.is_file() or not digest_path.is_file() or sha256(path) != digest_path.read_text().split()[0]:
            raise DataMateError("Chưa xác minh đủ hai ZIP; giữ runtime để tránh mất kết quả")
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise DataMateError("ZIP bị lỗi; chưa ngắt runtime")
    if enabled:
        disconnect_colab_after_report(output_dir / "diagnostic_bundle.zip")
