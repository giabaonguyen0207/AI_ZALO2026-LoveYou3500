"""Copy this entire file into ONE cell below the loaded agent, then run it.

Diagnostic inference only. Reuses the current model; does not train or change
prompts/tools/decoding. The default draws uniformly from all training questions.
Reference labels never select that sample or enter the agent; scoring is later.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
import unicodedata
import zipfile

from datamate.data.split import ensure_outside_dataset, read_jsonl
from datamate.domain import DataMateError, json_text
from datamate.debug_log import write_debug_log
from datamate.evaluation.scoring import score
from datamate.models.artifacts import sha256
from datamate.training.dataset import business_holdout, require_approved_source

DIAGNOSTIC_VERSION = 4

# CẤU HÌNH LƯỢT CHẠY: chỉ cần execute một lần sau ô load model.
SAMPLE_SIZE = 100
SAMPLING_MODE = "random"
SAMPLING_SEED = 20261103      # Chỉ để lặp lại đúng mẫu; không đổi seed của model.
MAX_RUN_MINUTES = 90           # Mục tiêu khoảng 15 Unit ở mức 10 Unit/giờ; trần mềm.
AUTO_DISCONNECT = False        # True: thử đồng bộ Drive và ngắt Colab sau khi xuất ZIP.
RUN_NAME_SUFFIX = "random100-round3-01"  # Đổi tên nếu thay model/config/code.
KNOWN_CASE = None             # Không ép đưa câu lỗi cũ vào mẫu ngẫu nhiên.


def question_profile(text):
    original = unicodedata.normalize("NFC", text).lower()
    plain = "".join(ch for ch in unicodedata.normalize("NFD", original)
                    if not unicodedata.combining(ch)).replace("đ", "d")
    tokens = set(re.findall(r"\w+", plain))
    abbreviations = sorted(tokens & {"dthu", "dt", "bn", "bnhiu", "bnhiêu", "bnhieu", "nhiu",
                                    "dc", "sp", "sl", "hqua", "hnay", "trc", "vs", "hsd", "tb"})
    patterns = {
        "comparison": (2, r"cung ky|so voi|so sanh|chenh lech|tang hay giam|hon hay kem|ngay lien truoc|ngay ngay truoc"),
        "multi_level_aggregation": (3, r"(?:khach|san pham|mat hang|ngay).*(?:it nhat|tro len|dat .*dong|tu \d+ don)|co tu \d+"),
        "ratio": (1, r"(?:ti|ty) (?:le|trong)|phan tram|%"),
        "debt_or_cost": (2, r"lai (?:gop|rong)|gia von|tuoi no|tra no|con no|chua tra|ton kho|con ton"),
        "date_or_calendar": (1, r"\btet\b|am lich|thang chap|\bquy\b|thang.*lien nhau|\d+ ngay (?:ke tu|gan nhat)|ca (?:sang|chieu)"),
        "unit_conversion": (2, r"don vi (?:co ban|lon)|quy doi|nguyen.*le|bao nhieu.*(?:thung|hop).*bao nhieu"),
        "ranking_or_series": (1, r"\btop\s*\d|liet ke|moi thang|tung thang|nhieu nhat|cao nhat|ban chay"),
        "ranking_over_aggregates": (2, r"(?:ngay nao|thang nao).*(?:cao nhat|nhieu nhat)|(?:cao nhat|nhieu nhat).*vao (?:ngay|thang)"),
    }
    cues = [name for name, (_, pattern) in patterns.items() if re.search(pattern, plain)]
    score_hint = sum(patterns[name][0] for name in cues) + int(len(plain.split()) >= 22)
    return {"no_diacritics": original == plain and bool(re.search(r"[a-z]", plain)),
            "abbreviated": bool(abbreviations), "abbreviation_tokens": abbreviations,
            "complex_hint": score_hint >= 2, "complexity_score_hint": score_hint,
            "complexity_cues": cues}


def language_stress_plan(questions, reference_types, seed, sample_size, probe):
    development, holdout = business_holdout(questions, .2, seed)
    pool = sorted((q for q in questions if q.business_id in development), key=lambda q: q.id)
    if sample_size != 3 * len(development):
        raise DataMateError("language_stress_3 cần đúng 3 câu mỗi development business (72 câu với train hiện tại)")
    rng = random.Random(seed)
    rng.shuffle(pool)
    profiles = {q.id: question_profile(q.question) for q in pool}
    counts, type_counts, selected = Counter(), Counter(), []
    businesses = sorted(development)
    rng.shuffle(businesses)

    def take(candidates, prefer_abbreviation=False):
        if not candidates:
            raise DataMateError("Không đủ câu để chọn mẫu ngôn ngữ")
        q = min(candidates, key=lambda q: (
            -int(prefer_abbreviation and profiles[q.id]["abbreviated"]),
            type_counts[reference_types[q.id]], -profiles[q.id]["complexity_score_hint"]))
        selected.append(q)
        pool.remove(q)
        counts[q.business_id] += 1
        type_counts[reference_types[q.id]] += 1

    known = next((q for q in pool if probe == (q.business_id, q.as_of_date, q.question)), None)
    if known:
        take([known])
    for kind in ("no_diacritics", "complex_hint", "abbreviated"):
        for business in businesses:
            if counts[business] >= 3:
                continue
            existing = [q for q in selected if q.business_id == business]
            if any(profiles[q.id][kind] for q in existing):
                continue
            candidates = [q for q in pool if q.business_id == business and profiles[q.id][kind]]
            if not candidates:
                raise DataMateError(f"Không có câu thuộc nhóm {kind} cho {business}; cần xem lại tiêu chí, không tự bịa câu")
            take(candidates, prefer_abbreviation=kind == "no_diacritics")
    for business in businesses:
        while counts[business] < 3:
            take([q for q in pool if q.business_id == business])
    if len(selected) != sample_size or set(reference_types[q.id] for q in selected) != set(reference_types[q.id] for q in questions if q.business_id in development):
        raise DataMateError("Mẫu chưa đủ số câu/type; cần điều chỉnh chọn mẫu trước khi dùng GPU")
    for business in businesses:
        chosen = [profiles[q.id] for q in selected if q.business_id == business]
        if not all(any(profile[kind] for profile in chosen) for kind in ("no_diacritics", "abbreviated", "complex_hint")):
            raise DataMateError(f"Mẫu của {business} chưa phủ đủ nhóm ngôn ngữ/độ khó")
    return selected, sorted(development), sorted(holdout), known.id if known else None


def diagnostic_plan(questions, reference_types, seed=42, sample_size=120, probe=None, sampling_mode="balanced"):
    if sampling_mode == "random":
        if type(sample_size) is not int or not 1 <= sample_size <= len(questions):
            raise DataMateError("sample_size phải là số nguyên từ 1 đến tổng số câu train")
        pool = sorted(questions, key=lambda q: q.id)
        if len({q.id for q in pool}) != len(pool):
            raise DataMateError("Không chọn mẫu từ danh sách trùng question ID")
        # No reference types, text profiles, business quotas, forced cases,
        # previous-run exclusions or holdout filtering affect this draw.
        return random.Random(seed).sample(pool, sample_size), sorted({q.business_id for q in pool}), [], None
    if sampling_mode == "language_stress_3":
        return language_stress_plan(questions, reference_types, seed, sample_size, probe)
    if sampling_mode != "balanced":
        raise DataMateError("sampling_mode không hợp lệ")
    if sample_size < 1:
        raise DataMateError("sample_size phải dương")
    development, holdout = business_holdout(questions, 0.2, seed)
    pool = sorted((q for q in questions if q.business_id in development), key=lambda q: q.id)
    if any(q.id not in reference_types for q in pool):
        raise DataMateError("Train thiếu type tham chiếu khi chọn mẫu")
    target = min(sample_size, len(pool))
    if target < len(development):
        raise DataMateError("sample_size cần đủ ít nhất một câu mỗi development business")
    rng = random.Random(seed)
    businesses = sorted(development)
    rng.shuffle(businesses)
    rng.shuffle(pool)
    selected, type_counts, business_counts = [], Counter(), Counter()

    def take(candidates):
        question = min(candidates, key=lambda q: (type_counts[reference_types[q.id]], business_counts[q.business_id]))
        selected.append(question)
        pool.remove(question)
        type_counts[reference_types[question.id]] += 1
        business_counts[question.business_id] += 1

    known = next((q for q in questions if probe == (q.business_id, q.as_of_date, q.question)
                  and q.business_id in development), None)
    if known:
        take([known])
    for business in businesses:
        if not business_counts[business]:
            take([q for q in pool if q.business_id == business])
    while len(selected) < target:
        take(pool)
    return selected, sorted(development), sorted(holdout), known.id if known else None


def reference_type_index(path):
    # Only ID/type metadata are retained for stratification. Gold answers are
    # not supplied to the agent; values are used by write_report after inference.
    with Path(path).open(encoding="utf-8-sig") as handle:
        return {row["id"]: row["type"] for line in handle if line.strip()
                for row in [json.loads(line)]}


def runtime_snapshot(backend):
    import datamate
    result = {"python": sys.version, "backend": type(backend).__name__, "packages": {}}
    result['datamate_code_version']=datamate.__version__
    result['effective_generation']={'do_sample':False,'num_beams':1,'protocol_version':getattr(backend,'protocol_version',None)}
    for package in ("torch", "transformers", "accelerate", "bitsandbytes", "datamate-agent"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    torch = getattr(backend, "torch", None)
    if torch is not None:
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_version"] = torch.version.cuda
        if result["cuda_available"]:
            result["gpus"] = [{"name": torch.cuda.get_device_name(i),
                               "total_vram_gib": torch.cuda.get_device_properties(i).total_memory / 2**30}
                              for i in range(torch.cuda.device_count())]
    model = getattr(backend, "model", None)
    if model is not None:
        result["hf_device_map"] = {key: str(value) for key, value in getattr(model, "hf_device_map", {}).items()}
        result["parameter_devices"] = sorted({str(p.device) for p in model.parameters()})
        result["dtype"] = str(getattr(model, "dtype", "unknown"))
        generation = getattr(model, "generation_config", None)
        if generation is not None:
            result["generation_config"] = {name: getattr(generation, name, None)
                                           for name in ("do_sample", "num_beams", "eos_token_id", "pad_token_id")}
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is not None:
        result["tokenizer_class"] = type(tokenizer).__name__
        result["chat_template"] = getattr(tokenizer, "chat_template", None)
    return result


def source_hashes():
    import datamate
    root=Path(datamate.__file__).parent
    return {str(p.relative_to(root).with_suffix('')).replace('\\','.').replace('/','.'):sha256(p)
            for p in sorted(root.rglob('*.py'))}


class TimedBackend:
    def __init__(self, delegate):
        self.delegate, self.calls = delegate, []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def generate(self, messages, remaining_seconds):
        started = time.monotonic()
        event = {"input_characters": sum(len(m.get("content", "")) for m in messages)}
        try:
            output = self.delegate.generate(messages, remaining_seconds)
            event["output_characters"] = len(output)
            return output
        except Exception as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            event["seconds"] = round(time.monotonic() - started, 4)
            generation=getattr(self.delegate,'last_generation',None)
            if generation:
                event['generation']=generation
            self.calls.append(event)


def diagnostic_flags(record, reference):
    flags = set()
    predicted = record["submission"]
    if predicted["type"] != "abstain" and predicted["type"] != reference["type"]:
        flags.add("type_mismatch")
    if predicted["type"] == "unanswerable" and reference["type"] != "unanswerable":
        flags.add("false_unanswerable")
    for event in record.get("trace", []):
        action, observation = event.get("action", {}), event.get("observation", {})
        error = observation.get("error") or event.get("error", "")
        if error:
            kind=observation.get('error_kind') or event.get('error_kind')
            if kind:
                flags.add(kind)
            if "Context vượt" in error:
                flags.add("context_limit")
            elif action.get("action") == "sql":
                flags.add("sql_error")
            elif "raw" in event and not action:
                flags.add("action_parse_error")
            else:
                flags.add("runtime_or_tool_error")
        if observation.get("truncated"):
            flags.add("truncated_sql_result")
        if event.get('loop_stopped'):
            flags.add('repeated_action_stopped')
    if predicted["type"] == "abstain":
        flags.add("abstain")
        if "ngân sách" in predicted.get("reason", "").lower():
            flags.add("step_or_time_budget")
    return sorted(flags)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_report(destination, plan, manifest, target, runtime, batch_seconds):
    # Gold labels enter this reporting function only, after predictions exist.
    references = {r["id"]: r for r in read_jsonl(Path(manifest["settings"]["split_root"]) / "answers.jsonl")}
    records = [json.loads((destination / "logs" / (q.id + ".json")).read_text(encoding="utf-8"))
               for q in plan if (destination / "logs" / (q.id + ".json")).is_file()]
    for record in records:
        readable = destination / 'logs' / (record['id'] + '.md')
        if not readable.is_file():
            write_debug_log(readable, record)
    ids = {row["id"] for row in records}
    metrics = score([r["submission"] for r in records], list(references.values()), ids)
    outcomes = {row["id"]: row["result"] for row in metrics.pop("details")}
    review = [{**r, "reference": references[r["id"]], "outcome": outcomes[r["id"]],
               "question_profile": question_profile(r["question"]),
               "flags": diagnostic_flags(r, references[r["id"]])} for r in records]
    times = sorted(r["elapsed_seconds"] for r in records)
    groups = defaultdict(list)
    for row in records:
        groups[references[row["id"]]["type"]].append(row)
    by_type = {kind: score([r["submission"] for r in rows], list(references.values()), {r["id"] for r in rows})
               for kind, rows in groups.items()}
    for value in by_type.values():
        value.pop("details")
    by_language = {}
    for group in ("no_diacritics", "abbreviated", "complex_hint"):
        subset = [r for r in review if r["question_profile"][group]]
        value = score([r["submission"] for r in subset], list(references.values()), {r["id"] for r in subset})
        value.pop("details")
        by_language[group] = value
    summary = {**metrics, "diagnostic_only": True, "model_key": manifest["settings"]["model_key"],
               "planned_questions": len(plan), "requested_cumulative_limit": target,
               "completed_questions": len(records), "remaining_questions": len(plan) - len(records),
               "completed_businesses": len({r["business_id"] for r in records}),
               "status": "complete" if len(records) >= target else "partial",
               "planned_type_counts": manifest["planned_type_counts"],
               "planned_language_counts": manifest["planned_language_counts"],
               "by_question_group": by_language,
               "per_business_counts": dict(Counter(r["business_id"] for r in records)),
               "predicted_type_counts": dict(Counter(r["submission"]["type"] for r in records)),
               "mean_seconds": statistics.mean(times) if times else None,
               "median_seconds": statistics.median(times) if times else None,
               "p95_seconds": times[max(0, math.ceil(.95 * len(times)) - 1)] if times else None,
               "total_model_calls": sum(r["model_calls"] for r in records),
               "flag_counts": dict(Counter(flag for r in review for flag in r["flags"])),
               "by_reference_type": by_type, "this_invocation_seconds": round(batch_seconds, 3),
               "model_load_excluded": True,
               "limitations": "Mẫu development chủ động chọn theo type/ngôn ngữ và tín hiệu độ khó, có thể gồm câu lỗi đã biết; không đại diện tỉ lệ câu toàn train. Nhóm có thể chồng lấp; complex_hint là heuristic, không phải độ khó đã đo. PA chỉ tính câu đã chạy."}
    save_json(destination / "summary.json", summary)
    save_json(destination / "runtime.json", runtime)
    (destination / "review.jsonl").write_text("".join(json_text(r) + "\n" for r in review), encoding="utf-8")
    (destination / "submission.jsonl").write_text("".join(json_text(r["submission"]) + "\n" for r in records), encoding="utf-8")
    with (destination / "review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["id", "business_id", "question", "as_of_date", "outcome", "predicted_type", "reference_type",
                  "predicted_answer", "reference_answer", "elapsed_seconds", "model_calls", "flags",
                  "no_diacritics", "abbreviated", "abbreviation_tokens", "complex_hint", "complexity_cues"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in review:
            writer.writerow({**{key: r[key] for key in ("id", "business_id", "question", "as_of_date", "outcome", "elapsed_seconds", "model_calls")},
                             "predicted_type": r["submission"]["type"], "reference_type": r["reference"]["type"],
                             "predicted_answer": json_text(r["submission"].get("answer")),
                             "reference_answer": json_text(r["reference"].get("answer")), "flags": ",".join(r["flags"]),
                             **{key: r["question_profile"][key] for key in ("no_diacritics", "abbreviated", "complex_hint")},
                             "abbreviation_tokens": ",".join(r["question_profile"]["abbreviation_tokens"]),
                             "complexity_cues": ",".join(r["question_profile"]["complexity_cues"])})
    bundle = destination / "diagnostic_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path != bundle and not path.name.endswith(".tmp"):
                archive.write(path, path.relative_to(destination).as_posix())
    return summary, bundle


def disconnect_colab_after_report(bundle):
    """Optional, only called explicitly after a readable report is persisted."""
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise DataMateError("ZIP chưa vượt kiểm tra; chưa ngắt Colab")
    import os
    if not os.environ.get("TBE_RUNTIME_ADDR"):
        print("Không có API quản lý runtime Colab. Hãy ngắt runtime GPU thủ công sau khi lấy REPORT.", flush=True)
        return
    try:
        from google.colab import drive, runtime
        print("Đang đồng bộ các ghi chờ trên Drive trước khi kết thúc runtime.", flush=True)
        drive.flush_and_unmount(timeout_ms=60000)
        print("Đã ghi REPORT; đang yêu cầu Colab kết thúc runtime để tiết kiệm Unit.", flush=True)
        runtime.unassign()
    except Exception as exc:
        print(f"Chưa xác nhận tự ngắt được runtime: {exc}. Hãy kiểm tra và ngắt thủ công.", flush=True)


def run_diagnostic(agent, output_dir, sources, limit=None, wall_budget_seconds=10800,
                   seed=42, sample_size=120, probe=None, sampling_mode="balanced"):
    if (limit is not None and limit < 1) or wall_budget_seconds <= 0:
        raise DataMateError("limit và wall_budget_seconds phải dương")
    if getattr(agent.backend, "counts_as_model", True) is False:
        raise DataMateError("Chẩn đoán này cần model thật; không chạy Demo")
    if getattr(agent,'pipeline_version',0)<2:
        raise DataMateError('Agent trong RAM là bản cũ. Thay source ZIP và khởi động lại kernel, rồi load agent trước batch v2')
    if type(agent.backend).__name__=='TransformersBackend' and getattr(agent.backend,'protocol_version',0)<2:
        raise DataMateError('Backend trong RAM là bản cũ; khởi động lại kernel sau khi cập nhật source')
    original = agent.backend
    if getattr(original, "settings", agent.settings).to_dict() != agent.settings.to_dict():
        raise DataMateError("Config agent và backend không khớp; load lại model trước")
    split = agent.split
    approved = require_approved_source(split.root, sources)
    reference_types = {} if sampling_mode == "random" else reference_type_index(split.root / "answers.jsonl")
    plan, development, holdout, known = diagnostic_plan(split.questions(), reference_types, seed, sample_size, probe, sampling_mode)
    if sampling_mode == "random":
        # Read ID/type metadata for reporting only AFTER the sample is fixed.
        reference_types = reference_type_index(split.root / "answers.jsonl")
    settings = {**agent.settings.to_dict(), "split_root": str(split.root.resolve())}
    model_manifest = Path(settings["model_dir"]) / "artifact_manifest.json"
    if not model_manifest.is_file():
        raise DataMateError("Không thấy manifest của model đang dùng; kiểm tra model_dir")
    identity = {"diagnostic_version": DIAGNOSTIC_VERSION, "settings": settings,
                "source_fingerprint": approved["fingerprint"], "source_hashes": source_hashes(),
                "model_manifest_sha256": sha256(model_manifest), "seed": seed,
                "development_businesses": development if sampling_mode != "random" else None,
                "reserved_holdout_businesses": holdout,
                "sampling_population_businesses": development,
                "sampling_scope": "all_training_questions" if sampling_mode == "random" else "development_businesses",
                "sampling": sampling_mode,
                "planned_type_counts": dict(Counter(reference_types[q.id] for q in plan)),
                "planned_language_counts": {key: sum(question_profile(q.question)[key] for q in plan)
                                            for key in ("no_diacritics", "abbreviated", "complex_hint")},
                "known_case_id": known, "question_ids": [q.id for q in plan]}
    digest = hashlib.sha256(json_text(identity, sort_keys=True).encode()).hexdigest()
    destination = ensure_outside_dataset(output_dir, split.root)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8"))["identity_hash"] != digest:
            raise DataMateError("Lượt cũ khác model/config/code/data/sample. Đổi RUN_NAME; không trộn benchmark.")
    elif any(destination.iterdir()):
        raise DataMateError("Output có file nhưng thiếu manifest; dùng RUN_NAME mới")
    manifest = {**identity, "identity_hash": digest}
    save_json(manifest_path, manifest)
    (destination / "logs").mkdir(exist_ok=True)
    (destination / "selected_questions.jsonl").write_text(
        "".join(json_text({"id": q.id, "business_id": q.business_id, **q.prompt_dict(),
                          "question_profile": question_profile(q.question)}) + "\n" for q in plan), encoding="utf-8")
    runtime = runtime_snapshot(original)
    print("Model:", settings["model_key"], "quantization:", settings["quantization"], flush=True)
    print("GPU/devices:", json_text({k: v for k, v in runtime.items() if k != "chat_template"}), flush=True)
    target = len(plan) if limit is None else min(limit, len(plan))
    print(f"Plan: {len(plan)} câu; lượt này chạy đến câu {target}.", flush=True)
    if sampling_mode == 'random':
        print(f"Ngẫu nhiên đều từ toàn bộ train; seed={seed}. Không cân đối type/doanh nghiệp/ngôn ngữ, không lọc holdout hoặc câu cũ.", flush=True)
    else:
        print(f"Giữ riêng {len(holdout)} doanh nghiệp holdout.", flush=True)
    print("Loại đáp án:", json_text(identity["planned_type_counts"]), flush=True)
    print("Nhóm câu hỏi (có thể chồng lấp):", json_text(identity["planned_language_counts"]), flush=True)
    print("Giữ nguyên prompt, SQL tools, decoding và ngân sách của agent hiện tại.", flush=True)
    timer, started = TimedBackend(original), time.monotonic()
    try:
        agent.backend = timer
        for index, question in enumerate(plan[:target], 1):
            path = destination / "logs" / (question.id + ".json")
            if path.exists():
                print(f"[{index}/{target}] đã có {question.id}, bỏ qua", flush=True)
                continue
            remaining = wall_budget_seconds - (time.monotonic() - started)
            if remaining <= 0:
                print("Hết ngân sách lượt chạy; có thể chạy lại để tiếp tục.", flush=True)
                break
            timer.calls = []
            print(f"[{index}/{target}] {question.business_id} | {question.as_of_date} | {question.question}", flush=True)
            result = agent.run(question, remaining_seconds=remaining)
            record = {"id": question.id, "business_id": question.business_id, **question.prompt_dict(),
                      **result.to_dict(question.id), "generation_calls": list(timer.calls)}
            save_json(path, record)
            write_debug_log(path.with_suffix('.md'), record)
            print(f"  -> {result.answer.type} | {result.elapsed_seconds:.1f}s | {result.model_calls} model calls", flush=True)
    finally:
        agent.backend = original
        summary, bundle = write_report(destination, plan, manifest, target, runtime, time.monotonic() - started)
        print(json_text(summary, indent=2), flush=True)
        print("REPORT:", bundle, flush=True)
    return summary, bundle


if __name__ == "__main__":
    # Copy the ENTIRE file to the current notebook, below its model-loading cell.
    # Do not run it in a new kernel: it reuses the existing `agent`.
    if any(name not in globals() for name in ("agent", "WORK", "SAVE")):
        raise RuntimeError("Dán toàn bộ file vào ô mới trong notebook inference đã load model (cần agent, WORK, SAVE).")
    RUN_NAME = agent.settings.model_key + "-" + RUN_NAME_SUFFIX
    diagnostic_summary, diagnostic_bundle = run_diagnostic(
        agent, Path(SAVE) / "diagnostics" / RUN_NAME,
        sources=Path(WORK) / "configs/training_sources.json",
        sample_size=SAMPLE_SIZE, wall_budget_seconds=MAX_RUN_MINUTES * 60, probe=KNOWN_CASE,
        sampling_mode=SAMPLING_MODE, seed=SAMPLING_SEED,
    )
    if AUTO_DISCONNECT:
        disconnect_colab_after_report(diagnostic_bundle)
