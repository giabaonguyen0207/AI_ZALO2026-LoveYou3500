"""Verify a frozen benchmark or compare submissions; Python 3.11+, no pip installs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: JSON không hợp lệ") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: cần JSON object")
                rows.append(row)
    return rows


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")


def indexed(rows, label):
    result = {}
    for row in rows:
        qid = row.get("id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"{label}: thiếu ID chuỗi không rỗng")
        if qid in result:
            raise ValueError(f"{label}: trùng ID {qid}")
        result[qid] = row
    return result


def valid_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def verify(root):
    root = Path(root).resolve()
    manifest = read_json(root / "benchmark_manifest.json")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"File thiếu/khác hash: {name}")
    questions_path = root / manifest["questions_path"]
    questions = read_jsonl(questions_path)
    ids = read_json(root / "question_ids.json")
    if (len(questions) != manifest["sample_size"] or len(set(ids)) != len(ids)
            or ids != [q["id"] for q in questions]):
        raise ValueError("Số câu hoặc thứ tự ID khác manifest")
    indexed(questions, "questions")
    if any(set(q) != {"id", "business_id", "question", "as_of_date"} for q in questions):
        raise ValueError("Câu hỏi phải chỉ có bốn trường đầu vào")
    if manifest["package_role"] == "participant" and (root / "dataset/answers.jsonl").exists():
        raise ValueError("Gói participant không được có answers.jsonl trong dataset")
    return manifest, questions


def scorer():
    # Release ZIPs carry these exact source files. Never use an installed copy
    # in preference to the frozen evaluator bundled with the benchmark.
    here = Path(__file__).resolve().parent
    vendor = here / "_vendor"
    sys.path.insert(0, str(vendor if vendor.is_dir() else here.parent / "src"))
    from datamate.domain import Answer, DataMateError
    from datamate.evaluation.scoring import matches, score
    return Answer, DataMateError, matches, score


def latency_percentile(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def load_run(name, folder, manifest, questions, references):
    Answer, DataMateError, matches, score = scorer()
    folder = Path(folder).resolve()
    predictions = indexed(read_jsonl(folder / "submission.jsonl"), name)
    ids = {q["id"] for q in questions}
    extra = predictions.keys() - ids
    if extra:
        raise ValueError(f"{name}: có ID ngoài benchmark: {sorted(extra)[:3]}")
    valid, invalid = {}, {}
    for qid, prediction in predictions.items():
        try:
            Answer.from_dict(prediction)
            valid[qid] = prediction
        except (DataMateError, ValueError, TypeError, KeyError) as exc:
            # Invalid payloads are wrong (-1), including a malformed abstain.
            # Preserve the original payload in the per-question comparison.
            invalid[qid] = str(exc)
    scoring_rows = [valid.get(qid, {"id": qid, "type": "__invalid__"}) for qid in predictions]
    report = score(scoring_rows, references, ids)
    details = {row["id"]: row for row in report.pop("details")}
    logs = {}
    warnings = []
    logs_dir = folder / "logs"
    if logs_dir.exists():
        for path in sorted(logs_dir.glob("*.json")):
            record = read_json(path)
            qid = record.get("id")
            if qid not in ids or qid in logs:
                raise ValueError(f"{name}: log có ID ngoài benchmark hoặc trùng: {qid}")
            logged_prediction = record.get("submission", record.get("final_answer"))
            if logged_prediction is not None and qid in valid:
                try:
                    equal = matches(logged_prediction, valid[qid]) and matches(valid[qid], logged_prediction)
                except (ValueError, TypeError, KeyError):
                    equal = False
                if not equal:
                    warnings.append(f"{qid}: đáp án trong log khác submission")
            logs[qid] = record
    latencies = [record["elapsed_seconds"] for record in logs.values()
                 if valid_number(record.get("elapsed_seconds"))]
    call_values = [record["model_calls"] for record in logs.values()
                   if type(record.get("model_calls")) is int and record["model_calls"] >= 0]
    metadata_path = folder / "run_metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    if any(key in metadata and metadata[key] != manifest[key]
           for key in ("benchmark_id", "questions_sha256")):
        raise ValueError(f"{name}: run_metadata không khớp benchmark")
    if not metadata:
        warnings.append("Thiếu run_metadata.json; điểm vẫn chấm, thời gian/phần cứng chưa đủ đối chiếu")
    if logs and len(latencies) != len(predictions):
        warnings.append(f"Có timing hợp lệ cho {len(latencies)}/{len(predictions)} dự đoán")
    timing = metadata.get("timing") or {}
    units = metadata.get("compute_units") or {}
    load_seconds = metadata.get("load_seconds", timing.get("load_seconds"))
    inference_seconds = metadata.get("inference_seconds", timing.get("inference_wall_seconds"))
    for label, value in (("load_seconds", load_seconds), ("inference_seconds", inference_seconds)):
        if value is not None and not valid_number(value):
            raise ValueError(f"{name}: {label} phải là số không âm hoặc null")
    end_to_end = (load_seconds + inference_seconds
                  if load_seconds is not None and inference_seconds is not None
                  else timing.get("end_to_end_seconds"))
    batch_mean = (inference_seconds / len(ids)
                  if inference_seconds is not None and len(predictions) == len(ids) else None)
    mean = batch_mean if batch_mean is not None else (sum(latencies) / len(latencies) if latencies else None)
    mean_source = ("batch_wall_time_divided_by_question_count" if batch_mean is not None
                   else "per_question_logs" if latencies else None)
    if inference_seconds is not None and len(predictions) != len(ids):
        warnings.append("Nộp chưa đủ câu: không lấy thời gian batch chia cho toàn bộ số câu benchmark")
    row = {"participant": name, **report,
           "answered_precision": report["correct"] / (report["correct"] + report["wrong"])
           if report["correct"] + report["wrong"] else None,
           "submitted": len(predictions), "missing": len(ids - predictions.keys()),
           "invalid": len(invalid), "timed_questions": len(latencies),
           "mean_seconds": mean, "mean_seconds_source": mean_source,
           "p50_seconds": latency_percentile(latencies, .50),
           "p95_seconds": latency_percentile(latencies, .95),
           "max_seconds": max(latencies) if latencies else None,
           "sum_question_seconds": sum(latencies) if latencies else None,
           "model_calls": sum(call_values) if call_values else None,
           "end_to_end_seconds": end_to_end, "load_seconds": load_seconds,
           "inference_wall_seconds": inference_seconds,
           "models": metadata.get("models"), "platform": metadata.get("platform"), "gpu": metadata.get("gpu"),
           "compute_units_used": metadata.get("compute_units_used", units.get("used")),
           "submission_sha256": sha256(folder / "submission.jsonl"), "warnings": warnings}
    return {"summary": row, "predictions": predictions, "valid": valid, "invalid": invalid,
            "logs": logs, "details": details, "metadata": metadata}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def compare(benchmark, runs, output):
    manifest, questions = verify(benchmark)
    if manifest["package_role"] != "coordinator":
        raise ValueError("Chấm bằng gói coordinator có đáp án chuẩn")
    references = read_jsonl(Path(benchmark) / "answers.jsonl")
    ref_index = indexed(references, "answers")
    if set(ref_index) != {q["id"] for q in questions}:
        raise ValueError("Đáp án chuẩn không khớp 200 câu")
    names = [name for name, _ in runs]
    if not runs or len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("Tên thành viên phải duy nhất, chỉ gồm chữ ASCII, số, _ hoặc -")
    loaded = {name: load_run(name, folder, manifest, questions, references) for name, folder in runs}
    _, _, matches, _ = scorer()
    comparison, businesses = [], []
    for question in questions:
        qid = question["id"]
        reference = ref_index[qid]
        row = {**question, "reference_type": reference["type"],
               "reference_answer": reference.get("answer"), "reference_alias": reference.get("alias", [])}
        for name, run in loaded.items():
            prediction = run["predictions"].get(qid, {"type": "abstain"})
            detail, log = run["details"][qid], run["logs"].get(qid, {})
            row.update({f"{name}.type": prediction.get("type"),
                        f"{name}.answer": prediction.get("answer"),
                        f"{name}.result": detail["result"], f"{name}.points": detail["points"],
                        f"{name}.missing": qid not in run["predictions"],
                        f"{name}.invalid": run["invalid"].get(qid),
                        f"{name}.seconds": log.get("elapsed_seconds"),
                        f"{name}.model_calls": log.get("model_calls"), f"{name}.error": log.get("error")})
        comparison.append(row)
    for business in sorted({q["business_id"] for q in questions}):
        selected = [q["id"] for q in questions if q["business_id"] == business]
        for name, run in loaded.items():
            counts = Counter(run["details"][qid]["result"] for qid in selected)
            businesses.append({"business_id": business, "participant": name, "questions": len(selected),
                               "correct": counts["correct"], "wrong": counts["wrong"],
                               "abstain": counts["abstain"],
                               "penalized_accuracy": (counts["correct"] - counts["wrong"]) / len(selected)})
    pairwise = []
    for left, right in itertools.combinations(names, 2):
        a, b = loaded[left], loaded[right]
        both_answered = agreement = a_only = b_only = wins_a = wins_b = 0
        for question in questions:
            qid = question["id"]
            pa, pb = a["valid"].get(qid), b["valid"].get(qid)
            aa = pa is not None and pa["type"] != "abstain"
            ab = pb is not None and pb["type"] != "abstain"
            both_answered += aa and ab
            a_only += aa and not ab
            b_only += ab and not aa
            if aa and ab:
                # Reference aliases can establish equivalence of two different
                # valid strings; otherwise use symmetric local matching.
                agreement += ((a["details"][qid]["result"] == b["details"][qid]["result"] == "correct")
                              or (matches(pa, pb) and matches(pb, pa)))
            sa, sb = a["details"][qid]["points"], b["details"][qid]["points"]
            wins_a += sa > sb
            wins_b += sb > sa
        pairwise.append({"a": left, "b": right, "questions": len(questions),
                         "both_valid_non_abstain": both_answered, "equivalent_answers": agreement,
                         "agreement_when_both_answer": agreement / both_answered if both_answered else None,
                         "only_a_valid_non_abstain": a_only, "only_b_valid_non_abstain": b_only,
                         "a_higher_question_score": wins_a, "b_higher_question_score": wins_b,
                         "pa_a_minus_b": a["summary"]["penalized_accuracy"] - b["summary"]["penalized_accuracy"]})
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Thư mục báo cáo đã có file; chọn output mới")
    destination.mkdir(parents=True, exist_ok=True)
    ranking = sorted((run["summary"] for run in loaded.values()), key=lambda row: -row["penalized_accuracy"])
    report = {"benchmark_id": manifest["benchmark_id"], "questions": len(questions),
              "seed": manifest["seed"], "local_evaluator": True,
              "ranking_policy": "PA giảm dần; đồng điểm giữ đồng hạng, không dùng tốc độ khác GPU để phá hòa",
              "inference_performed_by_this_command": False, "leaderboard": ranking, "pairwise": pairwise}
    save_json(destination / "leaderboard.json", report)
    write_csv(destination / "leaderboard.csv", ranking)
    write_csv(destination / "comparison.csv", comparison)
    write_csv(destination / "by_business.csv", businesses)
    if pairwise:
        write_csv(destination / "pairwise.csv", pairwise)
    lines = [f"# Kết quả benchmark {manifest['benchmark_id']}", "",
             "PA = (đúng − sai) / toàn bộ số câu. Đây là bộ chấm local, không phải điểm chính thức.", "",
             "| Thành viên | Đúng | Sai | Abstain | Thiếu | Invalid | PA | TB giây/câu |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ranking:
        mean = f"{row['mean_seconds']:.2f}" if row['mean_seconds'] is not None else "—"
        lines.append(f"| {row['participant']} | {row['correct']} | {row['wrong']} | {row['abstain']} | {row['missing']} | {row['invalid']} | {row['penalized_accuracy']:.2%} | {mean} |")
        lines.extend(f"\n- {row['participant']}: {warning}" for warning in row['warnings'])
    lines += ["", "Đồng điểm PA là đồng hạng. Thời gian giữa các GPU chỉ để tham khảo.",
              "TB giây/câu = thời gian batch / số câu khi nộp đủ; nếu không có thời gian batch thì dùng log tự nguyện nếu có.",
              "Không có log từng câu thì không có p50/p95/max hoặc thời gian riêng từng câu; vẫn chấm và so được toàn bộ đáp án.",
              "Mẫu lấy từ toàn bộ train, không chứng minh khả năng tổng quát trên private test.",
              "comparison.csv chứa đáp án chuẩn: chỉ người chấm giữ trước khi khóa kết quả."]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Kiểm tra SHA256 và danh sách câu")
    check.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parent)
    grading = commands.add_parser("compare", help="Chấm tập trung một hoặc nhiều submission")
    grading.add_argument("--benchmark", type=Path, default=Path(__file__).resolve().parent)
    grading.add_argument("--run", action="append", required=True, metavar="NAME=RESULT_DIR")
    grading.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "verify":
            manifest, questions = verify(args.benchmark)
            print(f"OK: {manifest['benchmark_id']}; {len(questions)} câu; seed={manifest['seed']}")
        else:
            runs = []
            for spec in args.run:
                name, separator, folder = spec.partition("=")
                if not separator or not folder:
                    raise ValueError("--run phải có dạng NAME=RESULT_DIR")
                runs.append((name, folder))
            report = compare(args.benchmark, runs, args.output)
            print(json.dumps(report["leaderboard"], ensure_ascii=False, allow_nan=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Benchmark: {exc}\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
