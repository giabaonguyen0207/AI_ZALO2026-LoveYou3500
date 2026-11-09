"""Build separate participant/coordinator ZIPs from a frozen random TRAIN sample."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from team_benchmark import indexed, read_jsonl, save_json, sha256, verify


from team_benchmark_instructions import COORDINATOR_README, PARTICIPANT_README, SUBMISSION_EXAMPLES


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
                          encoding="utf-8", newline="\n")


def seal(folder, common, role, questions_path):
    files = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    manifest = {**common, "package_role": role, "questions_path": questions_path, "files": files}
    save_json(folder / "benchmark_manifest.json", manifest)
    verify(folder)
    return manifest


def make_zip(folder, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                archive.write(path, (Path(folder.name) / path.relative_to(folder)).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"ZIP hỏng: {destination}")
        prefix = folder.name + "/"
        manifest = json.loads(archive.read(prefix + "benchmark_manifest.json"))
        for name, digest in manifest["files"].items():
            with archive.open(prefix + name) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                    raise ValueError(f"ZIP khác hash: {name}")
    return {"file": destination.name, "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def prepare(project, source, output, seed=None, size=200, sources=None):
    project, source, output = Path(project).resolve(), Path(source).resolve(), Path(output).resolve()
    if output == source or output.is_relative_to(source):
        raise ValueError("Không ghi benchmark vào thư mục dữ liệu")
    if output.exists():
        raise ValueError("Output đã tồn tại; không bốc lại hoặc ghi đè một bản benchmark đã phát")
    from colab_archives import verify_train_directory
    approved = verify_train_directory(source, sources or project / "configs/training_sources.json")
    questions = read_jsonl(source / "questions.jsonl")
    indexed(questions, "TRAIN questions")
    if type(size) is not int or not 1 <= size <= len(questions):
        raise ValueError("size phải từ 1 đến tổng số câu TRAIN")
    seed = secrets.randbits(32) if seed is None else seed
    # Gold labels and all prior runs are read only AFTER the selection is fixed.
    selected = random.Random(seed).sample(sorted(questions, key=lambda q: q["id"]), size)
    selected = [{key: q[key] for key in ("id", "business_id", "question", "as_of_date")} for q in selected]
    ids = [q["id"] for q in selected]
    references = indexed(read_jsonl(source / "answers.jsonl"), "TRAIN answers")
    if set(references) != {q["id"] for q in questions}:
        raise ValueError("TRAIN questions/answers không khớp ID")
    labels = [references[qid] for qid in ids]
    businesses = sorted({q["business_id"] for q in selected})
    now = datetime.now(timezone.utc)
    # The release date is the user's Vietnam timezone, independent of host TZ.
    from datetime import timedelta
    local_date = (now + timedelta(hours=7)).date().isoformat()
    benchmark_id = f"datamate-train{size}-seed{seed}"
    participant, coordinator = output / "participant", output / "coordinator"
    participant.mkdir(parents=True)
    coordinator.mkdir()
    dataset = participant / "dataset"
    dataset.mkdir()
    write_jsonl(dataset / "questions.jsonl", selected)
    write_jsonl(coordinator / "questions.jsonl", selected)
    write_jsonl(coordinator / "answers.jsonl", labels)
    for name in ("rulebook.md", "calendar.csv"):
        shutil.copyfile(source / name, dataset / name)
    for name in approved["files"]:
        parts = Path(name).parts
        if len(parts) >= 3 and parts[0] == "data" and parts[1] in businesses:
            target = dataset / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
    common = {"benchmark_id": benchmark_id, "protocol_version": 2, "prepared_on_vietnam": local_date,
              "sample_size": size, "population_size": len(questions), "seed": seed,
              "sampling": "uniform_without_replacement_from_all_training_questions",
              "algorithm": "random.Random(seed).sample(questions sorted by id, sample_size)",
              "stratification": None, "forced_questions": [], "excluded_question_ids": [],
              "holdout_excluded": False, "references_used_for_selection": False,
              "source_role": "competition_training", "source_fingerprint": approved["fingerprint"],
              "source_questions_sha256": approved["files"]["questions.jsonl"],
              "questions_sha256": sha256(dataset / "questions.jsonl"),
              "sample_businesses": businesses, "status": "prepared_not_inferred",
              "evaluation": "frozen_local_evaluator_not_official", "inference_performed": False}
    vendor_names = ("__init__.py", "domain.py", "data/__init__.py", "data/transforms.py",
                    "data/split.py", "evaluation/__init__.py", "evaluation/scoring.py")
    for folder in (participant, coordinator):
        save_json(folder / "question_ids.json", ids)
        shutil.copyfile(project / "tools/team_benchmark.py", folder / "benchmark.py")
        for name in vendor_names:
            target = folder / "_vendor/datamate" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / "src/datamate" / name, target)
    values = {"size": size, "population": len(questions), "benchmark_id": benchmark_id,
              "seed": seed, "date": local_date}
    (participant / "README.md").write_text(PARTICIPANT_README.format(**values), encoding="utf-8")
    (coordinator / "README.md").write_text(COORDINATOR_README.format(**values), encoding="utf-8")
    templates = participant / "templates"
    templates.mkdir()
    write_jsonl(templates / "submission.template.jsonl", SUBMISSION_EXAMPLES)
    metadata = {"participant": None, "models": [], "gpu": None,
                "load_seconds": None, "inference_seconds": None, "compute_units_used": None}
    save_json(templates / "run_metadata.template.json", metadata)
    old_sample = project / "outputs/inference-round3-random100/question_ids.json"
    old_ids = set(json.loads(old_sample.read_text(encoding="utf-8"))) if old_sample.exists() else set()
    audit = {"selected_before_reading_labels": True, "no_reselection": True,
             "per_business_counts": dict(sorted(Counter(q["business_id"] for q in selected).items())),
             "reference_type_counts": dict(sorted(Counter(r["type"] for r in labels).items())),
             "overlap_with_previous_random100": sorted(set(ids) & old_ids) if old_ids else None,
             "prior_sample_overlap_used_for_selection": False,
             "independent_holdout": False, "other_teams_training_overlap": "unknown",
             "limitations": "Mẫu ngẫu nhiên từ toàn bộ TRAIN; có thể chứa câu từng fine-tune/debug. Không có inference mới."}
    save_json(coordinator / "selection_audit.json", audit)
    seal(participant, common, "participant", "dataset/questions.jsonl")
    seal(coordinator, common, "coordinator", "questions.jsonl")
    packages = [make_zip(participant, output / "participants.zip"),
                make_zip(coordinator, output / "coordinator-private.zip")]
    message = (f"Cả nhóm chạy cùng bộ {size} câu TRAIN trong participants.zip. "
               f"Benchmark {benchmark_id}; seed {seed}; không bốc lại mẫu. "
               "Giải nén, đọc README.md và chạy python benchmark.py verify trước. "
               "Mỗi người khóa một phiên bản agent, chạy tự động theo thứ tự file, không dùng nhãn "
               "hay sửa đáp án tay. Mọi thiết kế agent/model/framework đều dùng được. "
               "Chỉ gửi lại submission.jsonl đúng định dạng cuộc thi và run_metadata.json sáu trường; không cần question log. "
               "Điểm chấm tập trung: PA=(đúng-sai)/" + str(size) + "; thiếu/abstain=0. "
               "Metadata chỉ ghi tên, model(s), GPU, thời gian khởi tạo, thời gian batch và unit nếu có. "
               "TB giây/câu lấy từ thời gian batch chia số câu khi nộp đủ; không đo p95. "
               "Khác GPU thì tốc độ chỉ tham khảo. Đây là benchmark nội bộ trên train; "
               "khai báo dữ liệu từng fine-tune/debug. Hướng dẫn chi tiết và mẫu file có trong ZIP.\n")
    (output / "TIN_NHAN_GUI_NHOM.txt").write_text(message, encoding="utf-8")
    (output / "SHA256SUMS.txt").write_text("".join(f"{p['sha256']}  {p['file']}\n" for p in packages), encoding="utf-8")
    release = {**common, "packages": packages, "selected_business_count": len(businesses),
               "verified_source_files": len(approved["files"]), "zip_crc_and_sha256_verified": True,
               "participant_has_gold_labels": False, "sample_reproduced":
               ids == [q["id"] for q in random.Random(seed).sample(sorted(questions, key=lambda q: q["id"]), size)]}
    save_json(output / "release_verification.json", release)
    return release


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=project / "train/train")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs/packages/team-benchmark-train200-2026-11-07-v2")
    parser.add_argument("--seed", type=int, help="Bỏ trống để sinh một seed ngẫu nhiên rồi khóa trong manifest")
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()
    result = prepare(project, args.split_root, args.output_dir, args.seed, args.size)
    print(json.dumps({key: result[key] for key in ("benchmark_id", "seed", "sample_size", "selected_business_count", "packages")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
