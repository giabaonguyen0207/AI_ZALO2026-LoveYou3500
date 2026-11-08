from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path

from ..agent.prompts import initial_messages
from ..data.ingest import Ingestor
from ..data.split import DatasetSplit, ensure_outside_dataset, read_jsonl
from ..domain import Answer, DataMateError, json_text
from ..evaluation.scoring import matches
from ..models.artifacts import sha256


def source_manifest(root: str | Path) -> dict:
    root = Path(root).resolve()
    required = ["questions.jsonl", "answers.jsonl", "rulebook.md", "calendar.csv"]
    paths = [root / name for name in required]
    paths += sorted(path for path in (root / "data").rglob("*") if path.is_file())
    if not all(path.is_file() for path in paths):
        raise DataMateError("Training cần đầy đủ train questions/answers/rulebook/calendar/data")
    files = {path.relative_to(root).as_posix(): sha256(path) for path in paths}
    digest = hashlib.sha256(json_text(files, sort_keys=True).encode()).hexdigest()
    return {"fingerprint": digest, "files": files}


def require_approved_source(root: str | Path, sources_path: str | Path) -> dict:
    manifest = source_manifest(root)
    allowed = json.loads(Path(sources_path).read_text(encoding="utf-8"))
    entry = next((row for row in allowed["sources"] if row["fingerprint"] == manifest["fingerprint"]
                  and row["role"] == "competition_training"), None)
    if not entry:
        raise DataMateError("Nguồn chưa được đăng ký là TRAIN. Không chuẩn bị training từ public/private test hoặc dữ liệu bị sửa.")
    return entry


def business_holdout(questions, fraction: float, seed: int):
    if not 0 < fraction < 1:
        raise DataMateError("validation_fraction phải trong (0,1)")
    businesses = sorted({q.business_id for q in questions})
    if len(businesses) < 2:
        raise DataMateError("Cần tối thiểu 2 doanh nghiệp để tách holdout")
    random.Random(seed).shuffle(businesses)
    count = max(1, min(len(businesses)-1, round(len(businesses)*fraction)))
    return set(businesses[count:]), set(businesses[:count])


def prepare_training(split_root, output, kind="format", logs=None, validation_fraction=0.2, seed=42,
                     sources="configs/training_sources.json"):
    split = DatasetSplit(split_root)
    approved = require_approved_source(split.root, sources)
    destination = ensure_outside_dataset(output, split.root)
    if destination.exists() and any(destination.iterdir()):
        raise DataMateError("Thư mục training output phải rỗng để không trộn các lượt chuẩn bị")
    questions = split.questions()
    references = {row["id"]: row for row in read_jsonl(split.root / "answers.jsonl")}
    if set(references) != {q.id for q in questions}:
        raise DataMateError("Train questions và answers không khớp ID")
    train_businesses, validation_businesses = business_holdout(questions, validation_fraction, seed)
    question_map = {q.id: q for q in questions}
    examples, rejected = [], 0
    if kind == "format":
        # A transparent format-only warm-up, not fabricated reasoning or SQL.
        for question in questions:
            target = Answer.from_dict(references[question.id]).to_dict()
            messages = [{"role": "system", "content": "Chuẩn hóa kết quả đã được cung cấp thành JSON đáp án DataMate. Chỉ xuất type và answer, hoặc type unanswerable. Không giải câu hỏi, không thêm thông tin."},
                        {"role": "user", "content": json_text({"verified_result": target})},
                        {"role": "assistant", "content": json_text(target)}]
            examples.append({"question_id": question.id, "business_id": question.business_id, "messages": messages, "kind": "format_only"})
    elif kind == "trajectories":
        if not logs:
            raise DataMateError("Trajectories cần --logs từ baseline chạy trên TRAIN, không dùng đáp án để sinh tool calls")
        ingestor = Ingestor(split, destination.parent / "catalog_cache")
        seen = set()
        for path in sorted(Path(logs).glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            question = question_map.get(record.get("id"))
            if question is None or record.get("business_id") != question.business_id or record.get("rulebook_sha256") != split.rulebook_hash:
                raise DataMateError("Log không thuộc train source đã đăng ký")
            if question.id in seen:
                raise DataMateError("Trùng log câu hỏi")
            seen.add(question.id)
            if not matches(record.get("submission", {}), references[question.id]):
                rejected += 1
                continue
            trace = record.get("trace", [])
            if not trace or not trace[-1].get("accepted"):
                rejected += 1
                continue
            answer_steps=[step for step in trace if step.get('phase')!='preparation']
            if any(step.get('error') or step.get('observation',{}).get('error') for step in answer_steps):
                rejected += 1
                continue  # A correct final answer does not make failed actions good SFT targets.
            from ..agent.protocol import parse_action
            try:
                parsed_steps=[parse_action(step['raw']) for step in answer_steps]
            except (DataMateError,KeyError,TypeError):
                rejected += 1
                continue
            if record.get('initial_messages'):
                messages=deepcopy(record['initial_messages'])
            elif any(step.get('phase')=='preparation' for step in trace):
                raise DataMateError('Trajectory thích nghi thiếu initial_messages thực tế; không dựng lại bằng catalog khác')
            else:
                catalog = ingestor.catalog(question.business_id).summary()
                members = {name for table in catalog["tables"] for name in table.get("view_of", [])}
                catalog["tables"] = [table for table in catalog["tables"] if table["name"] not in members]
                messages = initial_messages(question, split.rulebook, split.readme(question.business_id), catalog)
            for step,parsed in zip(answer_steps,parsed_steps):
                messages.append({"role": "assistant", "content": json_text(parsed)})
                if 'model_observation' in step or 'observation' in step:
                    messages.append({"role": "user", "content": json_text({"tool_result": step.get('model_observation',step.get('observation'))})})
            examples.append({"question_id": question.id, "business_id": question.business_id, "messages": messages, "kind": "verified_trajectory"})
        if not examples:
            raise DataMateError("Chưa có trajectory train đúng để fine-tune. Chạy baseline/kiểm tra lỗi trước; không tạo lời giải giả.")
    else:
        raise DataMateError("kind không hợp lệ")
    train = [row for row in examples if row["business_id"] in train_businesses]
    validation = [row for row in examples if row["business_id"] in validation_businesses]
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        path = destination / f"{name}.jsonl"
        path.write_text("".join(json_text(row) + "\n" for row in rows), encoding="utf-8")
        ids = [q.id for q in questions if q.business_id in (train_businesses if name == "train" else validation_businesses)]
        (destination / f"{name}_ids.json").write_text(json_text(ids), encoding="utf-8")
    manifest = {"kind": kind, "source_fingerprint": approved["fingerprint"], "source_role": approved["role"],
                "split_strategy": "business_holdout", "seed": seed, "train_businesses": sorted(train_businesses),
                "validation_businesses": sorted(validation_businesses), "train_examples": len(train),
                "validation_examples": len(validation), "rejected_trajectories": rejected,
                "files": {name: sha256(destination / name) for name in ("train.jsonl", "validation.jsonl", "train_ids.json", "validation_ids.json")}}
    (destination / "manifest.json").write_text(json_text(manifest, indent=2), encoding="utf-8")
    return manifest
