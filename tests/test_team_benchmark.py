"""Release isolation and scoring checks; these runs use synthetic answers only."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
from prepare_team_benchmark import prepare, write_jsonl
from team_benchmark import compare, read_json, read_jsonl, save_json, scorer, sha256, verify


@pytest.fixture
def release(tmp_path):
    source = tmp_path / "train"
    data = source / "data/biz_1"
    data.mkdir(parents=True)
    (data / "README.md").write_text("Dữ liệu kiểm thử tổng hợp", encoding="utf-8")
    (data / "full.csv").write_text("amount\n1\n2\n3\n4\n5\n", encoding="utf-8")
    (source / "rulebook.md").write_text("Quy tắc tổng hợp", encoding="utf-8")
    (source / "calendar.csv").write_text("date\n2026-11-07\n", encoding="utf-8")
    write_jsonl(source / "questions.jsonl", [{"id": f"q_{i:04d}", "business_id": "biz_1",
                 "question": f"Câu tổng hợp {i}", "as_of_date": "2026-11-07"} for i in range(300)])
    write_jsonl(source / "answers.jsonl", [{"id": f"q_{i:04d}", "type": "number", "answer": i,
                 "alias": []} for i in range(300)])
    files = {p.relative_to(source).as_posix(): sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    fingerprint = hashlib.sha256(json.dumps(files, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()).hexdigest()
    sources = tmp_path / "sources.json"
    save_json(sources, {"sources": [{"role": "competition_training", "files": files, "fingerprint": fingerprint}]})
    output = tmp_path / "release"
    manifest = prepare(PROJECT, source, output, seed=736271, size=200, sources=sources)
    return source, output, manifest


def test_release_keeps_labels_private_full_data_and_frozen_sample(release, tmp_path):
    source, output, manifest = release
    participant, coordinator = output / "participant", output / "coordinator"
    _, questions = verify(participant)
    _, private_questions = verify(coordinator)
    assert questions == private_questions and len(questions) == 200
    assert len({q["id"] for q in questions}) == 200
    assert all(set(q) == {"id", "business_id", "question", "as_of_date"} for q in questions)
    assert (participant / "dataset/data/biz_1/full.csv").read_bytes() == (source / "data/biz_1/full.csv").read_bytes()
    with zipfile.ZipFile(output / "participants.zip") as archive:
        assert not any(Path(name).name == "answers.jsonl" for name in archive.namelist())
        assert not any(Path(name).name == "question_log.template.json" for name in archive.namelist())
    metadata = read_json(participant / "templates/run_metadata.template.json")
    assert set(metadata) == {"participant", "models", "gpu", "load_seconds", "inference_seconds", "compute_units_used"}
    examples = read_jsonl(participant / "templates/submission.template.jsonl")
    Answer, _, _, _ = scorer()
    assert {r["type"] for r in examples} == {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}
    assert not {r["id"] for r in examples} & {q["id"] for q in questions}
    for example in examples:
        Answer.from_dict(example)
        if example["type"] not in {"abstain", "unanswerable"}:
            assert "answer" in example
    instructions = (participant / "README.md").read_text(encoding="utf-8")
    assert "qwen" not in instructions.lower() and "gemma" not in instructions.lower()
    labels = read_jsonl(coordinator / "answers.jsonl")
    assert [r["id"] for r in labels] == [q["id"] for q in questions]
    assert manifest["sample_reproduced"] and not manifest["inference_performed"]
    with pytest.raises(ValueError, match="Output đã tồn tại"):
        prepare(PROJECT, source, output, seed=1, sources=tmp_path / "sources.json")
    (participant / "dataset/questions.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="khác hash"):
        verify(participant)


def test_partial_and_invalid_submissions_keep_full_denominator_and_pairwise(release, tmp_path):
    _, output, manifest = release
    coordinator = output / "coordinator"
    references = read_jsonl(coordinator / "answers.jsonl")
    partial, perfect = tmp_path / "partial", tmp_path / "perfect"
    partial.mkdir()
    perfect.mkdir()
    write_jsonl(partial / "submission.jsonl", [references[0],
               {"id": references[1]["id"], "type": "unanswerable"},
               {"id": references[2]["id"], "type": "abstain", "answer": 123}])
    write_jsonl(perfect / "submission.jsonl", references)
    logs = partial / "logs"
    logs.mkdir()
    save_json(logs / "first.json", {"id": references[0]["id"], "submission": references[0],
                                   "elapsed_seconds": 1.25, "model_calls": 2})
    report = compare(coordinator, [("partial", partial), ("perfect", perfect)], tmp_path / "comparison")
    first, second = report["leaderboard"]
    assert first["participant"] == "perfect" and first["penalized_accuracy"] == 1
    assert (second["correct"], second["wrong"], second["abstain"], second["missing"], second["invalid"]) == (1, 2, 197, 197, 1)
    assert second["penalized_accuracy"] == -.005
    assert second["mean_seconds"] == 1.25 and second["end_to_end_seconds"] is None
    assert report["pairwise"][0]["b_higher_question_score"] == 199
    assert (tmp_path / "comparison/comparison.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert not report["inference_performed_by_this_command"]
    write_jsonl(partial / "submission.jsonl", [references[0], references[0]])
    with pytest.raises(ValueError, match="trùng ID"):
        compare(coordinator, [("partial", partial)], tmp_path / "bad-duplicate")
    write_jsonl(partial / "submission.jsonl", [{"id": "outside", "type": "abstain"}])
    with pytest.raises(ValueError, match="ngoài benchmark"):
        compare(coordinator, [("partial", partial)], tmp_path / "bad-extra")
    write_jsonl(partial / "submission.jsonl", [references[0]])
    save_json(partial / "run_metadata.json", {"benchmark_id": "different", "questions_sha256": manifest["questions_sha256"]})
    with pytest.raises(ValueError, match="không khớp benchmark"):
        compare(coordinator, [("partial", partial)], tmp_path / "bad-identity")


def test_coordinator_runs_as_standalone_without_project_import_path(release, tmp_path):
    _, output, _ = release
    coordinator = output / "coordinator"
    run = tmp_path / "synthetic-run"
    run.mkdir()
    write_jsonl(run / "submission.jsonl", [])
    completed = subprocess.run([sys.executable, "-I", str(coordinator / "benchmark.py"), "compare",
                               "--run", f"synthetic={run}", "--output", str(tmp_path / "standalone-report")],
                              cwd=tmp_path, text=True, capture_output=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    report = read_json(tmp_path / "standalone-report/leaderboard.json")
    assert report["leaderboard"][0]["abstain"] == 200
    assert report["leaderboard"][0]["penalized_accuracy"] == 0


def test_six_metadata_fields_without_question_logs_are_enough(release, tmp_path):
    _, output, _ = release
    coordinator = output / "coordinator"
    questions = read_jsonl(coordinator / "questions.jsonl")
    run = tmp_path / "any-agent"
    run.mkdir()
    write_jsonl(run / "submission.jsonl", [{"id": q["id"], "type": "abstain"} for q in questions])
    save_json(run / "run_metadata.json", {"participant": "Any agent", "models": ["Any eligible checkpoint"],
              "gpu": "GPU chosen by participant", "load_seconds": 120, "inference_seconds": 600,
              "compute_units_used": 2.5})
    report = compare(coordinator, [("any_agent", run)], tmp_path / "minimal-report")
    summary = report["leaderboard"][0]
    assert summary["end_to_end_seconds"] == 720 and summary["mean_seconds"] == 3
    assert summary["mean_seconds_source"] == "batch_wall_time_divided_by_question_count"
    assert summary["p50_seconds"] is None and summary["p95_seconds"] is None
    assert summary["compute_units_used"] == 2.5 and not summary["warnings"]
    write_jsonl(run / "submission.jsonl", [{"id": q["id"], "type": "abstain"} for q in questions[:100]])
    partial = compare(coordinator, [("any_agent", run)], tmp_path / "minimal-partial-report")
    assert partial["leaderboard"][0]["mean_seconds"] is None
