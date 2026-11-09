"""Synthetic runner checks: no model weights or real GPU inference are used."""
from __future__ import annotations

import csv
import ast
import hashlib
import json
import types
import uuid
import zipfile
from pathlib import Path

import pytest

from datamate.config import Settings
from datamate.data.split import DatasetSplit
from datamate.domain import Answer, DataMateError, RunResult
from tools import infer_team200 as runner
from tools.create_team200_colab import BATCH, INSTALL, LOAD, SETUP


@pytest.fixture
def team_case(tmp_path, monkeypatch):
    root = tmp_path / "participant"
    data = root / "dataset"
    business = data / "data/fixture"
    business.mkdir(parents=True)
    (business / "README.md").write_text("Synthetic fixture", encoding="utf-8")
    (business / "sales.csv").write_text("amount\n1\n", encoding="utf-8")
    (data / "rulebook.md").write_text("Synthetic rulebook", encoding="utf-8")
    (data / "calendar.csv").write_text("date\n2026-11-01\n", encoding="utf-8")
    questions = [{"id": f"q_{i:03}", "business_id": "fixture", "question": "Synthetic question", "as_of_date": "2026-11-01"} for i in range(200)]
    runner.save_jsonl(data / "questions.jsonl", questions)
    qhash = runner.sha256(data / "questions.jsonl")
    monkeypatch.setattr(runner, "QUESTIONS_SHA256", qhash)
    runner.save_json(root / "question_ids.json", [q["id"] for q in questions])
    files = {p.relative_to(root).as_posix(): runner.sha256(p) for p in root.rglob("*") if p.is_file()}
    runner.save_json(root / "benchmark_manifest.json", {"benchmark_id": runner.BENCHMARK_ID, "protocol_version": 2,
                     "sample_size": 200, "questions_sha256": qhash, "files": files})
    labels = [{"id": q["id"], "type": "number", "answer": 1} for q in questions]
    content = "".join(json.dumps(row) + "\n" for row in labels).encode()
    refzip = tmp_path / "references.zip"
    with zipfile.ZipFile(refzip, "w") as archive:
        archive.writestr("coordinator/answers.jsonl", content)
        archive.writestr("coordinator/benchmark_manifest.json", json.dumps({"benchmark_id": runner.BENCHMARK_ID,
                         "questions_sha256": qhash, "files": {"answers.jsonl": hashlib.sha256(content).hexdigest()}}))
    monkeypatch.setattr(runner, "REFERENCE_ZIP_SHA256", runner.sha256(refzip))
    model = tmp_path / "model"
    model.mkdir()
    runner.save_json(model / "artifact_manifest.json", {"source": {"repo_id": "synthetic/model", "revision": "fixture"}})
    original = types.SimpleNamespace(counts_as_model=True)

    class FakeAgent:
        pipeline_version = 6
        backend = original
        split = DatasetSplit(data)
        settings = Settings(split_root=str(data), model_dir=str(model), cache_dir=str(tmp_path / "cache"))
        calls = 0
        fail_on = None
        in_generation = False

        def run(self, question, remaining_seconds):
            self.in_generation = True
            self.calls += 1
            try:
                if self.calls == self.fail_on:
                    raise KeyboardInterrupt()
                return RunResult(Answer("number", 1), self.calls * .25, [{"raw": "synthetic raw", "action": {"action": "final"}}], 1,
                                 [{"role": "user", "content": question.question}])
            finally:
                self.in_generation = False

    agent = FakeAgent()
    original_loader = runner.load_references

    def checked_load(*args):
        assert not agent.in_generation and agent.calls > 0
        return original_loader(*args)

    monkeypatch.setattr(runner, "load_references", checked_load)
    kwargs = {"participant": "synthetic", "load_seconds": 12, "load_session_id": "load-1", "source_archive_sha256": "fixture-source"}
    return agent, root, refzip, tmp_path / "output", kwargs, original


def test_resume_exports_two_file_submission_and_full_private_review(team_case):
    agent, root, refs, output, kwargs, original = team_case
    summary, group, private = runner.run_team200(agent, root, output, refs, limit=2, **kwargs)
    assert summary["completed_questions"] == 2 and summary["penalized_accuracy"] == .01
    assert summary["questions"] == 200 and summary["abstain"] == 198
    assert summary["median_seconds"] == .375
    with zipfile.ZipFile(group) as archive:
        assert set(archive.namelist()) == {"submission.jsonl", "run_metadata.json"}
        metadata = json.loads(archive.read("run_metadata.json"))
        assert len(metadata) == 6 and metadata["load_seconds"] == 12
        assert metadata["compute_units_used"] is None
    with zipfile.ZipFile(private) as archive:
        assert {"review.jsonl", "review.csv", "review.md", "logs/q_000.json", "logs/q_000.md", "runtime.json"} <= set(archive.namelist())
    assert len(list(csv.DictReader((output / "review.csv").open(encoding="utf-8-sig")))) == 200
    second, _, _ = runner.run_team200(agent, root, output, refs, limit=3, **kwargs)
    assert second["completed_questions"] == 3 and agent.calls == 3
    assert second["load_seconds"] == 12 and agent.backend is original
    assert len(runner.read_jsonl(output / "submission.jsonl")) == 3
    agent.settings.max_steps += 1
    with pytest.raises(DataMateError, match="Không resume"):
        runner.run_team200(agent, root, output, refs, **kwargs)


def test_interrupt_keeps_partial_report_and_does_not_overwrite_edited_answers(team_case):
    agent, root, refs, output, kwargs, original = team_case
    agent.fail_on = 2
    with pytest.raises(KeyboardInterrupt):
        runner.run_team200(agent, root, output, refs, **kwargs)
    assert agent.backend is original
    assert (output / "diagnostic_bundle.zip").is_file() and (output / "report_receipt.json").is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_questions"] == 1 and summary["penalized_accuracy"] == .005
    runner.save_jsonl(output / "submission.jsonl", [{"id": "q_000", "type": "number", "answer": 999}])
    with pytest.raises(DataMateError, match="Submission khác log"):
        runner.run_team200(agent, root, output, refs, **kwargs)


def test_new_model_load_is_counted_and_hard_loss_does_not_invent_batch_time(team_case):
    agent, root, refs, output, kwargs, _ = team_case
    runner.run_team200(agent, root, output, refs, limit=1, **kwargs)
    ledger = json.loads((output / "session_history.json").read_text())
    ledger["batches"][0]["closed"] = False
    runner.save_json(output / "session_history.json", ledger)
    summary, _, _ = runner.run_team200(agent, root, output, refs, limit=2, **{**kwargs, "load_session_id": "load-2"})
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert summary["timing_incomplete_after_hard_runtime_loss"]
    assert metadata["load_seconds"] == 24 and metadata["inference_seconds"] is None


def test_auto_disconnect_requires_both_verified_bundles(team_case, monkeypatch):
    agent, root, refs, output, kwargs, _ = team_case
    runner.run_team200(agent, root, output, refs, limit=1, **kwargs)
    calls = []
    monkeypatch.setattr(runner, "disconnect_colab_after_report", lambda path: calls.append(path))
    runner.finish_and_disconnect(output)
    assert calls == [output / "diagnostic_bundle.zip"]
    (output / "group_submission.zip").write_bytes(b"broken")
    with pytest.raises(DataMateError, match="Chưa xác minh"):
        runner.finish_and_disconnect(output)
    assert len(calls) == 1


def test_missing_gold_keeps_submission_and_ungraded_review(team_case):
    agent, root, refs, output, kwargs, _ = team_case
    summary, group, private = runner.run_team200(agent, root, output, refs.with_name("missing.zip"), limit=1, **kwargs)
    assert summary["review_error"] and "penalized_accuracy" not in summary
    assert group.is_file() and private.is_file() and len(runner.read_jsonl(output / "review.jsonl")) == 200


def test_notebook_batch_cell_disconnects_only_for_its_own_new_report(team_case, monkeypatch):
    agent, root, refs, output, kwargs, _ = team_case
    calls = []
    monkeypatch.setattr(runner, "disconnect_colab_after_report", lambda path: calls.append(path))
    module = {"run_team200": runner.run_team200, "finish_and_disconnect": runner.finish_and_disconnect}
    namespace = {"runpy": types.SimpleNamespace(run_path=lambda _: module), "WORK": output.parent,
                 "agent": agent, "PARTICIPANT_ROOT": root, "SAVE": output, "REFERENCE_ZIP": refs,
                 "PARTICIPANT": kwargs["participant"], "LOAD_SECONDS": kwargs["load_seconds"],
                 "LOAD_SESSION_ID": kwargs["load_session_id"], "SOURCE_SHA256": kwargs["source_archive_sha256"],
                 "RUN_LIMIT": 1, "MAX_RUN_MINUTES": 180, "COMPUTE_UNITS_USED": None,
                 "AUTO_DISCONNECT": True, "uuid": uuid, "json": json}
    exec(compile(BATCH, "notebook-batch", "exec"), namespace)
    assert len(calls) == 1
    agent.settings.max_steps += 1
    with pytest.raises(DataMateError, match="Không resume"):
        exec(compile(BATCH, "notebook-batch", "exec"), namespace)
    assert len(calls) == 1, "A stale prior ZIP must not trigger shutdown after a failed new invocation"


def test_notebook_cells_compile_and_torch_constraint_ends_in_real_newline():
    for source in (SETUP.replace("__EXPECTED_ARCHIVES__", "{}"), INSTALL, LOAD, BATCH):
        compile(source, "notebook-cell", "exec")
    tree = ast.parse(INSTALL)
    write = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text")
    assert write.args[0].right.value == "\n", "pip constraints must not contain literal backslash-n"
    assert "team200-layout-Nov9th-01" in SETUP


def test_runner_rejects_old_pipeline_before_loading_benchmark_or_generating(team_case):
    agent, root, refs, output, kwargs, _ = team_case
    agent.pipeline_version = 5
    with pytest.raises(DataMateError, match="pipeline >=6"):
        runner.run_team200(agent, root, output, refs, **kwargs)
    assert agent.calls == 0 and not output.exists()
