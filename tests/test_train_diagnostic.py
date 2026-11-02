import json
from collections import Counter
from pathlib import Path
import sys
import types
import zipfile

import pytest

from datamate.agent.runner import Agent
from datamate.config import Settings
from datamate.data.split import DatasetSplit
from datamate.domain import DataMateError, Question
from datamate.training.dataset import business_holdout, source_manifest
from tools.diagnose_train import diagnostic_flags, diagnostic_plan, disconnect_colab_after_report, question_profile, run_diagnostic


def setup_agent(split_root, tmp_path, backend):
    model = tmp_path / "model"
    model.mkdir()
    (model / "artifact_manifest.json").write_text('{"fixture":true}')
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sources": [{
        **source_manifest(split_root), "role": "competition_training"
    }]}), encoding="utf-8")
    settings = Settings(split_root=str(split_root), cache_dir=str(tmp_path / "cache"),
                        result_dir=str(tmp_path / "result"), model_dir=str(model))
    return Agent(settings, backend), sources


def test_plan_is_repeatable_and_keeps_holdout_separate(split_root):
    questions = DatasetSplit(split_root).questions()
    probe = (questions[0].business_id, questions[0].as_of_date, questions[0].question)
    kinds = {q.id: "number" for q in questions}
    first = diagnostic_plan(questions, kinds, probe=probe)
    second = diagnostic_plan(list(reversed(questions)), kinds, probe=probe)
    assert [q.id for q in first[0]] == [q.id for q in second[0]]
    assert set(first[1]).isdisjoint(first[2])
    assert {q.business_id for q in first[0]} <= set(first[1])
    assert first[0][0] == questions[0]


def test_diagnostic_scores_after_predictions_resumes_and_restores_backend(split_root, tmp_path, scripted_backend, monkeypatch):
    backend = scripted_backend([
        {"action": "sql", "sql": 'SELECT SUM(num_en("Tiền")) FROM raw_0_sales'},
        {"action": "final", "type": "number", "from_query": True},
        {"action": "sql", "sql": 'SELECT SUM(num_en("Tiền")) FROM raw_0_sales'},
        {"action": "final", "type": "number", "from_query": True},
    ])
    agent, sources = setup_agent(split_root, tmp_path, backend)
    from tools import diagnose_train
    original_reader = diagnose_train.read_jsonl
    state = {"read_answers": False}

    def tracked_reader(path):
        if Path(path).name == "answers.jsonl":
            state["read_answers"] = True
        return original_reader(path)

    generate = backend.generate

    def checked_generate(messages, remaining_seconds):
        assert not state["read_answers"], "Labels were deserialized before generation"
        assert "verified_result" not in str(messages)
        return generate(messages, remaining_seconds)

    monkeypatch.setattr(diagnose_train, "read_jsonl", tracked_reader)
    backend.generate = checked_generate
    output = tmp_path / "diagnostic"
    first, bundle = run_diagnostic(agent, output, sources, limit=1)
    assert first["questions"] == 1 and first["correct"] == 1
    assert first["completed_questions"] == 1 and first["remaining_questions"] == 1
    assert bundle.is_file() and agent.backend is backend
    state["read_answers"] = False
    second, _ = run_diagnostic(agent, output, sources, limit=2)
    assert second["questions"] == 2 and second["correct"] == 2
    assert len(backend.messages) == 4 and agent.backend is backend
    assert second["total_model_calls"] == 4
    record = json.loads(next((output / "logs").glob("*.json")).read_text(encoding="utf-8"))
    assert len(record["generation_calls"]) == 2
    agent.settings.max_steps += 1
    with pytest.raises(DataMateError, match="khác model/config/code/data/sample"):
        run_diagnostic(agent, output, sources)


def test_interrupt_saves_partial_report_and_restores_backend(split_root, tmp_path):
    class Interrupted:
        def generate(self, messages, remaining_seconds):
            raise KeyboardInterrupt()

    backend = Interrupted()
    agent, sources = setup_agent(split_root, tmp_path, backend)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_diagnostic(agent, output, sources, limit=1)
    assert agent.backend is backend
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_questions"] == 0
    assert (output / "diagnostic_bundle.zip").is_file()


def test_observable_flags_do_not_claim_semantic_sql_validation():
    row = {"submission": {"type": "unanswerable"}, "trace": [
        {"raw": "thought ...", "observation": {"error": "bad JSON"}},
        {"action": {"action": "sql"}, "observation": {"error": "function arity"}},
    ]}
    assert diagnostic_flags(row, {"type": "number"}) == [
        "action_parse_error", "false_unanswerable", "sql_error", "type_mismatch"]


def test_120_sample_balances_types_and_covers_all_development_businesses():
    kinds = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]
    questions, references = [], {}
    for b in range(30):
        for i, kind in enumerate(kinds):
            q = Question(f"q_{b}_{i}", f"biz_{b:03}", f"question {i}", "2026-06-15")
            questions.append(q)
            references[q.id] = kind
    development, holdout = business_holdout(questions, .2, 42)
    known = next(q for q in questions if q.business_id in development)
    plan, dev, reserved, known_id = diagnostic_plan(
        questions, references, probe=(known.business_id, known.as_of_date, known.question))
    assert len(plan) == len({q.id for q in plan}) == 120
    assert {q.business_id for q in plan} == set(dev) == development
    assert set(reserved) == holdout
    assert Counter(references[q.id] for q in plan) == {kind: 15 for kind in kinds}
    assert plan[0].id == known_id == known.id
    repeated = diagnostic_plan(list(reversed(questions)), references,
                               probe=(known.business_id, known.as_of_date, known.question))[0]
    assert [q.id for q in plan] == [q.id for q in repeated]


@pytest.mark.parametrize("flush_fails", [False, True])
def test_optional_disconnect_flushes_drive_before_unassign(tmp_path, monkeypatch, flush_fails):
    bundle = tmp_path / "report.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("summary.json", "{}")
    calls = []

    def flush(**kwargs):
        calls.append("flush")
        assert kwargs["timeout_ms"] == 60000
        if flush_fails:
            raise ValueError("fixture flush failure")

    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(flush_and_unmount=flush)
    colab.runtime = types.SimpleNamespace(unassign=lambda: calls.append("unassign"))
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setenv("TBE_RUNTIME_ADDR", "fixture-only")
    disconnect_colab_after_report(bundle)
    assert calls == (["flush"] if flush_fails else ["flush", "unassign"])


def test_entire_notebook_cell_executes_once_and_exports_all_records(split_root, tmp_path, scripted_backend):
    backend = scripted_backend([
        {"action": "sql", "sql": 'SELECT SUM(num_en("Tiền")) FROM raw_0_sales'},
        {"action": "final", "type": "number", "from_query": True},
        {"action": "sql", "sql": 'SELECT SUM(num_en("Tiền")) FROM raw_0_sales'},
        {"action": "final", "type": "number", "from_query": True},
    ])
    agent, sources = setup_agent(split_root, tmp_path, backend)
    work = tmp_path / "work"
    (work / "configs").mkdir(parents=True)
    (work / "configs/training_sources.json").write_bytes(sources.read_bytes())
    path = Path(__file__).resolve().parents[1] / "tools/diagnose_train.py"
    namespace = {"__name__": "__main__", "agent": agent, "WORK": work, "SAVE": tmp_path / "saved"}
    # Fixture has one question per business; only the public configuration is
    # changed for this tiny end-to-end execution, with real Agent/SQL/scoring.
    source = path.read_text(encoding="utf-8").replace('SAMPLE_SIZE = 72', 'SAMPLE_SIZE = 2').replace(
        'SAMPLING_MODE = "language_stress_3"', 'SAMPLING_MODE = "balanced"')
    exec(compile(source, str(path), "exec"), namespace)
    assert agent.backend is backend
    assert namespace["diagnostic_summary"]["status"] == "complete"
    assert namespace["diagnostic_summary"]["correct"] == 2
    with zipfile.ZipFile(namespace["diagnostic_bundle"]) as archive:
        assert archive.testzip() is None
        assert {"summary.json", "review.csv", "review.jsonl", "submission.jsonl", "runtime.json"} <= set(archive.namelist())
        assert len(archive.read("submission.jsonl").splitlines()) == 2


def test_language_profiles_and_72_plan_require_three_per_business():
    assert question_profile("cho toi biet dthu hnay la bn")["abbreviated"]
    assert question_profile("cho toi biet dthu hnay la bn")["no_diacritics"]
    assert not question_profile("Cho tôi biết doanh thu hôm nay.")["no_diacritics"]
    assert question_profile("So với cùng kỳ tháng trước, doanh thu tăng bao nhiêu phần trăm?")["complex_hint"]
    kinds = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]
    texts = ["tong dthu hnay la bn", "So với cùng kỳ, doanh thu tăng hay giảm bao nhiêu phần trăm?",
             "Doanh thu hôm nay là bao nhiêu?", "ngay nao dthu cao nhat trong thang nay",
             "Số đơn hqua là bn?", "Giúp liệt kê các khách còn nợ trên 30 ngày.",
             "Lãi gộp tháng này là bao nhiêu?", "Cho biết tuổi nợ trung bình của từng khách."]
    questions, references = [], {}
    for b in range(30):
        for i, text in enumerate(texts):
            q = Question(f"q_{b}_{i}", f"biz_{b:03}", text, "2026-06-15")
            questions.append(q)
            references[q.id] = kinds[(b + i) % len(kinds)]
    plan, development, holdout, _ = diagnostic_plan(questions, references, sample_size=72,
                                                  sampling_mode="language_stress_3")
    assert len(plan) == len({q.id for q in plan}) == 72
    assert Counter(q.business_id for q in plan) == {b: 3 for b in development}
    assert not {q.business_id for q in plan} & set(holdout)
    assert len({references[q.id] for q in plan}) == 8
    for business in development:
        profiles = [question_profile(q.question) for q in plan if q.business_id == business]
        assert all(any(p[kind] for p in profiles) for kind in ("no_diacritics", "abbreviated", "complex_hint"))
    again = diagnostic_plan(list(reversed(questions)), references, sample_size=72,
                            sampling_mode="language_stress_3")[0]
    assert [q.id for q in plan] == [q.id for q in again]
