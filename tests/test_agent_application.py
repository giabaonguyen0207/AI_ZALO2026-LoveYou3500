import json

from datamate.agent.runner import Agent
from datamate.application import run_batch
from datamate.config import Settings
from datamate.data.split import DatasetSplit
from datamate.domain import Question


def settings(split_root, tmp_path, **overrides):
    return Settings(split_root=str(split_root), cache_dir=str(tmp_path / "cache"), result_dir=str(tmp_path / "result"), backend="demo", **overrides)


def test_agent_repairs_sql_and_projects_answer_without_model_arithmetic(split_root, tmp_path, scripted_backend):
    backend = scripted_backend([
        {"action": "sql", "sql": "SELECT no_such_column FROM calendar"},
        {"action": "sql", "sql": 'SELECT sum(num_en("Tiền")) FROM raw_0_sales'},
        {"action": "final", "type": "number", "from_query": True}])
    agent = Agent(settings(split_root, tmp_path), backend)
    result = agent.run(Question("id1", "shop_a", "Tổng tiền?", "2026-06-02"))
    assert result.answer.answer == 4900
    assert result.model_calls == 3 and result.trace[0]["observation"].get("error")
    assert result.trace[-1]["accepted"]
    assert "id1" not in backend.messages[0][1]["content"]


def test_false_numeric_final_is_rejected_and_missing_evidence_abstains(split_root, tmp_path, scripted_backend):
    backend = scripted_backend([{"action":"sql","sql":"SELECT 12"},
                               {"action":"final","type":"number","answer":999},
                               {"action":"final","type":"abstain"}])
    result = Agent(settings(split_root, tmp_path), backend).run(Question("id1", "shop_a", "q", "2026-06-02"))
    assert result.answer.type == "abstain"
    assert "khác query" in result.trace[1]["observation"]["error"]
    direct = scripted_backend([{"action":"final","type":"unanswerable"}, {"action":"final","type":"abstain"}])
    assert Agent(settings(split_root,tmp_path),direct).run(Question("id2","shop_a","q","2026-06-02")).answer.type == "abstain"


def test_self_review_uses_same_backend_and_requires_new_final(split_root, tmp_path, scripted_backend):
    backend = scripted_backend([{"action":"sql","sql":"SELECT 12"},
                               {"action":"final","type":"number","from_query":True},
                               {"action":"sql","sql":"SELECT 6+6"},
                               {"action":"final","type":"number","from_query":True}])
    result=Agent(settings(split_root,tmp_path,mode="self_review"),backend).run(Question("q","shop_a","q","2026-06-02"))
    assert result.answer.answer == 12 and result.model_calls == 4


def test_batch_flushes_valid_jsonl_logs_and_safe_resume(split_root, tmp_path):
    config = settings(split_root, tmp_path)
    result = run_batch(config)
    output = tmp_path / "result"
    rows = [json.loads(line) for line in (output / "submission.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["written"] == 3 and len(rows) == 3 and all(row["type"] == "abstain" for row in rows)
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in (output / "logs").glob("*.json")]
    assert len(logs) == 3 and all("elapsed_seconds" in log and "submission" in log and "id" in log for log in logs)
    assert all(log["model_calls"] == 0 for log in logs)
    assert run_batch(config, resume=True)["written"] == 0
