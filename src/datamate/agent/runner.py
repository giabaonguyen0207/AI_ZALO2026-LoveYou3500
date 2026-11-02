from __future__ import annotations

import json
import re
import time

from ..config import Settings
from ..data.ingest import Ingestor, quote
from ..data.split import DatasetSplit
from ..data.sql import SQLExecutor
from ..domain import Answer, DataMateError, Question, RunResult, json_text
from ..models.backends import ChatBackend
from .prompts import initial_messages


def parse_action(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise DataMateError("Model phải trả đúng một JSON object hợp lệ") from exc
    if not isinstance(value, dict) or value.get("action") not in {"schema", "sql", "final"}:
        raise DataMateError("action phải là schema/sql/final")
    return value


def answer_from_query(answer_type: str, result: dict | None) -> Answer:
    if not result or result.get("truncated"):
        raise DataMateError("Cần query cuối đầy đủ, không bị cắt")
    rows, columns = result["rows"], result["columns"]
    if answer_type == "table":
        value = {"columns": columns, "rows": rows}
    elif answer_type == "list":
        if len(columns) == 1:
            value = [row[0] for row in rows]
        elif len(rows) == 1:
            value = rows[0]
        else:
            raise DataMateError("List cần query 1 cột hoặc 1 dòng")
    else:
        if len(rows) != 1 or len(columns) != 1:
            raise DataMateError("Đáp án đơn cần query đúng 1 dòng 1 cột")
        value = rows[0][0]
        if answer_type == "boolean" and type(value) in (int, bool) and value in (0, 1):
            value = bool(value)
    return Answer(answer_type, value)


class Agent:
    def __init__(self, settings: Settings, backend: ChatBackend, split: DatasetSplit | None = None):
        self.settings, self.backend = settings, backend
        self.split = split or DatasetSplit(settings.split_root)
        self.ingestor = Ingestor(self.split, settings.cache_dir)

    def run(self, question: Question, remaining_seconds: float | None = None) -> RunResult:
        started = time.monotonic()
        budget = min(self.settings.question_timeout_seconds, remaining_seconds) if remaining_seconds is not None else self.settings.question_timeout_seconds
        deadline = started + budget
        trace, calls, successful_queries, reviewed = [], 0, 0, False
        last_query = None
        answer = Answer("abstain", reason="Hết ngân sách bước hoặc thời gian")
        try:
            catalog = self.ingestor.catalog(question.business_id)
            # Avoid listing every monthly sheet's identical columns in full.
            summary = catalog.summary()
            union_members = {name for table in summary["tables"] for name in table.get("view_of", [])}
            summary["tables"] = [table for table in summary["tables"] if table["name"] not in union_members]
            messages = initial_messages(question, self.split.rulebook, self.split.readme(question.business_id), summary)
            executor = SQLExecutor(catalog.database, self.settings.sql_timeout_seconds, self.settings.max_result_rows)
            for step in range(self.settings.max_steps):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                raw = self.backend.generate(messages, remaining)
                calls += int(getattr(self.backend, "counts_as_model", True))
                messages.append({"role": "assistant", "content": raw})
                event = {"step": step + 1, "raw": raw}
                try:
                    action = parse_action(raw)
                    event["action"] = action
                    if action["action"] == "final":
                        if action.get("type") in {"abstain", "unanswerable"}:
                            candidate = Answer.from_dict(action)
                        else:
                            candidate = answer_from_query(action.get("type"), last_query)
                            if "answer" in action and action["answer"] != candidate.answer:
                                raise DataMateError("answer khác query cuối; tính/định dạng lại bằng SQL")
                        if candidate.type != "abstain" and successful_queries == 0:
                            raise DataMateError("Cần thực thi ít nhất một query thành công trước khi kết luận")
                        if self.settings.mode == "self_review" and not reviewed and candidate.type != "abstain":
                            reviewed = True
                            event["review_requested"] = True
                            feedback = {"review": "Kiểm tra lại kỳ theo as_of_date, loại đơn, join, đơn vị, làm tròn, dữ liệu ngoài khoảng và type. Dùng query độc lập nếu cần. Chỉ final khi có bằng chứng; không chắc thì abstain.", "candidate": candidate.to_dict()}
                        else:
                            answer = candidate
                            event["accepted"] = True
                            trace.append(event)
                            break
                    elif action["action"] == "schema":
                        table = next((item for item in catalog.tables if item["name"] == action.get("table")), None)
                        if table is None:
                            raise DataMateError("Bảng không có trong catalog")
                        preview = executor.execute(f"SELECT * FROM {quote(table['name'])} LIMIT 3", remaining_seconds=deadline-time.monotonic())
                        feedback = {"schema": table, "sample": preview}
                    else:
                        feedback = executor.execute(action.get("sql", ""), action.get("parameters"), deadline - time.monotonic())
                        last_query = feedback
                        successful_queries += 1
                    event["observation"] = feedback
                except (DataMateError, KeyError, TypeError, ValueError) as exc:
                    feedback = {"error": str(exc), "remaining_steps": self.settings.max_steps - step - 1}
                    event["observation"] = feedback
                trace.append(event)
                messages.append({"role": "user", "content": json_text({"tool_result": feedback})})
        except Exception as exc:
            # Per-question errors become honest abstentions and preserve partial
            # batch progress. Setup/model errors are raised before the batch.
            trace.append({"error": f"{type(exc).__name__}: {exc}"})
            answer = Answer("abstain", reason=f"Lỗi xử lý: {str(exc)[:250]}")
        return RunResult(answer, time.monotonic() - started, trace, calls)
