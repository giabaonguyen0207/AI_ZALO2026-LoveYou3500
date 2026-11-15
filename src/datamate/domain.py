"""Shared contracts. This module has no infrastructure dependencies."""
from __future__ import annotations

import calendar
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

ANSWER_TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}


class DataMateError(Exception):
    """A user-visible, recoverable application error."""


def finite_number(value: Any) -> bool:
    return type(value) in (int, float) and (type(value) is int or math.isfinite(value))


def json_text(value: Any, **kwargs: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, **kwargs)


@dataclass(frozen=True)
class Question:
    id: str
    business_id: str
    question: str
    as_of_date: str

    def __post_init__(self) -> None:
        for name in ("id", "business_id", "question", "as_of_date"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DataMateError(f"{name} phải là chuỗi không rỗng")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.business_id):
            raise DataMateError("business_id không hợp lệ")
        try:
            date.fromisoformat(self.as_of_date)
        except ValueError as exc:
            raise DataMateError("as_of_date phải là YYYY-MM-DD") from exc

    @classmethod
    def from_dict(cls, value: dict) -> Question:
        try:
            return cls(**{key: value[key] for key in ("id", "business_id", "question", "as_of_date")})
        except (KeyError, TypeError) as exc:
            raise DataMateError("Câu hỏi thiếu trường bắt buộc") from exc

    def prompt_dict(self) -> dict:
        # Routing IDs never enter model context or determine an answer.
        return {"question": self.question, "as_of_date": self.as_of_date}


@dataclass(frozen=True)
class Answer:
    type: str
    answer: Any = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.type not in ANSWER_TYPES:
            raise DataMateError(f"type không hỗ trợ: {self.type}")
        if self.reason is not None and not isinstance(self.reason, str):
            raise DataMateError("reason phải là chuỗi")
        value = self.answer
        valid = True
        if self.type in {"number", "percent"}:
            valid = finite_number(value)
        elif self.type == "boolean":
            valid = type(value) is bool
        elif self.type == "string":
            valid = isinstance(value, str)
        elif self.type == "date":
            valid = isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", value))
            if valid:
                try:
                    date.fromisoformat(value if len(value) == 10 else value + "-01")
                except ValueError:
                    valid = False
        elif self.type == "list":
            valid = isinstance(value, list) and all(isinstance(x, str) or finite_number(x) for x in value)
        elif self.type == "table":
            valid = (isinstance(value, dict) and set(value) == {"columns", "rows"}
                     and isinstance(value["columns"], list) and bool(value["columns"])
                     and all(isinstance(x, str) for x in value["columns"])
                     and len(value["columns"]) == len(set(value["columns"]))
                     and isinstance(value["rows"], list))
            if valid:
                valid = all(isinstance(row, list) and len(row) == len(value["columns"])
                            and all(isinstance(x, str) or finite_number(x) or type(x) is bool for x in row)
                            for row in value["rows"])
        elif self.type in {"unanswerable", "abstain"}:
            valid = value is None
        if not valid:
            raise DataMateError(f"Giá trị không đúng định dạng {self.type}")

    @classmethod
    def from_dict(cls, value: dict) -> Answer:
        if not isinstance(value, dict) or "type" not in value:
            raise DataMateError("Đáp án phải là object có type")
        return cls(value["type"], value.get("answer"), value.get("reason"))

    def to_dict(self, question_id: str | None = None) -> dict:
        out = {"type": self.type}
        if question_id is not None:
            out = {"id": question_id, **out}
        if self.type not in {"abstain", "unanswerable"}:
            out["answer"] = self.answer
        if self.reason is not None:
            out["reason"] = self.reason
        return out


@dataclass
class RunResult:
    answer: Answer
    elapsed_seconds: float
    trace: list[dict] = field(default_factory=list)
    model_calls: int = 0
    initial_messages: list[dict] = field(default_factory=list)
    executions: list[dict] | None = None

    def to_dict(self, question_id: str) -> dict:
        return {"submission": self.answer.to_dict(question_id), "elapsed_seconds": round(self.elapsed_seconds, 4),
                "model_calls": self.model_calls, "trace": self.trace,
                "initial_messages": self.initial_messages, "executions": self.executions}
