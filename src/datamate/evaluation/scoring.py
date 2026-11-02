from __future__ import annotations

from collections import Counter
from decimal import Decimal
from pathlib import Path

from ..data.split import read_jsonl
from ..data.transforms import normalized
from ..domain import Answer, DataMateError, finite_number, json_text


def same_cell(left, right) -> bool:
    if finite_number(left) and finite_number(right):
        return abs(Decimal(str(left)) - Decimal(str(right))) <= Decimal("0.00000001")
    if isinstance(left, str) and isinstance(right, str):
        return normalized(left) == normalized(right)
    return type(left) is type(right) and left == right


def same_value(answer_type: str, predicted, reference) -> bool:
    if answer_type in {"unanswerable", "abstain"}:
        return True
    if answer_type == "date":
        return predicted == reference
    if answer_type == "list":
        return isinstance(reference, list) and len(predicted) == len(reference) and all(same_cell(a, b) for a, b in zip(predicted, reference))
    if answer_type == "table":
        if not isinstance(reference, dict) or len(predicted["columns"]) != len(reference.get("columns", [])):
            return False
        pcols = [normalized(x) for x in predicted["columns"]]
        rcols = [normalized(x) for x in reference["columns"]]
        order = [pcols.index(x) for x in rcols] if len(set(pcols)) == len(pcols) and set(pcols) == set(rcols) else list(range(len(pcols)))
        def canonical(value):
            if isinstance(value, str):
                return ("string", normalized(value))
            if type(value) is bool:
                return ("boolean", value)
            if finite_number(value):
                return ("number", Decimal(str(value)))
            return ("invalid", str(value))
        prows = {tuple(canonical(row[i]) for i in order) for row in predicted["rows"]}
        rrows = {tuple(canonical(x) for x in row) for row in reference["rows"]}
        return prows == rrows
    return same_cell(predicted, reference)


def matches(predicted: dict, reference: dict) -> bool:
    try:
        answer = Answer.from_dict(predicted)
    except (DataMateError, TypeError):
        return False
    if answer.type != reference.get("type"):
        return False
    values = [reference.get("answer"), *reference.get("alias", [])]
    return any(same_value(answer.type, answer.answer, value) for value in values)


def score(predictions: list[dict], references: list[dict], question_ids: set[str] | None = None) -> dict:
    mapping = {}
    for row in predictions:
        if row.get("id") in mapping:
            raise DataMateError("Trùng ID trong submission")
        mapping[row.get("id")] = row
    selected = [row for row in references if question_ids is None or row["id"] in question_ids]
    ref_ids = {row["id"] for row in selected}
    if question_ids is not None and question_ids != ref_ids:
        raise DataMateError("Danh sách ID có câu không nằm trong train references")
    counts, details = Counter(), []
    for ref in selected:
        prediction = mapping.get(ref["id"], {"type": "abstain"})
        if prediction.get("type") == "abstain":
            result, points = "abstain", 0
        elif matches(prediction, ref):
            result, points = "correct", 1
        else:
            result, points = "wrong", -1
        counts[result] += 1
        details.append({"id": ref["id"], "result": result, "points": points, "type": ref["type"]})
    n = len(selected)
    return {"questions": n, "correct": counts["correct"], "wrong": counts["wrong"], "abstain": counts["abstain"],
            "penalized_accuracy": (counts["correct"] - counts["wrong"]) / n if n else 0,
            "coverage": (counts["correct"] + counts["wrong"]) / n if n else 0,
            "local_evaluator": True, "details": details}
