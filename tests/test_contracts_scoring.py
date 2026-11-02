import pytest

from datamate.domain import Answer, DataMateError, Question
from datamate.evaluation.scoring import matches, score
from datamate.models.registry import inventory
from datamate.data.transforms import number, iso_date, normalized, round_half_up


@pytest.mark.parametrize("kind,value", [("number", True), ("number", "12"), ("percent", float("nan")),
    ("boolean", 1), ("date", "2026-02-30"), ("date", "2026-13"), ("list", [True]),
    ("table", {"columns": ["a"], "rows": [[None]]}), ("unanswerable", 0)])
def test_reject_invalid_answer_types(kind, value):
    with pytest.raises(DataMateError):
        Answer(kind, value)


def test_answer_semantics_and_ordering():
    assert matches({"type": "string", "answer": " CÀ PHÊ — SỮA "}, {"type": "string", "answer": "Cà phê sữa", "alias": []})
    assert not matches({"type": "string", "answer": "Ca phe sua"}, {"type": "string", "answer": "Cà phê sữa"})
    assert matches({"type": "string", "answer": "SP001"}, {"type": "string", "answer": "Cà phê sữa", "alias": ["SP001"]})
    assert not matches({"type": "list", "answer": [2, 1]}, {"type": "list", "answer": [1, 2]})
    assert not matches({"type": "number", "answer": 1}, {"type": "boolean", "answer": True})
    assert not matches({"type": "number", "answer": 10_000_001}, {"type": "number", "answer": 10_000_000})


def test_table_matching_reorders_named_columns_and_ignores_row_order():
    pred = {"type": "table", "answer": {"columns": ["Tiền", "Khách"], "rows": [[20, "B"], [10, "A"]]}}
    ref = {"type": "table", "answer": {"columns": ["khách", "tiền"], "rows": [["A", 10], ["B", 20]]}}
    assert matches(pred, ref)


def test_pa_distinguishes_missing_and_wrong_unanswerable():
    refs = [{"id": str(i), "type": "number", "answer": 1} for i in range(4)]
    results = [{"id": "0", "type": "number", "answer": 1}, {"id": "1", "type": "unanswerable"}, {"id": "2", "type": "abstain"}]
    report = score(results, refs)
    assert (report["correct"], report["wrong"], report["abstain"], report["penalized_accuracy"]) == (1, 1, 2, 0)


def test_ids_dates_and_parameter_accounting():
    with pytest.raises(DataMateError):
        Question("x", "../data", "q", "2026-01-01")
    assert inventory("gemma4-12b", "gemma4-12b-assistant")["total_parameters"] == 12_382_587_188
    with pytest.raises(DataMateError):
        inventory("qwen35-9b", "gemma4-12b-assistant")


def test_explicit_date_money_and_rounding_helpers():
    assert number("19,000.0000", "en") == 19000
    assert number("1.234,50", "vi") == 1234.5
    assert number("560,000") == 560000
    with pytest.raises(ValueError):
        number("1.234")
    assert round_half_up(1.005, 2) == 1.01
    assert round_half_up(0.5, 0) == 1
    assert iso_date("1 thg 6 2025") == "2025-06-01"
    assert iso_date("03/02", 2026) == "2026-02-03"
    with pytest.raises(ValueError):
        iso_date("03/02")
    assert normalized("Cà-phê") == "cà phê"
