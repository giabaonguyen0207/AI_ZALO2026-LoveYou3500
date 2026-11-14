"""One action per turn. Recognized envelopes are parsed, prose is not guessed."""
from __future__ import annotations

import json
import re
import math
from decimal import Decimal

from ..domain import Answer, ANSWER_TYPES, DataMateError, finite_number
from ..data.transforms import iso_date, normalized, round_half_up
from .native import parse_native_call

ACTIONS = {"schema", "profile", "lookup", "period", "check", "metric", "sql", "final", "prepare", "plan", "calculate"}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Trùng khóa JSON")
        result[key] = value
    return result


def parse_action(text):
    if not isinstance(text, str) or len(text)>100000:
        raise DataMateError("Output action không phải chuỗi hoặc quá dài")
    text = text.strip()
    # Keep boundaries while decoding so a thinking channel cannot be confused
    # with a tool call. Unclosed channels are rejected rather than harvested.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", text, flags=re.S).strip()
    text = re.sub(r"^(?:<\|turn>model\\?n?\s*)", "", text).strip()
    # Only remove consecutive known trailers. Preserve internal turn boundaries
    # so a second action or a model-generated conversation still fails parsing.
    text = re.sub(r"(?:(?:<turn\|>|<\|im_end\|>|<\|endoftext\|>|<eos>)\s*)+$", "", text).strip()
    # Compatibility with logs decoded by the old skip_special_tokens=True.
    text = re.sub(r"^(?:thought\s*\n)+", "", text).strip()
    envelope = None
    native=text.startswith('<|tool_call>') or re.match(r'^call:[A-Za-z_][A-Za-z_0-9]*\s*:?',text)
    if native:
        value=parse_native_call(text,ACTIONS)
    elif text.startswith("call:"):
        text = text[5:].strip()
        match = re.match(r"^(schema|profile|lookup|period|check|metric|sql|final)\s*:?(?=\s*\{)", text)
        if match:
            envelope, text = match[1], text[match.end():].strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
        if not match:
            raise DataMateError("Code fence/action chưa đóng; không tự lấy JSON dở")
        text = match[1]
    if not native:
        try:
            value = json.loads(text, object_pairs_hook=_pairs,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Số không hữu hạn")))
        except (ValueError, TypeError,RecursionError) as exc:
            raise DataMateError("Model phải trả đúng một JSON object hợp lệ; không kèm reasoning/nhiều action") from exc
    stack=[(value,0)]
    while stack:
        node,depth=stack.pop()
        if depth>32 or isinstance(node,float) and not math.isfinite(node):
            raise DataMateError('Action lồng quá sâu hoặc có số không hữu hạn')
        if isinstance(node,dict):
            stack.extend((v,depth+1) for v in node.values())
        elif isinstance(node,list):
            stack.extend((v,depth+1) for v in node)
    if not isinstance(value, dict):
        raise DataMateError("Action phải là JSON object")
    if envelope:
        if "action" in value and value["action"] != envelope:
            raise DataMateError("Tên native call khác action JSON")
        value = {"action": envelope, **value}
    if value.get("action") not in ACTIONS:
        raise DataMateError("action phải là schema/profile/lookup/period/check/metric/sql/final/prepare/plan")
    kind = value["action"]
    if kind in {"schema", "profile"} and not isinstance(value.get("table"), str):
        raise DataMateError("schema/profile cần table là chuỗi")
    if kind == "sql" and (not isinstance(value.get("sql"), str) or not isinstance(value.get("parameters", []), list)):
        raise DataMateError("sql cần sql chuỗi và parameters list")
    if kind == "final" and value.get("type") not in ANSWER_TYPES:
        raise DataMateError("final cần type hợp lệ")
    return value


def answer_from_query(answer_type, result, columns=None, percent_places=None):
    if not result or result.get("truncated"):
        raise DataMateError("Cần query đầy đủ, không bị cắt")
    rows, names = result["rows"], result["columns"]
    if columns is not None:
        if not isinstance(columns, list) or not columns or len(set(columns))!=len(columns):
            raise DataMateError("columns cần list tên cột không trùng")
        if len(set(names)) != len(names) or any(name not in names for name in columns):
            raise DataMateError("Tên cột chiếu không tồn tại hoặc query có tên cột trùng")
        indexes = [names.index(name) for name in columns]
        rows, names = [[row[i] for i in indexes] for row in rows], columns
    if answer_type == "table":
        value = {"columns": names, "rows": rows}
    elif answer_type == "list":
        if len(names)==1:
            value = [row[0] for row in rows]
        elif len(rows)==1:
            value = rows[0]
        else:
            raise DataMateError("List cần 1 cột hoặc 1 dòng; chọn columns hoặc tổng hợp bằng metric/SQL")
    else:
        if len(rows)!=1 or len(names)!=1:
            raise DataMateError(f"Đáp án đơn cần 1 dòng × 1 cột; query có {len(rows)} dòng × {len(names)} cột {names}. Dùng columns để chiếu hoặc metric để tính.")
        value = rows[0][0]
        if answer_type == "date" and isinstance(value,str):
            value = value if re.fullmatch(r"\d{4}-\d{2}",value) else iso_date(value)
        elif answer_type == "boolean":
            if type(value) in (int,bool) and value in (0,1):
                value = bool(value)
            elif isinstance(value,str) and normalized(value) in {"true","false","có","không"}:
                value = normalized(value) in {"true","có"}
        elif answer_type == "percent" and percent_places is not None:
            value = round_half_up(value, percent_places)
    return Answer(answer_type,value)


def _same(left, right):
    if finite_number(left) and finite_number(right):
        return abs(Decimal(str(left))-Decimal(str(right))) <= Decimal("0.00000001")
    if isinstance(left,str) and isinstance(right,str):
        return normalized(left)==normalized(right)
    if isinstance(left,list) and isinstance(right,list):
        return len(left)==len(right) and all(_same(a,b) for a,b in zip(left,right))
    return type(left) is type(right) and left==right


def verify_legacy_answer(action, candidate):
    """Legacy answers are checked, but values are always emitted from the query."""
    if "answer" not in action:
        return
    supplied = action["answer"]
    if candidate.type == "table":
        valid = (isinstance(supplied,dict) and set(supplied)=={"columns","rows"}
                 and isinstance(supplied["columns"],list)
                 and len(supplied["columns"])==len(candidate.answer["columns"])
                 and _same(supplied["rows"],candidate.answer["rows"]))
    else:
        try:
            normalized_answer = answer_from_query(candidate.type, {"columns":["value"],"rows":[[supplied]]}).answer if candidate.type in {"date","boolean"} else supplied
        except (DataMateError,ValueError,TypeError):
            normalized_answer = supplied
        valid = _same(normalized_answer,candidate.answer)
    if not valid:
        raise DataMateError("answer khác query cuối; dùng query_id/columns, không tự viết lại giá trị")
