"""Whole-column evidence, including rare values outside the first preview rows."""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from .transforms import iso_date, typed_date, numeric_text, parse_number, search_key, status_class, payment_class, quantity_coefficient


def quote(name):
    return '"' + name.replace('"', '""') + '"'


def infer_locale(values, readme=""):
    votes = Counter()
    note = search_key(readme) or ""
    if "phay phan cach nghin" in note or "phay nghin" in note:
        votes["en"] += 100
    if "cham phan cach nghin" in note or "cham nghin" in note:
        votes["vi"] += 100
    for value in dict.fromkeys(values):
        if not isinstance(value, str) or not value.strip():
            continue
        text, scale = numeric_text(value)
        text = text.replace(" ", "")
        if scale != 1:
            continue
        if "," in text and "." in text:
            votes["vi" if text.rfind(",") > text.rfind(".") else "en"] += 1
        elif re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", text):
            votes["en"] += 1
        elif re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+", text):
            # Currency-marked 50.000đ supplies evidence that a bare 1.234
            # does not. Do not silently choose a locale for the latter.
            if text.count(".") > 1 or re.search(r"(?:đ|₫|vnd|đồng)\s*$", value, re.I):
                votes["vi"] += 1
        elif re.fullmatch(r"[-+]?\d+,\d{1,2}", text):
            votes["vi"] += 1
        elif re.fullmatch(r"[-+]?\d+\.\d{1,2}|[-+]?\d+\.\d{4,}", text):
            votes["en"] += 1
    if votes["en"] and votes["vi"]:
        return "mixed"
    return "en" if votes["en"] else "vi" if votes["vi"] else "auto"


@lru_cache(maxsize=65536)
def parse_profile_number(value, locale):
    if locale != "mixed":
        return parse_number(value, locale)
    text, scale = numeric_text(value) if isinstance(value, str) else (str(value), 1)
    if scale != 1 or not isinstance(value, str):
        return parse_number(value)
    if re.search(r"(?:đ|₫|vnd|đồng)\s*$", value, re.I) and "." in text and "," not in text:
        return parse_number(value, "vi")
    # Both separators, comma thousands and decimal coefficients are explicit.
    # A bare 1.234 remains an error when the column has conflicting evidence.
    return parse_number(value)


def column_profile(values, role=None, readme="", years=None, units=None, excel_epoch=None):
    count = len(values)
    present = [v for v in values if v is not None and str(v).strip()]
    types = Counter(type(v).__name__ for v in values)
    result = {"scope": "all_rows", "rows_scanned": count, "nonempty": len(present),
              "empty": count-len(present), "storage_types": dict(types), "role": role}
    if role in {"status",'payment_method'}:
        counts = Counter(str(v) for v in present)
        result.update(distinct_count=len(counts), values_complete=True,
                      values=[{"value": v, "count": n, "canonical": status_class(v) if role=='status' else payment_class(v)}
                              for v, n in sorted(counts.items())])
    if role in {"number", "money", "quantity"}:
        if role=='quantity' and units is not None:
            present=[quantity_coefficient(v,u) for v,u in zip(values,units) if v is not None and str(v).strip()]
        locale = infer_locale(present, readme)
        good, bad, patterns, examples = [], Counter(), Counter(), {}
        for value in present:
            try:
                parsed = parse_profile_number(value, locale)
                good.append(parsed)
                _, scale = numeric_text(value)
                shape = "scaled" if scale != 1 else "currency" if re.search(r"[đ₫]|vnd", str(value), re.I) else "plain"
                patterns[shape] += 1
                examples.setdefault(shape, str(value))
            except (ValueError, TypeError, ArithmeticError) as exc:
                bad[(str(value), str(exc))] += 1
        result.update(locale=locale, parsed=len(good), parse_errors=sum(bad.values()),
                      patterns=dict(patterns), examples=examples,
                      error_examples=[{"value": v, "error": e, "count": n} for (v,e),n in bad.most_common(5)])
        if good:
            result.update(min=min(good), max=max(good))
    if role == "date":
        good, bad = [], Counter()
        for i, value in enumerate(values):
            if value is None or not str(value).strip():
                continue
            try:
                good.append(typed_date(value, years[i] if years else None,excel_epoch))
            except (ValueError, TypeError) as exc:
                bad[(str(value), str(exc))] += 1
        result.update(parsed=len(good), parse_errors=sum(bad.values()),
                      error_examples=[{"value": v, "error": e, "count": n} for (v,e),n in bad.most_common(5)])
        if excel_epoch:
            result['excel_epoch']=excel_epoch
            result['serial_date_cells']=sum(type(v) in (int,float) for v in values)
        if good:
            result.update(min=min(good), max=max(good))
    return result


def inspect_column(connection, table, column, offset=0, limit=50):
    """Exact distinct values with pagination; never call a top-k page complete."""
    if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset >= 0; limit trong 1..100")
    total = connection.execute(f"SELECT COUNT(*) FROM (SELECT {quote(column)} FROM {quote(table)} GROUP BY 1)").fetchone()[0]
    rows = connection.execute(f"SELECT {quote(column)},COUNT(*) FROM {quote(table)} GROUP BY 1 ORDER BY 2 ASC,1 LIMIT ? OFFSET ?", (limit,offset)).fetchall()
    return {"scope": "all_rows", "distinct_count": total, "offset": offset,
            "values": [{"value": v, "count": n} for v,n in rows], "values_complete": offset == 0 and len(rows) == total,
            "next_offset": offset+len(rows) if offset+len(rows)<total else None}
