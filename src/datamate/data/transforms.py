"""Explicit SQL helpers: never overwrite or silently normalize source cells."""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP


def number(value, locale="auto"):
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace("\u00a0", "").replace(" ", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    if locale == "vi":
        text = text.replace(".", "").replace(",", ".")
    elif locale == "en":
        text = text.replace(",", "")
    elif locale == "auto":
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", text):
            text = text.replace(",", "")
        elif re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+", text):
            # A single 1.234 is ambiguous: choose a locale explicitly.
            if text.count(".") == 1:
                raise ValueError("Số mơ hồ: dùng num_vi hoặc num_en cho dấu chấm")
            text = text.replace(".", "")
        else:
            text = text.replace(",", ".")
    else:
        raise ValueError("locale phải là auto, vi, en")
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Không đọc được số") from exc
    if not value.is_finite():
        raise ValueError("Số không hữu hạn")
    if negative:
        value = -value
    return int(value) if value == value.to_integral() else float(value)


def iso_date(value, year=None):
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return date.fromisoformat(text[:10]).isoformat()
    match = re.match(r"^(\d{1,2})\s+thg\s+(\d{1,2})\s+(\d{4})", text, re.I)
    if match:
        day, month, yr = map(int, match.groups())
        return date(yr, month, day).isoformat()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    if year is not None and re.fullmatch(r"\d{1,2}[/-]\d{1,2}", text):
        day, month = map(int, re.split(r"[/-]", text))
        return date(int(year), month, day).isoformat()
    raise ValueError("Ngày không rõ định dạng/năm; đọc README, dùng iso_date_year nếu thiếu năm")


def round_half_up(value, places=0):
    if value is None:
        return None
    places = int(places)
    if not -9 <= places <= 9:
        raise ValueError("Số chữ số làm tròn ngoài giới hạn")
    rounded = Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return int(rounded) if places <= 0 else float(rounded)


def normalized(value):
    if value is None:
        return None
    text = unicodedata.normalize("NFC", str(value)).lower()
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def register_functions(connection):
    for name, arity, fn in (
        ("num", 1, number), ("num_vi", 1, lambda x: number(x, "vi")),
        ("num_en", 1, lambda x: number(x, "en")), ("iso_date", 1, iso_date),
        ("iso_date_year", 2, iso_date), ("norm", 1, normalized),
        ("round_half_up", 2, round_half_up),
    ):
        connection.create_function(name, arity, fn, deterministic=True)
