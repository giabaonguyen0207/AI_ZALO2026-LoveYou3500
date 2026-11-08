"""Explicit SQL helpers: never overwrite or silently normalize source cells."""
from __future__ import annotations

import re
import math
import unicodedata
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache


def number(value, locale="auto"):
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, (int, float)):
        if type(value) is bool or not math.isfinite(value):
            raise ValueError("Số không hữu hạn/boolean")
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


@lru_cache(maxsize=65536)
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


@lru_cache(maxsize=65536)
def search_key(value):
    """Accent folding is for lookup only; never use it to format an answer."""
    if value is None:
        return None
    text = unicodedata.normalize("NFKD", str(value).replace("đ", "d").replace("Đ", "D"))
    return normalized("".join(ch for ch in text if not unicodedata.combining(ch)))


def numeric_text(value):
    text = str(value).strip().lower().replace("\u00a0", " ")
    text = re.sub(r"\s*(?:vnd|vnđ|đồng|dong|₫|đ)\s*$", "", text).strip()
    match = re.fullmatch(r"(.+?)\s*(triệu|trieu|tr|nghìn|nghin|ngàn|ngan|k|tỷ|ty)\s*", text)
    multiplier = Decimal(1)
    if match:
        text, suffix = match.groups()
        multiplier = Decimal(1_000_000 if suffix in {"triệu", "trieu", "tr"} else
                             1_000_000_000 if suffix in {"tỷ", "ty"} else 1000)
    return text.strip(), multiplier


def parse_number(value, locale="auto"):
    """Explicit scale/currency support; ambiguous bare separators still fail."""
    if value is None or str(value).strip() == "":
        return None
    if type(value) in (int, float):
        return number(value)
    compound=re.fullmatch(r'\s*([-+]?)(\d+)\s*(?:tr|triệu|trieu)\s*(\d{1,3})\s*(?:đ|₫|vnd)?\s*',str(value),re.I)
    if compound:
        sign,whole,tail=compound.groups()
        return (-1 if sign=='-' else 1)*(int(whole)*1000000+int(tail.ljust(3,'0'))*1000)
    text, multiplier = numeric_text(value)
    if multiplier != 1:
        # 1,156tr is 1.156 million, not 1,156 million. Scaled amounts use
        # a decimal coefficient; grouping separators are not guessed here.
        if "," in text and "." not in text:
            text = text.replace(",", ".")
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            raise ValueError("Hệ số tiền viết tắt không rõ định dạng")
        result = Decimal(text) * multiplier
    else:
        result = Decimal(str(number(text, locale)))
    if not result.is_finite():
        raise ValueError("Số không hữu hạn")
    return int(result) if result == result.to_integral() else float(result)


def quantity_coefficient(value, unit):
    if isinstance(value,str) and unit:
        match=re.fullmatch(r'\s*([-+]?\d+(?:[.,]\d+)*)\s+(.+?)\s*',value)
        if match and search_key(match[2])==search_key(unit):
            return match[1]
    return value


def parse_quantity(value, unit=None, locale='auto'):
    return parse_number(quantity_coefficient(value,unit),locale)


@lru_cache(maxsize=65536)
def typed_date(value, year=None, excel_epoch=None):
    numeric = type(value) in (int,float) or isinstance(value,str) and re.fullmatch(r'\d{1,6}(?:\.\d+)?',value.strip())
    if numeric and excel_epoch:
        from openpyxl.utils.datetime import from_excel
        value=float(value)
        if value<1:
            raise ValueError('Serial Excel không có phần ngày')
        result=from_excel(value,epoch=datetime.fromisoformat(excel_epoch))
        return result.date().isoformat()
    return iso_date(value,year)


STATUS_ALIASES = {
    "completed": {"completed", "complete", "done", "xong", "hoan thanh", "da hoan thanh", "hoan tat",
                  "da giao", "da giao hang", "da lay", "thanh cong", "da ban",'da xong','ok'},
    "cancelled": {"cancelled", "canceled", "cancel", "huy", "da huy", "da huy don", "huy don",
                  "khach huy", "don huy",'khong giao duoc'},
    "returned": {"returned", "return", "tra hang", "da tra hang", "doi tra", "hoan tra", "hoan hang",
                 "da hoan hang", "tra mot phan", "tra hang mot phan",'hang tra ve','tra lai'},
    "shipping": {"shipping", "dang giao", "dang giao hang", "dang van chuyen", "dang giao dich",
                 "dang xu ly", "processing",'cho giao','dang gui'},
}


def status_class(value):
    key = search_key(value)
    if not key:
        return None
    return next((kind for kind, aliases in STATUS_ALIASES.items() if key in aliases), "unknown")


def payment_class(value):
    key = search_key(value)
    groups = {"debt": {"debt", "ghi no", "ban no", "cong no", "no"},
              "cod": {"cod", "giao hang thu tien",'thu ho cod','ship cod','cash on delivery'}, "cash": {"cash", "tien mat", "tm"},
              "transfer": {"transfer", "chuyen khoan", "ck", "bank transfer"},
              "ewallet": {"ewallet", "vi dien tu"}}
    groups['marketplace']={'san tmdt','tmdt','marketplace','shopee','lazada','tiki'}
    if not key:
        return None
    return next((kind for kind, names in groups.items() if key in names), "unknown")


def register_functions(connection):
    for name, arity, fn in (
        ("num", 1, number), ("num_vi", 1, lambda x: number(x, "vi")),
        ("num_en", 1, lambda x: number(x, "en")), ("iso_date", 1, iso_date),
        ("iso_date_year", 2, iso_date), ("norm", 1, normalized),
        ("round_half_up", 2, round_half_up),
        ("search_key", 1, search_key), ("status_class", 1, status_class),
        ("parse_number", 1, parse_number), ("parse_number_locale", 2, parse_number),
    ):
        connection.create_function(name, arity, fn, deterministic=True)
