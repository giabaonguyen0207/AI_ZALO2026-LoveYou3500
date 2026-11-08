"""Small, explicit rulebook adapter. Unsupported wording is reported, not guessed."""
from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass
from datetime import date, timedelta

from .transforms import search_key
from ..domain import DataMateError


@dataclass
class RulePolicy:
    sales_statuses: list[str] | None
    exclude_whole_returned_order: bool
    deduplicate_order_rows: bool
    fill_order_fields: bool
    week_start: int | None
    line_revenue: bool
    unknown: list[str]
    debt_supported: bool = False
    quantity_ranking: bool = False
    exclude_service_product: bool = False
    shifts: dict | None = None
    percent_places: int | None = None
    gross_profit_line_cost: bool = False
    money_places: int | None = None
    price_all_states: bool = False
    debt_flow_supported: bool = False

    def to_dict(self):
        return asdict(self)


def section(text, number):
    match = re.search(rf"(?ms)^##\s+{number}[.\s].*?(?=^##\s+\d|\Z)", text)
    return match.group() if match else text


def read_policy(rulebook):
    sales = search_key(section(rulebook, 1)) or ""
    layout = search_key(section(rulebook, 8)) or ""
    timing = search_key(section(rulebook, 2)) or ""
    no_cancel = any(x in sales for x in ("khong huy", "bo don huy", "loai don huy", "khong tinh don huy"))
    no_return = any(x in sales for x in ("khong tra hang", "bo ca don", "loai don tra"))
    include_cancel = "tinh ca don huy" in sales
    include_return = "tinh ca don tra" in sales
    shipping = "don dang giao van tinh" in sales or "tinh ca don dang giao" in sales
    statuses = None
    unknown = []
    if (no_cancel or include_cancel) and (no_return or include_return) and shipping:
        statuses = ["completed", "shipping"]
        if include_cancel:
            statuses.append("cancelled")
        if include_return:
            statuses.append("returned")
    else:
        unknown.append("Chưa giải được đầy đủ luật trạng thái doanh thu/số đơn của rulebook này")
    line_revenue = "thanh tien cac dong hang" in sales or "tong thanh tien dong" in sales
    if not line_revenue:
        unknown.append("Chưa xác nhận định nghĩa doanh thu từ thành tiền dòng")
    week_start = 0 if "thu hai" in timing else 6 if "bat dau chu nhat" in timing else None
    debt = search_key(section(rulebook,4)) or ''
    ranking = search_key(section(rulebook,3)) or ''
    shifts={}
    for key,label in [('morning','sáng'),('afternoon','chiều')]:
        match=re.search(r'ca\s+'+label+r'\s+(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})',section(rulebook,2).replace('**',''),re.I)
        if match:
            h,m,eh,em=map(int,match.groups());end=eh*60+em+1
            shifts[key]=[f'{h:02}:{m:02}',f'{end//60:02}:{end%60:02}']
    rounding=re.search(r'phần trăm[^\n]*?(\d+)\s+chữ số',section(rulebook,9),re.I)
    return RulePolicy(statuses, no_return and "ca don" in sales,
                      "cung mot don" in layout and "tinh mot lan" in layout,
                      "dong dau nhom" in layout and "o de trong" in layout,
                      week_start, line_revenue, unknown,
                      debt_supported='don khong huy' in debt and 'don tra hang van tinh' in debt and 'khong phai khach no' in debt,
                      quantity_ranking='ban chay' in ranking and 'so luong quy ve don vi co ban' in ranking,
                      exclude_service_product='phi phuc vu' in ranking and 'khong phai mat hang' in ranking,
                      shifts=shifts if len(shifts)==2 else None,
                      percent_places=int(rounding[1]) if rounding else None,
                      gross_profit_line_cost='gia von ghi tren dong ban' in (search_key(section(rulebook,5)) or ''),
                      money_places=0 if any(x in (search_key(rulebook) or '') for x in ('lam tron ve dong','tien la so nguyen vnd')) else None,
                      price_all_states='tinh ca dong cua don huy don tra' in (search_key(section(rulebook,5)) or ''),
                      debt_flow_supported=all(x in debt for x in ('no them','sau ngay mua','cung ngay mua','ghi no')))


def resolve_period(spec, as_of_date, policy):
    """Inclusive ISO dates. No system clock; calendar-dependent holidays stay SQL."""
    if not isinstance(spec, dict):
        raise DataMateError("period phải là object")
    today = date.fromisoformat(as_of_date)
    kind = spec.get("kind")
    allowed={'today':set(),'yesterday':set(),'this_week':set(),'last_week':set(),'this_month':set(),
             'previous_month_to_date':set(),'year_to_date':set(),'month':{'month','year'},'quarter':{'quarter','year'},
             'range':{'start','end'},'last_days':{'days','date'},'days_from':{'days','date'},'days_before':{'days','date'},
             'month_offset':{'anchor','months','to_date'}}
    if kind in allowed and set(spec)-({'kind'}|allowed[kind]):
        raise DataMateError('period có trường không được dùng: '+str(sorted(set(spec)-({'kind'}|allowed[kind]))))
    if kind=='month_offset':
        anchor=spec.get('anchor');offset=spec.get('months')
        if not isinstance(anchor,dict) or anchor.get('kind') not in {'month','this_month'} or type(offset) is not int or not -120<=offset<=120:
            raise DataMateError('month_offset cần anchor month/this_month và months nguyên trong -120..120')
        resolved=resolve_period(anchor,as_of_date,policy)
        first=date.fromisoformat(resolved['start']);index=first.year*12+first.month-1+offset
        year,month=index//12,index%12+1
        last=calendar.monthrange(year,month)[1]
        if type(spec.get('to_date',False)) is not bool:raise DataMateError('month_offset.to_date cần boolean')
        if spec.get('to_date'):last=min(last,date.fromisoformat(resolved['end']).day)
        return {'start':date(year,month,1).isoformat(),'end':date(year,month,last).isoformat(),'inclusive':True,'as_of_date':as_of_date}
    if kind == 'year_to_date':
        start, end = today.replace(month=1, day=1), today
    elif kind in {"today", "yesterday"}:
        start = end = today-timedelta(days=kind == "yesterday")
    elif kind in {"this_week", "last_week"}:
        if policy.week_start is None:
            raise DataMateError("Rulebook chưa xác nhận ngày bắt đầu tuần")
        monday = today-timedelta(days=(today.weekday()-policy.week_start) % 7)
        start, end = (monday, today) if kind == "this_week" else (monday-timedelta(days=7), monday-timedelta(days=1))
    elif kind in {"this_month", "previous_month_to_date"}:
        if kind == "this_month":
            start, end = today.replace(day=1), today
        else:
            previous = today.replace(day=1)-timedelta(days=1)
            start = previous.replace(day=1)
            end = previous.replace(day=min(today.day, previous.day))
    elif kind in {"month", "quarter"}:
        value = spec.get("month") if kind == "month" else spec.get("quarter")
        if type(value) is not int or not 1 <= value <= (12 if kind == "month" else 4):
            raise DataMateError("month/quarter không hợp lệ")
        month = value if kind == "month" else 3*(value-1)+1
        year = spec.get("year", today.year)
        if type(year) is not int or not 1900 <= year <= 2200:
            raise DataMateError("year không hợp lệ")
        if "year" not in spec and date(year, month, 1) > today:
            year -= 1
        last_month = month if kind == "month" else month+2
        start, end = date(year, month, 1), date(year, last_month, calendar.monthrange(year, last_month)[1])
    elif kind == "range":
        try:
            start, end = date.fromisoformat(spec["start"]), date.fromisoformat(spec["end"])
        except (KeyError, ValueError, TypeError) as exc:
            raise DataMateError("range cần start/end ISO") from exc
    elif kind in {"last_days", "days_from", "days_before"}:
        n = spec.get("days")
        if type(n) is not int or not 1 <= n <= 3660:
            raise DataMateError("days phải trong 1..3660")
        anchor = date.fromisoformat(spec.get("date", as_of_date))
        if kind == "days_from":
            start, end = anchor, anchor+timedelta(days=n-1)
        elif kind == "days_before":
            start, end = anchor-timedelta(days=n), anchor-timedelta(days=1)
        else:
            start, end = anchor-timedelta(days=n-1), anchor
    else:
        raise DataMateError("period kind chưa hỗ trợ; dùng range ISO hoặc tra calendar bằng SQL")
    if start > end:
        raise DataMateError("Ngày bắt đầu sau ngày kết thúc")
    return {"start": start.isoformat(), "end": end.isoformat(), "inclusive": True,
            "as_of_date": as_of_date}
