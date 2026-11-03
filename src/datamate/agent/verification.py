"""Question-grounded contracts and period checks; never uses reference labels."""
from __future__ import annotations

import calendar
import re
import unicodedata
from datetime import date, timedelta

from ..data.rules import resolve_period
from ..data.transforms import search_key
from ..domain import DataMateError


def question_key(text):
    folded=unicodedata.normalize('NFKD',text.replace('đ','d').replace('Đ','D')).lower()
    key=' '+' '.join(''.join(ch for ch in folded if not unicodedata.combining(ch)).split())+' '
    aliases = {"dthu":"doanh thu","revenue":"doanh thu","sp":"san pham","dc":"duoc","nhiu":"nhieu",
               "bnhiu":"bao nhieu","bn":"bao nhieu","bnhiu":"bao nhieu","trc":"truoc","hnay":"hom nay",
               "hqua":"hom qua","order":"don","orders":"don","vs":"voi","ko":"khong","khoong":"khong","cos":"co"}
    for token,value in aliases.items():
        key = re.sub(r"\b"+token+r"\b",value,key)
    return key


def _basic_contract(question):
    q = question_key(question)
    money = any(x in q for x in ("doanh thu","nhieu tien","chi tieu","chi nhieu tien","ban dat"))
    if any(x in q for x in ("con no","cong no","no nhieu","dang no")):
        limit=re.search(r'\b(\d+)\s+khach\b',q)
        return {"metrics":["debt_rank"],"type":"table",**({'limit':int(limit[1])} if limit else {})}
    if any(x in q for x in ("ton kho","trong kho","con ton")):
        units = any(x in q for x in ("nguyen va","chan","le bao nhieu","bao nhieu cuon le","may vien le"))
        return {"metrics":["inventory_units" if units else "inventory"],"type":"list" if units else "number"}
    if "ngay nao" in q and (money or "cao nhat" in q or "nhieu tien nhat" in q):
        return {"metrics":["best_day"],"type":"date"}
    if 'nhan vien' in q and any(x in q for x in ('gioi nhat','nhieu nhat','cao nhat')):
        return {'metrics':['rank_employees'],'type':'string'}
    if "ca sang" in q and "ca chieu" in q:
        return {"metrics":["shift_growth"],"type":"percent"}
    if "zalo" in q and "quay" in q:
        return {"metrics":["channel_revenue"],"type":"list"}
    if any(x in q for x in ("tang lien tuc","luon cao hon","thang sau co luon","xu huong doanh thu")):
        return {"metrics":["continuous_growth"],"type":"boolean"}
    if money and any(x in q for x in ('ghi lan luot','theo thu tu do','ghi lan luot theo thu tu')):
        return {'metrics':['revenue_series'],'type':'list'}
    if "khach" in q and any(x in q for x in ("it nhat","tro len","tu 3 don","tu 8 don","tu 14 don")):
        percent = "%" in question or "phan tram" in q or "ti le" in q
        threshold=re.search(r'(?:it nhat|tu)\s+(\d+)\s+don',q)
        return {"metrics":["customer_percent" if percent else "customers_with_orders"],"type":"percent" if percent else "number",
                **({'threshold':int(threshold[1])} if threshold else {})}
    if money and ("%" in question or "phan tram" in q):
        return {"metrics":["revenue_growth"],"type":"percent"}
    if ("%" in question or "phan tram" in q) and any(x in q for x in ("so ly","so binh","so luong","ban ra","ban duoc")):
        return {"metrics":["quantity_growth"],"type":"percent"}
    k = re.search(r"\b(?:top|liet ke)\s+(\d+)\b",q)
    if "khach" in q and any(x in q for x in ("nhieu nhat","nhat","top","liet ke")):
        return {"metrics":["rank_customers"],"type":"list" if k else "string",
                "limit":int(k[1]) if k else 1,"by":"orders" if "don" in q and not money else "revenue"}
    if any(x in q for x in ("ban chay","ban nhieu nhat","ban duoc nhieu nhat","ban nhieu nhat","ban duoc nhieu nhat")):
        return {"metrics":["rank_products"],"type":"list" if k else "string","limit":int(k[1]) if k else 1,
                "by":"revenue" if "theo doanh thu" in q else "quantity"}
    if "bao nhieu ngay" in q and money:
        return {"metrics":["day_count"],"type":"number"}
    if "don" in q and "huy" not in q and any(x in q for x in ("tong cong","bao nhieu","co bao nhieu")):
        return {"metrics":["orders"],"type":"number"}
    if money:
        return {"metrics":["revenue"],"type":"number"}
    return {"metrics":[],"type":None}


def question_contract(question):
    result=_basic_contract(question)
    q=question_key(question)
    extra=[x for x in ('nhom khach','nhom hang','khach le','vat','phi giao','tien mat','chuyen khoan','cuoi tuan',
                      'thu bay','chu nhat','tet','theo tung','tung san pham','tung khach') if x in q]
    if extra:
        return {'metrics':[],'type':None,'suggested_metrics':result['metrics'],'extra_constraints':extra,
                'next_action':'SQL trên facts đã chuẩn hóa; không dùng metric tổng rồi bỏ qua điều kiện'}
    return result


def question_periods(question,as_of_date,policy):
    q = question_key(question)
    specs = []
    explicit_days = list(re.finditer(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)",q))
    days = re.search(r"\b(\d+) ngay\b",q)
    if explicit_days and days and ("ke tu" in q or "ngay truoc" in q):
        d,m,y = map(int,explicit_days[0].groups());anchor=date(y,m,d).isoformat();n=int(days[1])
        specs=[{"kind":"days_from","date":anchor,"days":n},{"kind":"days_before","date":anchor,"days":n}]
    elif explicit_days:
        specs=[{"kind":"range","start":date(int(m[3]),int(m[2]),int(m[1])).isoformat(),"end":date(int(m[3]),int(m[2]),int(m[1])).isoformat()} for m in explicit_days]
    else:
        quarters = list(re.finditer(r"quy\s+(\d)\s*(?:/|nam\s*)?(\d{4})",q))
        if quarters:
            specs=[{"kind":"quarter","quarter":int(m[1]),"year":int(m[2])} for m in quarters]
        else:
            matches=list(re.finditer(r"(?<![\d/])(\d{1,2})\s*(?:/|nam\s+)(\d{4})(?!\d)",q))
            specs=[{"kind":"month","month":int(m[1]),"year":int(m[2])} for m in matches if 1<=int(m[1])<=12]
    if not specs:
        relative = [("thang nay","this_month"),("tuan truoc","last_week"),("tuan roi","last_week"),
                    ("tuan nay","this_week"),("hom qua","yesterday"),("hom nay","today")]
        kind=next((kind for text,kind in relative if text in q),None)
        if kind:
            specs=[{"kind":kind}]
    if len(specs)==1 and specs[0]['kind']=='month' and any(x in q for x in ("thang truoc","th truoc","cung ky nam truoc")):
        current=specs[0]
        if "cung ky nam truoc" in q:
            specs.append({**current,'year':current['year']-1})
        else:
            previous=date(current['year'],current['month'],1)-timedelta(days=1)
            specs.append({'kind':'month','year':previous.year,'month':previous.month})
    if len(specs)==1 and specs[0]['kind']=='this_month' and any(x in q for x in ('thang truoc','th truoc')):
        specs.append({'kind':'previous_month_to_date'})
    return [resolve_period(spec,as_of_date,policy) for spec in specs]


def verify_metric_request(action,question,as_of_date,policy):
    contract=question_contract(question)
    if contract.get('extra_constraints'):
        raise DataMateError('Câu hỏi có điều kiện ngoài template metric: '+str(contract['extra_constraints'])+'. Dùng SQL trên facts/clean dimensions để giữ đủ điều kiện')
    if contract['metrics'] and action.get('name') not in contract['metrics']:
        raise DataMateError("Đại lượng không khớp câu hỏi; metric cần là "+str(contract['metrics']))
    if 'limit' in contract and action.get('limit',1)!=contract['limit']:
        raise DataMateError("limit phải theo top k trong câu hỏi")
    if 'by' in contract and action.get('by',contract['by'])!=contract['by']:
        raise DataMateError("Tiêu chí xếp hạng khác câu hỏi/rulebook")
    if 'by' in contract:
        action.setdefault('by',contract['by'])
    if 'threshold' in contract and action.get('threshold')!=contract['threshold']:
        raise DataMateError('Ngưỡng số đơn không khớp câu hỏi')
    periods=question_periods(question,as_of_date,policy)
    if periods:
        specs=[{'kind':'range','start':p['start'],'end':p['end']} for p in periods]
        if action.get('name') in {'continuous_growth','revenue_series'}:
            action.setdefault('periods',specs)
        elif action.get('name') not in {'debt_rank','inventory','inventory_units'}:
            action.setdefault('period',specs[0])
            if action.get('name') in {'revenue_growth','quantity_growth'} and len(specs)>1:
                action.setdefault('compare_period',specs[1])
    if action.get('name') in {'inventory','inventory_units','debt_rank'}:
        expected_date=periods[0]['end'] if periods and periods[0]['start']==periods[0]['end'] else as_of_date
        action.setdefault('date',expected_date)
        if action.get('date',as_of_date)!=expected_date:
            raise DataMateError('Ngày chốt không khớp câu hỏi/as_of_date; không dùng snapshot tương lai')
    if periods and action.get('period'):
        resolved=resolve_period(action['period'],as_of_date,policy)
        if (resolved['start'],resolved['end'])!=(periods[0]['start'],periods[0]['end']):
            raise DataMateError("Kỳ hỏi không khớp thời gian trong câu hỏi/as_of_date: "+str(periods))
    if len(periods)>=2 and action.get('compare_period'):
        resolved=resolve_period(action['compare_period'],as_of_date,policy)
        if (resolved['start'],resolved['end'])!=(periods[1]['start'],periods[1]['end']):
            raise DataMateError("Kỳ so sánh không khớp câu hỏi: "+str(periods))
    if action.get('periods') and periods:
        resolved=[resolve_period(p,as_of_date,policy) for p in action['periods']]
        if [(p['start'],p['end']) for p in resolved] != [(p['start'],p['end']) for p in periods]:
            raise DataMateError("Các kỳ xu hướng không khớp thứ tự trong câu hỏi")
    return contract
