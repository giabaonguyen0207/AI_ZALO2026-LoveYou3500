"""Evidence and data checks for model-proposed business interpretations.

No question, reference answer or business identifier is an input to these checks.
Evidence proves provenance, not that a natural-language interpretation is true.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import fields
import re
import sqlite3
import time
import unicodedata

from .binding import ALIASES
from .profiling import column_profile, quote
from .rules import RulePolicy
from .transforms import search_key, status_class, payment_class
from ..domain import DataMateError

KINDS = {'orders', 'sales', 'products', 'customers', 'payments', 'returns', 'inventory', 'conversion', 'other'}
STATUSES = {'completed', 'cancelled', 'returned', 'shipping'}
EXTRA_TYPES = {'text', 'number', 'date', 'boolean'}


def evidence_in(quote_text, source):
    if isinstance(quote_text,list):
        return 0<len(quote_text)<=8 and all(evidence_in(part,source) for part in quote_text)
    if not isinstance(quote_text, str) or not quote_text.strip():
        return False
    # Markdown layout is not part of the meaning of an otherwise literal quote.
    # Retain accents, wording and punctuation; do not do fuzzy semantic matching.
    def plain(text):
        text=unicodedata.normalize('NFC',text).replace('`','').replace('**','').replace('|',' ')
        return ' '.join(text.split())
    return bool(plain(quote_text)) and plain(quote_text) in plain(source)


def label_evidence(citation, label, source):
    """A header or a label alone does not document an ambiguous label's meaning."""
    parts=citation if isinstance(citation,list) else [citation]
    return evidence_in(citation,source) and any(isinstance(part,str) and part.strip()!=label
        and re.search(r'(?<!\w)'+re.escape(label)+r'(?!\w)',part) for part in parts)


def merge_policy(proposal, rulebook, baseline):
    """Accept independent, valid fields; never erase a healthy baseline on error."""
    accepted, rejected = dict(baseline), {}
    if not isinstance(proposal, dict):
        return accepted, {'policy': 'policy cần object'}
    for name, item in proposal.items():
        try:
            checked = validate_policy({name: item}, rulebook)
            if name not in checked['unknown']:
                accepted[name] = checked[name]
        except (DataMateError, TypeError, ValueError) as exc:
            rejected[name] = str(exc)
    # The baseline's explanatory warnings are retained for audit; dependencies
    # are checked against actual field values, not the length of this list.
    return accepted, rejected


def survey_catalog(catalog, deadline=None):
    """All-row statistics, without assigning business meanings to columns."""
    members = {n for t in catalog.tables for n in t.get('view_of', [])}
    connection = sqlite3.connect(catalog.database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.set_progress_handler(lambda: int(deadline is not None and time.monotonic() > deadline), 1000)
    tables = []
    try:
        for table in catalog.tables:
            if table['name'] == 'calendar' or table['name'] in members:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                raise DataMateError('Hết thời gian khảo sát schema')
            profiles = {}
            for column in table['columns']:
                if deadline is not None and time.monotonic() >= deadline:
                    raise DataMateError('Hết thời gian khảo sát cột')
                values = [r[0] for r in connection.execute(f'SELECT {quote(column)} FROM {quote(table["name"])}')]
                counts = Counter(v for v in values if v is not None and str(v).strip())
                profile = column_profile(values)
                profile.update(distinct_count=len(counts), unique_nonempty=bool(counts) and max(counts.values()) == 1,
                               values_complete=len(counts) <= 16,
                               values=[{'value': v, 'count': n} for v, n in sorted(counts.items(), key=lambda x: (x[1], str(x[0])))[:16 if len(counts)<=16 else 4]],
                               next_offset=None if len(counts)<=16 else 4)
                if values and all(v is None or type(v) in (int, float) or
                                  isinstance(v, str) and re.fullmatch(r'[\s\d.,+()\-đ₫a-zA-Z]+', v) for v in values):
                    numeric = column_profile(values, 'number')
                    profile['numeric_candidate'] = {k: numeric[k] for k in ('locale', 'parsed', 'parse_errors', 'min', 'max') if k in numeric}
                date_like = any(isinstance(v, str) and re.search(r'\d[/-]\d|\d\s+thg\s+\d', v) for v in counts)
                if date_like:
                    dates = column_profile(values, 'date')
                    profile['date_candidate'] = {k: dates[k] for k in ('parsed', 'parse_errors', 'min', 'max') if k in dates}
                profiles[column] = profile
            tables.append({**table, 'columns': profiles})
    finally:
        connection.close()
    return {'scope': 'all_rows', 'tables': tables}


def validate_policy(proposal, rulebook):
    """Only named operations already implemented by the executor are accepted."""
    if not isinstance(proposal, dict):
        raise DataMateError('policy cần object các trường {value,evidence}')
    names = {f.name for f in fields(RulePolicy)} - {'unknown'}
    unknown_keys = set(proposal) - names
    if unknown_keys:
        raise DataMateError('Quy tắc chưa có phép thực thi: ' + str(sorted(unknown_keys)))
    defaults = dict(sales_statuses=None, exclude_whole_returned_order=False, deduplicate_order_rows=False,
                    fill_order_fields=False, week_start=None, line_revenue=False, unknown=[])
    policy = RulePolicy(**defaults).to_dict()
    boolean_names = {k for k, v in policy.items() if type(v) is bool}
    unresolved = []
    for name in names:
        item = proposal.get(name)
        if item is None or isinstance(item, dict) and item.get('value') is None:
            unresolved.append(name)
            continue
        if not isinstance(item, dict) or not evidence_in(item.get('evidence'), rulebook):
            raise DataMateError('Quy tắc ' + name + ' cần trích nguyên văn rulebook hiện tại')
        value = item.get('value')
        if name in boolean_names and type(value) is not bool:
            raise DataMateError(name + ' cần boolean hoặc null')
        if name == 'sales_statuses' and (not isinstance(value, list) or not value or len(value) != len(set(value)) or any(s not in STATUSES for s in value)):
            raise DataMateError('sales_statuses cần list trạng thái canonical không trùng')
        if name == 'week_start' and (type(value) is not int or not 0 <= value <= 6):
            raise DataMateError('week_start trong 0..6 (thứ Hai=0)')
        if name == 'percent_places' and (type(value) is not int or not 0 <= value <= 6):
            raise DataMateError('percent_places trong 0..6')
        if name == 'shifts':
            if not isinstance(value, dict) or set(value) != {'morning', 'afternoon'}:
                raise DataMateError('shifts cần morning và afternoon với khoảng [đầu,cuối), ISO HH:MM')
            for interval in value.values():
                if not isinstance(interval, list) or len(interval) != 2 or any(not isinstance(v, str) or not re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d|24:00', v) for v in interval) or interval[0] >= interval[1]:
                    raise DataMateError('Khung giờ không hợp lệ hoặc qua nửa đêm chưa hỗ trợ')
            a, b = value.values()
            if max(a[0], b[0]) < min(a[1], b[1]):
                raise DataMateError('Hai ca chồng lấn')
        policy[name] = value
    policy['unknown'] = unresolved
    return policy


def validate_binding(proposal, table, survey, readme, catalog, deadline=None):
    if not isinstance(proposal, dict) or proposal.get('kind') not in KINDS:
        raise DataMateError('kind không hợp lệ')
    columns = table['columns']
    roles, citations = proposal.get('roles', {}), proposal.get('evidence', {})
    if not isinstance(roles, dict) or not isinstance(citations, dict):
        raise DataMateError('roles/evidence cần object')
    if set(roles) - set(ALIASES):
        raise DataMateError('Vai trò mới cần đặt trong attributes để giữ nguyên và truy vấn SQL: ' + str(sorted(set(roles) - set(ALIASES))))
    observed = survey['columns']
    for role, column in roles.items():
        if column not in columns:
            raise DataMateError('Cột không tồn tại: ' + str(column))
        text = citations.get(role)
        # Header citations are explicitly distinguished from README evidence.
        if not evidence_in(text, readme) and text != column:
            raise DataMateError('Ánh xạ ' + role + ' cần căn cứ README hoặc đúng tên header nguồn')
    for name, item in proposal.get('attributes', {}).items():
        if not isinstance(name, str) or not isinstance(item, dict) or item.get('column') not in columns or item.get('type', 'text') not in EXTRA_TYPES:
            raise DataMateError('attributes cần tên, column nguồn, type text/number/date/boolean')
        if not evidence_in(item.get('evidence'), readme) and item.get('evidence') != item['column']:
            raise DataMateError('Thuộc tính ' + name + ' chưa có căn cứ')
    status_map = proposal.get('status_map', {})
    if not isinstance(status_map, dict) or any(v not in STATUSES | {'unknown'} for v in status_map.values()):
        raise DataMateError('status_map không hợp lệ')
    if status_map and 'status' not in roles:
        raise DataMateError('status_map cần cột status')
    if 'status' in roles:
        connection = sqlite3.connect(catalog.database.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            actual_values = {str(r[0]) for r in connection.execute(f'SELECT DISTINCT {quote(roles["status"])} FROM {quote(table["name"])}') if r[0] is not None and str(r[0]).strip()}
        finally:
            connection.close()
        if set(status_map) - actual_values:
            raise DataMateError('status_map có giá trị không nằm trong trang profile đã đọc')
        for label, state in status_map.items():
            # A status needs a documented meaning, not just its own spelling.
            citation = proposal.get('status_evidence', {}).get(label)
            literal = status_class(label) == state and citation == label
            if state != 'unknown' and not literal and not label_evidence(citation,label,readme):
                raise DataMateError('Trạng thái ' + label + ' cần README giải nghĩa; thiếu căn cứ giữ unknown')
    payment_map = proposal.get('payment_map', {})
    if not isinstance(payment_map, dict) or any(v not in {'debt','cod','cash','transfer','ewallet','marketplace','unknown'} for v in payment_map.values()):
        raise DataMateError('payment_map không hợp lệ')
    if payment_map and 'payment_method' not in roles:
        raise DataMateError('payment_map cần payment_method')
    for label, method in payment_map.items():
        citation = proposal.get('payment_evidence', {}).get(label)
        if method != 'unknown' and not (payment_class(label)==method and citation==label) and not label_evidence(citation,label,readme):
            raise DataMateError('Phương thức thanh toán cần căn cứ nghĩa từ README hoặc nhãn rõ nghĩa')
    required = {'orders': {'order_id', 'order_date', 'status'}, 'sales': {'line_amount'},
                'products': {'product_id', 'product_name'}, 'customers': {'customer_id', 'customer_name'},
                'payments': {'order_id', 'payment_date', 'payment_amount'},
                'returns': {'order_id', 'return_quantity'}, 'inventory': {'stock'}, 'conversion': {'factor'}}
    if not required.get(proposal['kind'], set()) <= set(roles):
        raise DataMateError('Thiếu vai trò bắt buộc cho ' + proposal['kind'] + ': ' + str(sorted(required[proposal['kind']] - set(roles))))
    # Identical values/unique keys only establish technical join consistency.
    checks = []
    connection = sqlite3.connect(catalog.database.resolve().as_uri() + '?mode=ro', uri=True)
    connection.set_progress_handler(lambda: int(deadline is not None and time.monotonic() > deadline), 1000)
    try:
        from .normalization import NUMERIC, DATES
        identity = {'orders':'order_id','products':'product_id','customers':'customer_id'}.get(proposal['kind'])
        if identity:
            key_column=roles[identity]
            columns_sql=','.join(quote(c) for c in columns)
            conflicts=connection.execute(f'SELECT COUNT(*) FROM (SELECT {quote(key_column)} FROM (SELECT DISTINCT {columns_sql} FROM {quote(table["name"])}) WHERE {quote(key_column)} IS NOT NULL GROUP BY {quote(key_column)} HAVING COUNT(*)>1)').fetchone()[0]
            if conflicts:
                raise DataMateError('Khóa '+key_column+' có nhiều dòng khác nhau; cấp dữ liệu không phù hợp '+proposal['kind'])
            checks.append({'identity':key_column,'scope':'all_rows','distinct_records_unique':True})
        for role, column in roles.items():
            if role not in NUMERIC | DATES:
                continue
            quantity_role=role in {'quantity','base_quantity','return_quantity'}
            unit_role='base_unit' if role=='base_quantity' and 'base_unit' in roles else 'unit'
            unit_column=roles.get(unit_role)
            unit_sql=quote(unit_column) if unit_column else 'NULL'
            records = connection.execute(f'SELECT {quote(column)},_dm_sheet,{unit_sql} FROM {quote(table["name"])}').fetchall()
            values = [r[0] for r in records]
            years=[]
            for _,sheet,_ in records:
                match=re.search(r'\b((?:19|20|21)\d{2})\b',str(sheet).replace('_',' '))
                years.append(int(match[1]) if match else None)
            profile = column_profile(values, 'quantity' if quantity_role else 'number' if role in NUMERIC else 'date', readme,
                                     years=years, units=[r[2] for r in records] if quantity_role else None,
                                     excel_epoch=table.get('excel_epoch'))
            checks.append({'role': role, 'column': column, 'parsed': profile['parsed'], 'parse_errors': profile['parse_errors']})
            # Some exports contain blanks/group headers. Errors are retained in
            # row-level quality checks; accepting a mapping never accepts a sum.
            if profile['nonempty'] and not profile['parsed']:
                raise DataMateError('Cột ' + column + ' không có giá trị nào đọc được theo kiểu ' + role)
        joins = proposal.get('joins', [])
        if not isinstance(joins, list):
            raise DataMateError('joins cần list')
        known = {t['name']: t for t in catalog.tables}
        for join in joins:
            target = known.get(join.get('target_table'))
            local, remote = join.get('column'), join.get('target_column')
            if not target or local not in columns or remote not in target['columns']:
                raise DataMateError('Cột/bảng trong joins không tồn tại')
            if not evidence_in(join.get('evidence'), readme) and join.get('evidence') != local:
                raise DataMateError('joins thiếu căn cứ')
            a = [str(r[0]) for r in connection.execute(f'SELECT {quote(local)} FROM {quote(table["name"])}') if r[0] is not None and str(r[0]).strip()]
            b = [str(r[0]) for r in connection.execute(f'SELECT {quote(remote)} FROM {quote(target["name"])}') if r[0] is not None and str(r[0]).strip()]
            if len(b) != len(set(b)):
                raise DataMateError('Khóa đích không duy nhất; nối bảng sẽ nhân bản giao dịch')
            orphan = len(set(a) - set(b))
            if orphan:
                raise DataMateError(f'Có {orphan} khóa nối không tìm thấy trong bảng đích; cần kiểm tra lại joins')
            checks.append({'join': join, 'scope': 'all_rows', 'target_unique': True, 'orphan_keys': orphan})
    finally:
        connection.close()
    return {'table': table['name'], 'kind': proposal['kind'], 'roles': roles, 'ambiguous': {},
            'origins': {c: 'LLM_HEADER' if citations.get(r)==c else 'LLM_README' for r, c in roles.items()},
            'evidence': citations, 'status_map': status_map, 'status_evidence': proposal.get('status_evidence', {}),
            'payment_map':payment_map, 'payment_evidence':proposal.get('payment_evidence',{}),
            'attributes': proposal.get('attributes', {}), 'joins': proposal.get('joins', []), 'validation': checks,
            'unmapped': [c for c in columns if c not in roles.values() and c not in {a['column'] for a in proposal.get('attributes',{}).values()}]}


def merge_binding(patch, baseline, table, readme, catalog, deadline=None):
    """Validate optional additions independently, retaining the baseline on error.

    Healthy roles cannot be overwritten by an on-demand patch. The full model
    interpretation mode keeps its stricter, all-or-nothing validation API.
    """
    rejected = {}
    accepted = deepcopy(baseline)
    proposal = {k: deepcopy(baseline.get(k, {} if k != 'joins' else []))
                for k in ('roles','evidence','attributes','status_map','status_evidence','payment_map','payment_evidence','joins')}
    proposal['evidence'] = {role: proposal['evidence'].get(role, column) for role,column in proposal['roles'].items()}
    proposal['kind'] = baseline['kind']
    if not isinstance(patch, dict):
        return accepted, {'table': 'Bản bổ sung cần object'}

    def attempt(label, update):
        nonlocal proposal, accepted
        candidate=deepcopy(proposal)
        update(candidate)
        try:
            checked=validate_binding(candidate,table,{'columns':{}},readme,catalog,deadline)
        except (DataMateError, KeyError, TypeError, ValueError) as exc:
            rejected[label]=str(exc)
            return
        proposal=candidate
        old_origins=accepted.get('origins',{})
        accepted={**accepted,**checked,'origins':{**checked['origins'],**old_origins}}
        accepted['ambiguous']={r:v for r,v in baseline.get('ambiguous',{}).items() if r not in accepted['roles']}

    roles=patch.get('roles',{})
    if not isinstance(roles,dict):
        rejected['roles']='roles cần object'
        roles={}
    for role,column in roles.items():
        if role in baseline['roles'] and baseline['roles'][role]!=column:
            rejected['roles.'+role]='Không thay vai trò baseline đang rõ; cần kiểm tra xung đột riêng'
            continue
        def add_role(candidate,role=role,column=column):
            candidate['roles'][role]=column
            candidate['evidence'][role]=patch.get('evidence',{}).get(role)
            # Retain the baseline grain while adding roles. A previously
            # unclassified table stays other until its requested kind passes
            # required-role and key checks at the end.
        attempt('roles.'+role,add_role)
    for name,item in patch.get('attributes',{}).items() if isinstance(patch.get('attributes',{}),dict) else []:
        attempt('attributes.'+name,lambda candidate,n=name,v=item: candidate['attributes'].update({n:v}))
    for field,evidence_field in (('status_map','status_evidence'),('payment_map','payment_evidence')):
        mapping=patch.get(field,{})
        if not isinstance(mapping,dict):
            rejected[field]='Cần object'
            continue
        for label,value in mapping.items():
            known = status_class(label) if field=='status_map' else payment_class(label)
            if known not in {None,'unknown'} and known != value:
                rejected[field+'.'+label]='Không thay nghĩa nhãn đã rõ'
                continue
            def add_value(candidate,label=label,value=value,field=field,evidence_field=evidence_field):
                candidate[field][label]=value
                candidate[evidence_field][label]=patch.get(evidence_field,{}).get(label)
            attempt(field+'.'+label,add_value)
    kind=patch.get('kind',baseline['kind'])
    if accepted['kind']!=kind:
        attempt('kind',lambda candidate: candidate.update(kind=kind))
    # If a requested grain failed its required-role/unique-key checks, valid
    # additions stay visible as other and cannot become a transaction total.
    accepted['unmapped']=[c for c in table['columns'] if c not in accepted['roles'].values()
                          and c not in {a['column'] for a in accepted.get('attributes',{}).values()}]
    return accepted,rejected
