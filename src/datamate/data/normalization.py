"""Cached typed data and business facts, built without questions or answers.

Raw tables remain unchanged. All decisions depend on the current split's
README/rulebook/schema/values, never on a business or question identifier.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import uuid
import threading
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .binding import bind_table, resolve_order_relationships
from .profiling import column_profile, parse_profile_number, quote
from .rules import read_policy, RulePolicy
from .dates import date_epoch
from .dimensions import dimensions,resolve_dimension
from .transforms import iso_date, typed_date, normalized, payment_class, search_key, status_class, quantity_coefficient

NUMERIC = {"line_amount", "header_revenue", "order_total", "balance", "paid", "quantity", "base_quantity",
           "stock", "price", "cost", "factor", "payment_amount", "return_quantity"}
DATES = {"order_date", "purchase_date", "payment_date", "expiry_date", "stock_timestamp"}
ORDER_FIELDS = {"order_date", "status", "customer_id", "customer_name", "timestamp", "hour", "channel", "payment_method", "employee","customer_group","gender"}
FACT_COLUMNS = {
    "dm_orders_all": ["order_id", "order_date", "order_timestamp", "status", "customer_id", "customer_name", "customer_key",
                      "payment_method", "channel", "employee", "order_total", "header_revenue", "paid", "balance",
                      "cancel_reason", "customer_group", "gender", "_dm_issues", "_dm_source_table", "_dm_file", "_dm_sheet", "_dm_row"],
    "dm_sales_all": ["order_id", "order_date", "order_timestamp", "status", "customer_id", "customer_name", "customer_key",
                     "product_id", "product_name", "product_key", "unit", "base_unit", "quantity", "base_quantity", "line_amount", "price", "cost",
                     "payment_method", "channel", "employee", "product_group", "customer_group", "gender", "prescription", "_dm_issues", "_dm_source_table", "_dm_file", "_dm_sheet", "_dm_row"],
    "dm_products": ["product_id", "product_name", "base_unit", "price", "cost", "product_group", "prescription", "_dm_issues", "_dm_source_table", "_dm_file", "_dm_sheet", "_dm_row"],
    "dm_customers": ["customer_id","customer_name","customer_group","gender","_dm_issues","_dm_source_table","_dm_file","_dm_sheet","_dm_row"],
    "dm_stock_ledger": ["product_id","product_name","quantity","occurred_at","location_id","_dm_issues","_dm_source_table","_dm_file","_dm_sheet","_dm_row"],
    "dm_inventory": ["product_id", "product_name", "unit", "quantity", "snapshot_date", "_dm_issues", "_dm_source_table", "_dm_file", "_dm_sheet", "_dm_row"],
    "dm_payments": ["order_id", "payment_date", "amount", "payment_method", "_dm_issues", "_dm_source_table", "_dm_file", "_dm_sheet", "_dm_row"],
    "dm_conversions": ["product_id", "product_name", "large_unit", "base_unit", "factor", "_dm_source_table", "_dm_file", "_dm_sheet", "_dm_row"],
}


def _source_version():
    digest = hashlib.sha256(b"datamate-semantic-v2")
    for filename in ("normalization.py", "binding.py", "profiling.py", "transforms.py", "rules.py",'ingest.py','dates.py','dimensions.py'):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    return digest.hexdigest()


_LOADED_VERSION = _source_version()


def semantic_version():
    return _LOADED_VERSION


def _provenance(row, table):
    return {"_dm_source_table": table, **{k: row.get(k) for k in ("_dm_file", "_dm_sheet", "_dm_row")}}


def _value(row, binding, role):
    return row.get(binding["roles"].get(role, ""))


def _nonempty(value):
    return value is not None and str(value).strip() != ""


def _subtotal(row,binding):
    roles=binding['roles']
    if _nonempty(_value(row,binding,'order_id')) or _nonempty(_value(row,binding,'order_date')) or _nonempty(_value(row,binding,'status')):
        return False
    marker=any(re.match(r'^(?:tong|cong|ghi chu|ngay in|nguoi lap|subtotal|grand total|total)\b',search_key(v) or '')
               for key,v in row.items() if not key.startswith('_dm_') and isinstance(v,str))
    return marker and (binding['kind']=='orders' or not _nonempty(_value(row,binding,'quantity')))


def _timestamp(value, day, hour=None):
    text = str(value or hour or "").strip()
    if not text:
        return None
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?\b", text)
    if not match or not day:
        return None
    h, m, s = int(match[1]), int(match[2]), int(match[3] or 0)
    return f"{day}T{h:02}:{m:02}:{s:02}" if h<24 and m<60 and s<60 else None


def _write_table(connection, name, columns, rows):
    connection.execute(f"CREATE TABLE {quote(name)} ({','.join(quote(c) for c in columns)})")
    connection.executemany(f"INSERT INTO {quote(name)} VALUES ({','.join('?' for _ in columns)})",
                           ([r.get(c) for c in columns] for r in rows))


def _cutoff(readme):
    match = re.search(r"dữ liệu có đến hết\s+(\d{1,2}/\d{1,2}/\d{4})", readme, re.I)
    return iso_date(match[1]) if match else None


def _readme_conversions(readme):
    rows = []
    for name, large, factor, base in re.findall(r"(?m)^\s*-\s+(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$", readme):
        try:
            value = parse_profile_number(factor, "auto")
        except ValueError:
            continue
        rows.append({"product_name": name, "large_unit": large, "factor": value, "base_unit": base,
                     "_dm_source_table": "README", "_dm_file": "README.md"})
    return rows


def build_semantics(connection, tables, rulebook, readme, interpretation=None):
    policy = RulePolicy(**interpretation['policy']) if interpretation else read_policy(rulebook)
    members = {name for table in tables for name in table.get("view_of", [])}
    preferred = [t for t in tables if t["name"] not in members and t["name"] != "calendar"]
    if interpretation:
        proposed = {b['table']: b for b in interpretation['bindings']}
        # Unresolved tables stay visible and raw-queryable. Header candidates
        # never silently replace a failed model interpretation.
        bindings = [json.loads(json.dumps(proposed[t['name']])) if t['name'] in proposed else
                    {'table':t['name'], 'kind':'other', 'roles':{}, 'ambiguous':{}, 'origins':{},
                     'unmapped':t['columns'], 'unresolved':True} for t in preferred]
    else:
        bindings = [bind_table(t, readme) for t in preferred]
        resolve_order_relationships(connection,bindings,preferred)
    binding_by_name = {b["table"]: b for b in bindings}
    overlay = bool(interpretation and interpretation.get('mode') == 'code_first')
    def state_of(row, binding):
        value = _value(row, binding, 'status')
        fallback = status_class(value) if not interpretation or overlay else 'unknown' if _nonempty(value) else None
        return binding.get('status_map', {}).get(str(value), fallback)
    def payment_of(row, binding):
        value = _value(row,binding,'payment_method')
        fallback = payment_class(value) if not interpretation or overlay else 'unknown' if _nonempty(value) else None
        return binding.get('payment_map',{}).get(str(value), fallback)
    profiles, clean, dedup, excluded = {}, {}, [], []
    for table in preferred:
        name = table["name"]
        binding = binding_by_name[name]
        role_by_column = {col: role for role, col in binding["roles"].items()}
        extra_types = {a['column']:a.get('type','text') for a in binding.get('attributes',{}).values()}
        cursor = connection.execute(f"SELECT * FROM {quote(name)}")
        columns = [x[0] for x in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor]
        epoch,epoch_evidence=date_epoch(table,rows,binding,_cutoff(readme))
        if epoch_evidence:binding['date_epoch_evidence']=epoch_evidence
        years = []
        for row in rows:
            match = re.search(r"\b((?:19|20|21)\d{2})\b", str(row.get("_dm_sheet", "")).replace("_", " "))
            years.append(int(match[1]) if match else None)
        profiles[name] = {}
        for column in table["columns"]:
            role = role_by_column.get(column)
            kind = 'quantity' if role in {'quantity','base_quantity','return_quantity'} else "number" if role in NUMERIC else "date" if role in DATES else role if role in {'status','payment_method'} else None
            if kind is None and extra_types.get(column) in {'number','date'}:
                kind = extra_types[column]
            profile = column_profile([row[column] for row in rows], kind, readme, years,
                                     [_value(row,binding,'base_unit' if role=='base_quantity' and 'base_unit' in binding['roles'] else 'unit') for row in rows],epoch)
            profile["semantic_role"] = role
            profiles[name][column] = profile
        seen, typed_rows, previous_order = {}, [], {}
        for index, raw in enumerate(rows):
            repeated_header=sum(isinstance(raw[c],str) and search_key(raw[c])==search_key(c) for c in table['columns'])
            if repeated_header>=max(3,len(table['columns'])//2):
                excluded.append({**_provenance(raw,name),'reason':'repeated_header_not_transaction'})
                previous_order.pop((raw.get('_dm_file'),raw.get('_dm_sheet')),None)
                continue
            if binding['kind'] in {'sales','orders'} and _subtotal(raw,binding):
                excluded.append({**_provenance(raw,name),'reason':'summary_row_not_transaction'})
                previous_order.pop((raw.get('_dm_file'),raw.get('_dm_sheet')),None)
                continue
            oid = _value(raw, binding, "order_id")
            inherited_order = None
            source = (raw.get("_dm_file"),raw.get("_dm_sheet"))
            last = previous_order.get(source)
            if (not _nonempty(oid) and last and policy.fill_order_fields and
                    binding["kind"] == 'sales' and "order_id" in binding["roles"] and
                    not _nonempty(_value(raw,binding,'order_date')) and not _nonempty(_value(raw,binding,'status')) and
                    _nonempty(_value(raw,binding,'quantity')) and
                    raw["_dm_row"] == last[1]+1):
                oid, inherited_order = last[0], last[0]
            if _nonempty(oid):
                previous_order[source] = (oid, raw["_dm_row"])
            if policy.deduplicate_order_rows and _nonempty(oid) and binding["kind"] in {"sales", "orders"}:
                signature = (str(oid), tuple((c, raw[c]) for c in table["columns"]))
                if signature in seen:
                    dedup.append({**_provenance(raw, name), "kept_row": seen[signature], "order_id": str(oid)})
                    continue
                seen[signature] = raw["_dm_row"]
            row, issues = dict(raw), []
            if inherited_order is not None:
                row[binding["roles"]["order_id"]] = inherited_order
            for role, column in binding["roles"].items():
                value = raw[column]
                if not _nonempty(value):
                    row[column] = None
                    continue
                try:
                    if role in NUMERIC:
                        value=quantity_coefficient(value,_value(raw,binding,'base_unit' if role=='base_quantity' and 'base_unit' in binding['roles'] else 'unit')) if role in {'quantity','base_quantity','return_quantity'} else value
                        row[column] = parse_profile_number(value, profiles[name][column]["locale"])
                    elif role in DATES:
                        row[column] = typed_date(value, years[index],epoch)
                    elif role == 'prescription':
                        key=search_key(str(value))
                        if key not in {'1','0','true','false','co','khong'}:
                            raise ValueError('Boolean chưa rõ')
                        row[column]=int(key in {'1','true','co'})
                except (ValueError, TypeError, ArithmeticError):
                    row[column] = None
                    issues.append("parse:"+column)
            for column, value_type in extra_types.items():
                if column in role_by_column or not _nonempty(raw[column]) or value_type=='text':
                    continue
                try:
                    if value_type=='number':
                        row[column]=parse_profile_number(raw[column],profiles[name][column]['locale'])
                    elif value_type=='date':
                        row[column]=typed_date(raw[column],years[index],table.get('excel_epoch'))
                    elif value_type=='boolean':
                        key=search_key(str(raw[column]))
                        if key not in {'1','0','true','false','co','khong'}:
                            raise ValueError('Boolean chưa rõ')
                        row[column]=int(key in {'1','true','co'})
                except (ValueError,TypeError,ArithmeticError):
                    row[column]=None
                    issues.append('parse:'+column)
            row["_dm_source_table"], row["_dm_issues"] = name, issues
            if 'stock_timestamp' in binding['roles']:
                column=binding['roles']['stock_timestamp']
                row['_dm_stock_timestamp']=_timestamp(raw[column],row.get(column))
            if inherited_order is not None:
                row[binding["roles"]["order_id"]] = inherited_order
            typed_rows.append(row)
        clean[name] = typed_rows

    # Forward-fill only order fields with a real order key, and only when the
    # rulebook explicitly defines blank group cells as inheritance.
    grouped = defaultdict(list)
    for binding in bindings:
        if binding["kind"] not in {"sales", "orders"} or "order_id" not in binding["roles"]:
            continue
        for row in clean[binding["table"]]:
            oid = _value(row, binding, "order_id")
            if _nonempty(oid):
                grouped[str(oid)].append((row, binding))
    if policy.fill_order_fields:
        for group in grouped.values():
            for role in ORDER_FIELDS:
                values = [_value(r,b,role) for r,b in group if _nonempty(_value(r,b,role))]
                keys = {state_of(r,b) if role == 'status' else normalized(_value(r,b,role)) if isinstance(_value(r,b,role),str) else _value(r,b,role)
                        for r,b in group if _nonempty(_value(r,b,role))}
                if len(keys)>1:
                    for row,_ in group:
                        row["_dm_issues"].append("conflicting_order:"+role)
                if values:
                    for row,binding in group:
                        col = binding["roles"].get(role)
                        if col and not _nonempty(row.get(col)):
                            row[col] = values[0]

    facts = {name: [] for name in FACT_COLUMNS}
    products=dimensions(bindings,clean,'product');customers=dimensions(bindings,clean,'customer')
    facts['dm_products']=products[0];facts['dm_customers']=customers[0]
    product_names={key:value['product_name'] for key,value in products[1].items() if value.get('product_name')}
    customer_names={key:value['customer_name'] for key,value in customers[1].items() if value.get('customer_name')}

    conversions = []
    for binding in bindings:
        if binding["kind"] != "conversion":
            continue
        for row in clean[binding["table"]]:
            pid = _value(row,binding,"product_id")
            conversions.append({"product_id": pid, "product_name": _value(row,binding,"product_name") or product_names.get(str(pid)),
                "large_unit": _value(row,binding,"large_unit") or _value(row,binding,"unit"),
                "base_unit": _value(row,binding,"base_unit"), "factor": _value(row,binding,"factor"), **_provenance(row,binding["table"])})
    if not conversions:
        conversions = _readme_conversions(readme)
    # Both sides of a unit conversion must use the same observed identity.
    # A catalog may supply IDs while the conversion file only supplies names.
    for conv in conversions:
        product,ambiguous=resolve_dimension(conv.get('product_id'),conv.get('product_name'),products)
        if product and not ambiguous:
            if not _nonempty(conv.get('product_id')):conv['product_id']=product.get('product_id')
            conv['product_name']=product.get('product_name') or conv.get('product_name')
    facts["dm_conversions"] = conversions
    conversion_index = defaultdict(list)
    base_units = {str(r["product_id"]): r["base_unit"] for r in facts["dm_products"] if _nonempty(r.get("product_id"))}
    for conv in conversions:
        key = str(conv["product_id"]) if _nonempty(conv.get("product_id")) else search_key(conv.get("product_name"))
        conversion_index[(key, search_key(conv.get("large_unit")))].append(conv.get("factor"))
        if conv.get("base_unit"):
            base_units[key] = conv["base_unit"]

    returned_orders = set()
    for binding in bindings:
        if binding["kind"] == "returns":
            for row in clean[binding["table"]]:
                value = _value(row,binding,"return_quantity")
                if value is not None and value>0:
                    returned_orders.add(str(_value(row,binding,"order_id")))

    sales_bindings = [b for b in bindings if b["kind"] == "sales"]
    order_bindings = [b for b in bindings if b["kind"] == "orders"]
    coded_customers = any("customer_id" in b["roles"] for b in order_bindings+sales_bindings)
    customer_identity = coded_customers or any("customer_name" in b["roles"] for b in order_bindings+sales_bindings)

    def customer(cid, name):
        if coded_customers:
            return str(cid) if _nonempty(cid) else None
        if not _nonempty(name) or search_key(name) in {"khach le", "khach vang lai", "vang lai", "le", "kl"}:
            return None
        return normalized(name)

    order_map = {}
    for oid, group in grouped.items():
        header_group = [(r,b) for r,b in group if b["kind"] == "orders"] or group
        first, first_binding = header_group[0]
        def get(role):
            return next((_value(r,b,role) for r,b in header_group if _nonempty(_value(r,b,role))), None)
        cid, cname = get("customer_id"), get("customer_name")
        cname = customer_names.get(str(cid), cname)
        day = get("order_date")
        statuses = {state_of(r,b) for r,b in header_group if _nonempty(_value(r,b,"status"))}
        state = next(iter(statuses)) if len(statuses)==1 else "unknown"
        if policy.exclude_whole_returned_order and 'returned' in statuses:
            state='returned'
        issues = sorted({issue for r,_ in header_group for issue in r["_dm_issues"]})
        customer_dimension,ambiguous=resolve_dimension(cid,cname,customers)
        if ambiguous:issues.append('ambiguous_customer_dimension')
        customer_dimension=customer_dimension or {}
        issues.extend(issue for issue in json.loads(customer_dimension.get('_dm_issues','[]')) if any(k in issue for k in ('customer_group','gender')))
        if state is None or state == "unknown":
            state = "unknown"
            issues.append("unknown_order_status")
        if policy.exclude_whole_returned_order and oid in returned_orders:
            state = "returned"
        order_map[oid] = {"order_id": oid, "order_date": day, "order_timestamp": _timestamp(get("timestamp"),day,get("hour")),
            "status": state, "customer_id": cid, "customer_name": cname, "customer_key": customer(cid,cname),
            "payment_method": next((payment_of(r,b) for r,b in header_group if _nonempty(_value(r,b,'payment_method'))),None), "channel": get("channel"), "employee": get("employee"),
            "order_total": get("order_total"), "header_revenue": get("header_revenue"), "paid": get("paid"),
            "balance": get("balance"), "cancel_reason": get("cancel_reason"),
            "customer_group":get('customer_group') or customer_dimension.get('customer_group'),"gender":get('gender') or customer_dimension.get('gender'),
            "_dm_issues": json.dumps(issues,ensure_ascii=False),
            **_provenance(first,first_binding["table"])}
    facts["dm_orders_all"] = list(order_map.values())

    for binding in sales_bindings:
        for row in clean[binding["table"]]:
            oid = _value(row,binding,"order_id")
            order = order_map.get(str(oid), {})
            day = order.get("order_date") or _value(row,binding,"order_date")
            state = order.get("status") or state_of(row,binding) or "unknown"
            cid = order.get("customer_id", _value(row,binding,"customer_id"))
            cname = order.get("customer_name", _value(row,binding,"customer_name"))
            pid = _value(row,binding,"product_id")
            pname = product_names.get(str(pid), _value(row,binding,"product_name"))
            product_dimension,ambiguous=resolve_dimension(pid,pname,products)
            product_dimension=product_dimension or {}
            pid=pid if _nonempty(pid) else product_dimension.get('product_id')
            pname=product_dimension.get('product_name') or pname
            unit, qty, base = (_value(row,binding,r) for r in ("unit", "quantity", "base_quantity"))
            product_lookup = str(pid) if _nonempty(pid) else search_key(pname)
            issues = list(row["_dm_issues"])+json.loads(order.get("_dm_issues","[]"))
            if ambiguous:issues.append('ambiguous_product_dimension')
            issues.extend(issue for issue in json.loads(product_dimension.get('_dm_issues','[]')) if any(k in issue for k in ('product_group','prescription','base_unit')))
            if base is None and qty is not None:
                key = str(pid) if _nonempty(pid) else search_key(pname)
                factors = set(conversion_index.get((key, search_key(unit)), []))
                if len(factors)>1 or None in factors:
                    issues.append("ambiguous_conversion")
                else:
                    base = qty*(next(iter(factors)) if factors else 1)
            if state == "unknown":
                issues.append("unknown_order_status")
            item = {"order_id": str(oid) if _nonempty(oid) else None, "order_date": day,
                "order_timestamp": order.get("order_timestamp") or _timestamp(_value(row,binding,"timestamp"),day,_value(row,binding,"hour")),
                "status": state, "customer_id": cid, "customer_name": cname, "customer_key": order.get("customer_key",customer(cid,cname)),
                "product_id": pid, "product_name": pname, "product_key": str(pid) if _nonempty(pid) else normalized(pname),
                "unit": unit, "base_unit": base_units.get(product_lookup,unit), "quantity": qty, "base_quantity": base, "line_amount": _value(row,binding,"line_amount"),
                "price": _value(row,binding,"price"), "cost": _value(row,binding,"cost"), "payment_method": order.get("payment_method") or payment_of(row,binding),
                "channel": order.get("channel",_value(row,binding,"channel")), "employee": order.get("employee",_value(row,binding,"employee")),
                "product_group":_value(row,binding,'product_group') or product_dimension.get('product_group'),
                "prescription":_value(row,binding,'prescription') if _value(row,binding,'prescription') is not None else product_dimension.get('prescription'),
                "customer_group":order.get('customer_group'),"gender":order.get('gender'),
                "_dm_issues": json.dumps(sorted(set(issues)),ensure_ascii=False), **_provenance(row,binding["table"])}
            facts["dm_sales_all"].append(item)
    if not sales_bindings:
        # Invoice-only datasets may explicitly supply the sum of line amounts.
        # Never substitute the amount payable after order-level adjustments.
        for order in facts["dm_orders_all"]:
            if order["header_revenue"] is not None:
                facts["dm_sales_all"].append({**order, "line_amount": order["header_revenue"]})

    for binding in bindings:
        if binding["kind"] == "inventory":
            for row in clean[binding["table"]]:
                pid = _value(row,binding,"product_id")
                if 'stock_timestamp' in binding['roles']:
                    occurred=row.get('_dm_stock_timestamp')
                    facts['dm_stock_ledger'].append({'product_id':pid,'product_name':product_names.get(str(pid),_value(row,binding,'product_name')),
                        'quantity':_value(row,binding,'stock'),'occurred_at':occurred,'location_id':_value(row,binding,'location_id'),
                        '_dm_issues':json.dumps(row['_dm_issues']),**_provenance(row,binding['table'])})
                    continue
                facts["dm_inventory"].append({"product_id": pid,
                    "product_name": product_names.get(str(pid),_value(row,binding,"product_name")),
                    "unit": _value(row,binding,"unit") or _value(row,binding,"base_unit"),
                    "quantity": _value(row,binding,"stock"), "snapshot_date": _cutoff(readme),
                    "_dm_issues": json.dumps(row["_dm_issues"],ensure_ascii=False), **_provenance(row,binding["table"])})
        if binding["kind"] == "payments":
            for row in clean[binding["table"]]:
                facts["dm_payments"].append({"order_id": str(_value(row,binding,"order_id")),
                    "payment_date": _value(row,binding,"payment_date"), "amount": _value(row,binding,"payment_amount"),
                    "payment_method": payment_of(row,binding),
                    "_dm_issues": json.dumps(row["_dm_issues"],ensure_ascii=False), **_provenance(row,binding["table"])})

    for binding in bindings:
        name = binding["table"]
        table = next(t for t in preferred if t["name"]==name)
        columns = ["_dm_row", "_dm_file", "_dm_sheet", *table["columns"], "_dm_source_table", "_dm_issues", "_dm_status"]
        records = []
        for row in clean[name]:
            records.append({**row, "_dm_issues": json.dumps(row["_dm_issues"],ensure_ascii=False),
                            "_dm_status": state_of(row,binding)})
        _write_table(connection, "clean_"+name.removeprefix("raw_"), columns, records)
        binding["clean_table"] = "clean_"+name.removeprefix("raw_")
    for name, columns in FACT_COLUMNS.items():
        _write_table(connection, name, columns, facts[name])
    allowed = ",".join("'"+v+"'" for v in policy.sales_statuses or []) or "NULL"
    for table in ("sales", "orders"):
        connection.execute(f"CREATE VIEW dm_{table} AS SELECT * FROM dm_{table}_all WHERE status IN ({allowed})")
        connection.execute(f"CREATE INDEX dm_{table}_date ON dm_{table}_all(order_date)")
        connection.execute(f"CREATE INDEX dm_{table}_order ON dm_{table}_all(order_id)")
    connection.execute("CREATE INDEX dm_payment_order ON dm_payments(order_id,payment_date)")
    _write_table(connection, "_dm_deduplicated_rows", ["order_id","_dm_source_table","_dm_file","_dm_sheet","_dm_row","kept_row"], dedup)
    _write_table(connection,'_dm_excluded_rows',['_dm_source_table','_dm_file','_dm_sheet','_dm_row','reason'],excluded)
    status_profiles = {b["table"]: {"column": b["roles"]["status"], **profiles[b["table"]][b["roles"]["status"]]}
                       for b in sales_bindings+order_bindings if "status" in b["roles"]}
    for binding in bindings:
        for item in status_profiles.get(binding['table'], {}).get('values', []):
            if interpretation:
                item['canonical'] = binding.get('status_map', {}).get(str(item['value']), status_class(item['value']) if overlay else 'unknown')
    dates = [r["order_date"] for r in facts["dm_sales_all"] if r.get("order_date")]
    if not dates:
        dates = [r["order_date"] for r in facts["dm_orders_all"] if r.get("order_date")]
    quality = {name: {"rows": len(rows), "issue_rows": sum(bool(json.loads(r.get("_dm_issues","[]"))) for r in rows)}
               for name, rows in facts.items()}
    domains={column:sorted({r[column] for rows in facts.values() for r in rows if column in r and r[column] is not None},key=str)
             for column in ('product_group','customer_group','gender','prescription','base_unit','payment_method','status')}
    return {"version": semantic_version(), "policy": policy.to_dict(), "bindings": bindings, "profiles": profiles,'domains':domains,
            "status_profiles": status_profiles, "tables": [{"name": n,"columns":c,"rows":len(facts[n])} for n,c in FACT_COLUMNS.items()]
                      +[{"name":"dm_sales","columns":FACT_COLUMNS["dm_sales_all"]},{"name":"dm_orders","columns":FACT_COLUMNS["dm_orders_all"]}],
            "quality": quality, "deduplicated_rows":len(dedup), 'excluded_summary_rows':len(excluded), "coverage": {"start":min(dates) if dates else None,
            "end":max(dates) if dates else None, "declared_end":_cutoff(readme)},
            "interpretation": {k:interpretation[k] for k in ('cache_key','unresolved','validation_scope') if k in interpretation} if interpretation else None,
            "capabilities": {"sales":bool(facts["dm_sales_all"]), "order_identity":bool(order_map), "customer_identity":customer_identity,
                "product_identity":any("product_name" in b["roles"] or "product_id" in b["roles"] for b in sales_bindings),
                "inventory_snapshot":bool(facts["dm_inventory"]), "payments":bool(facts["dm_payments"]),
                "order_total":any("order_total" in b["roles"] for b in order_bindings),
                "stock_ledger":bool(facts['dm_stock_ledger']),
                **{role:any(role in b['roles'] and profiles[b['table']][b['roles'][role]]['nonempty']>0 for b in bindings)
                   for role in ('product_group','customer_group','gender','prescription','return_reason')},
                "balance_snapshot":any("balance" in b["roles"] for b in order_bindings),
                "cancellation_reason":any("cancel_reason" in b["roles"] for b in sales_bindings+order_bindings),
                "channel":any("channel" in b["roles"] for b in sales_bindings+order_bindings)}}


_semantic_lock = threading.RLock()


def prepare_catalog(catalog, rulebook, readme, cache_dir, interpretation=None):
    # Different question workers may repair the same shop simultaneously.
    # Publish one complete SQLite+metadata pair, never replace a DB in use.
    with _semantic_lock:
        return _prepare_catalog(catalog, rulebook, readme, cache_dir, interpretation)


def _prepare_catalog(catalog, rulebook, readme, cache_dir, interpretation=None):
    key = hashlib.sha256((catalog.fingerprint+semantic_version()+json.dumps(interpretation,sort_keys=True,ensure_ascii=False)).encode()).hexdigest()
    path = Path(cache_dir)/(key+".semantic.sqlite")
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists():
        semantic = json.loads(metadata.read_text(encoding="utf-8"))
    else:
        temporary = path.with_name(key+"."+uuid.uuid4().hex+".tmp")
        shutil.copyfile(catalog.database, temporary)
        connection = sqlite3.connect(temporary)
        try:
            semantic = build_semantics(connection, catalog.tables, rulebook, readme, interpretation)
            connection.commit()
        except Exception:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        finally:
            connection.close()
        temporary.replace(path)
        meta_tmp = metadata.with_name(metadata.name+"."+uuid.uuid4().hex+".tmp")
        meta_tmp.write_text(json.dumps(semantic,ensure_ascii=False),encoding="utf-8")
        meta_tmp.replace(metadata)
    return replace(catalog, database=path, fingerprint=key, semantic=semantic)
