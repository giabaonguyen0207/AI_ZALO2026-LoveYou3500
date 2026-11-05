"""Deterministic query tools over normalized facts, with explicit evidence."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import date

from ..data.profiling import inspect_column, quote
from ..data.rules import read_policy, resolve_period
from ..data.sql import SQLExecutor
from ..data.transforms import search_key
from ..domain import DataMateError


def request_needs(text):
    q = " "+(search_key(text) or "")+" "
    result = []
    if any(x in q for x in ("khach", "kh ")):
        result.append("customer_identity")
    if any(x in q for x in ("ton kho", "trong kho", "con ton", "con hang")):
        result.append("inventory")
    if any(x in q for x in ("con no", "cong no", "no nhieu", "no them", "tra no", "thu no", "dang no")):
        result.append("debt")
    if "huy" in q and any(x in q for x in ("vi giao", "ly do", "nguyen nhan")):
        result.append("cancellation_reason")
    if any(x in q for x in ("zalo", "tai quay", "kenh ban", "qua san")):
        result.append("channel")
    if any(x in q for x in ('tien mat','chuyen khoan','hinh thuc thanh toan','cod')):
        result.append('payment_method')
    return result


class BusinessTools:
    def __init__(self, catalog, rulebook, as_of_date, sql_timeout=12, max_rows=150, deadline=None):
        self.catalog, self.semantic = catalog, catalog.semantic
        self.rulebook, self.as_of_date = rulebook, as_of_date
        self.policy = read_policy(rulebook)
        self.executor = SQLExecutor(catalog.database,sql_timeout,max_rows,deadline)

    def profile(self, table, column=None, offset=0, limit=50):
        raw = next((t for t in self.catalog.tables if t["name"]==table),None)
        typed = next((t for t in self.semantic["tables"] if t["name"]==table),None)
        clean_binding = next((b for b in self.semantic["bindings"] if b.get("clean_table")==table),None)
        if not raw and not typed and not clean_binding:
            raise DataMateError("Bảng không có trong catalog")
        if column is None:
            source = table if raw else clean_binding["table"] if clean_binding else None
            profiles={}
            for name,original in self.semantic['profiles'].get(source,{}).items():
                p=dict(original)
                if len(p.get('values',[]))>limit:
                    p.update(values=sorted(p['values'],key=lambda v:(v['count'],str(v['value'])))[:limit],values_complete=False,next_offset=limit)
                profiles[name]=p
            return {"table":table,"scope":"all_rows", "profiles":profiles,
                    "note":"Tần suất trạng thái quét toàn bộ dòng. profile(column) xem giá trị hiếm/phân trang."}
        columns = raw["columns"] if raw else typed["columns"] if typed else next(t["columns"] for t in self.catalog.tables if t["name"]==clean_binding["table"])
        if column not in columns:
            raise DataMateError("Cột không có trong catalog")
        connection = sqlite3.connect(self.catalog.database.resolve().as_uri()+"?mode=ro",uri=True)
        try:
            started=time.monotonic()
            connection.set_progress_handler(lambda:int(time.monotonic()-started>self.executor.timeout_seconds or self.executor.deadline is not None and time.monotonic()>self.executor.deadline),1000)
            values = inspect_column(connection,table,column,offset,limit)
        finally:
            connection.close()
        return {"table":table,"column":column, **values,
                "profile":self.semantic["profiles"].get(table,{}).get(column)}

    def lookup(self, entity, text, limit=10):
        if entity not in {"product","customer"} or not isinstance(text,str) or not text.strip():
            raise DataMateError("lookup cần entity product/customer và text không rỗng")
        if type(limit) is not int or not 1<=limit<=30:
            raise DataMateError("lookup limit trong 1..30")
        prefix = "product" if entity=="product" else "customer"
        name, key, identifier = prefix+"_name",prefix+"_key",prefix+"_id"
        # Entity lists can exceed SQL result limits; inspect a bounded candidate
        # set in SQL instead of inferring from an arbitrarily truncated preview.
        folded = search_key(text)
        exact = self.executor.execute(f"SELECT {quote(key)},{quote(identifier)},{quote(name)} FROM dm_sales_all WHERE {quote(key)} IS NOT NULL AND (search_key({quote(name)})=? OR CAST({quote(identifier)} AS TEXT)=?) GROUP BY 1,2,3",[folded,text])
        if exact["rows"]:
            found = exact
        else:
            tokens = folded.split()
            predicates = " AND ".join(f"search_key({quote(name)}) LIKE ? ESCAPE '\\'" for _ in tokens)
            escaped = ["%"+t.replace("\\","\\\\").replace("%","\\%").replace("_","\\_")+"%" for t in tokens]
            found = self.executor.execute(f"SELECT {quote(key)},{quote(identifier)},{quote(name)} FROM dm_sales_all WHERE {quote(key)} IS NOT NULL AND ({predicates}) GROUP BY 1,2,3 ORDER BY 1 LIMIT ?",[*escaped,limit+1])
        candidates = [{"key":k,"id":identifier,"name":n} for k,identifier,n in found["rows"][:limit]]
        return {"entity":entity,"query":text,"candidates":candidates,
                "ambiguous":len({c['key'] for c in candidates})!=1, "truncated":found["truncated"] or len(found["rows"])>limit,
                "output_names_keep_diacritics":True}

    def check(self, question, period=None):
        needs = request_needs(question)
        caps, missing, unsupported = dict(self.semantic["capabilities"]),[],[]
        caps['payment_method']=any('payment_method' in b['roles'] for b in self.semantic['bindings'] if b['kind'] in {'sales','orders'})
        uncertain_schema=any(b['ambiguous'] or b['kind']=='other' and not b['roles'] for b in self.semantic['bindings'])
        hints={'customer_identity':('khach','customer','client','buyer'),
               'cancellation_reason':('ly do','nguyen nhan','reason','cause','ghi chu'),
               'channel':('kenh','channel','source','nguon'),'payment_method':('thanh toan','payment','paid','pttt')}
        for need in needs:
            if need in {"customer_identity","cancellation_reason","channel",'payment_method'} and not caps.get(need):
                unexplained=any(any(token in (search_key(c) or '') for token in hints[need]) and self.semantic['profiles'][b['table']][c]['nonempty']>0
                                for b in self.semantic['bindings'] if b['kind'] in {'orders','sales','other'} for c in b.get('unmapped',[]))
                (unsupported if uncertain_schema or unexplained else missing).append(need)
            elif need == "inventory" and not caps["inventory_snapshot"]:
                # Some datasets have lot ledgers even without a stock snapshot.
                # An unimplemented reconstruction is not proof of missing data.
                has_inputs = any("purchase_id" in b["roles"] or "purchase_date" in b["roles"] for b in self.semantic["bindings"])
                (unsupported if has_inputs or uncertain_schema else missing).append("inventory_balance")
            elif need == "debt" and not (caps["balance_snapshot"] or caps["payments"] and caps["order_total"]):
                monetary_sources = any(b["kind"]=="other" and any(x in b["roles"] for x in ("paid","order_total","payment_amount")) for b in self.semantic["bindings"])
                (unsupported if monetary_sources or uncertain_schema else missing).append("debt_balance_or_payments")
        coverage = self.semantic["coverage"]
        if period and coverage["start"]:
            resolved = resolve_period(period,self.as_of_date,self.policy)
            date_errors=self.executor.execute('SELECT (SELECT COUNT(*) FROM dm_sales_all WHERE order_date IS NULL)+(SELECT COUNT(*) FROM dm_orders_all WHERE order_date IS NULL)')['rows'][0][0]>0
            if (resolved["end"]<coverage["start"] or resolved["start"]>(coverage["declared_end"] or coverage["end"])) and not date_errors:
                missing.append("period_outside_dataset")
        return {"availability":"missing" if missing else "unsupported" if unsupported else "available",
                "missing":missing,"unsupported":unsupported,"requirements":needs,"coverage":coverage,
                "evidence_scope":"all_source_schemas_and_full_column_profiles",
                "source_bindings":[{"table":b["table"],"kind":b["kind"],"roles":b["roles"]} for b in self.semantic["bindings"]]}

    def verify_fact_sql(self,result,question):
        """Open-ended SQL remains available, with normalized/provenance guards."""
        direct=set(result.get('direct_tables',[]))
        bindings={b['table']:b for b in self.semantic['bindings']}
        bindings.update({b['clean_table']:b for b in self.semantic['bindings']})
        if not direct:
            raise DataMateError('SQL kết luận phải đọc dữ liệu, không dùng SELECT hằng số làm bằng chứng nghiệp vụ')
        for table in direct:
            binding=bindings.get(table)
            if binding and binding['kind'] in {'sales','orders'}:
                raise DataMateError('Tính trên raw/clean giao dịch chưa bảo đảm trạng thái và cấp đơn. Dùng dm_sales/dm_orders hoặc công cụ metric')
            if binding and table.startswith('raw_'):
                used={r['column'] for r in result.get('references',[]) if r['table']==table}
                if any(binding['roles'].get(role) in used for role in ('quantity','price','cost','line_amount','balance','payment_amount')):
                    raise DataMateError('SQL đang tính số từ raw dimension; dùng clean_table đã parse và kiểm tra profile')
        all_tables=direct & {'dm_sales_all','dm_orders_all'}
        q=search_key(question) or ''
        if all_tables and not any(x in q for x in ('don huy','don bi huy','don tra','huy la','ti le don huy')):
            raise DataMateError('Bảng *_all gồm cả hủy/trả; đại lượng bán hàng dùng dm_sales/dm_orders')
        for name in direct & {'dm_sales','dm_orders','dm_sales_all','dm_orders_all'}:
            base=name if name.endswith('_all') else name+'_all'
            bad=self.executor.execute(f"SELECT COUNT(*) FROM {base} WHERE status='unknown' OR order_date IS NULL")['rows'][0][0]
            if bad:
                raise DataMateError('SQL tự do có nguồn chưa giải được ngày/trạng thái; dùng profile hoặc metric theo phạm vi, không cộng thiếu dòng')
        return True

    def _period(self, spec):
        return resolve_period(spec,self.as_of_date,self.policy)

    def _quality(self, table, period, required, customer_identity=False):
        if self.policy.sales_statuses is None or not self.policy.line_revenue:
            raise DataMateError("Luật split chưa được adapter xác nhận; không tự áp luật train. Đọc policy.unknown và dùng SQL có bằng chứng hoặc abstain")
        coverage = self.semantic["coverage"]
        if not coverage["start"]:
            raise DataMateError("Không có khoảng ngày đã chuẩn hóa để kiểm chứng")
        if self.executor.execute(f'SELECT COUNT(*) FROM {table}_all WHERE order_date IS NULL')["rows"][0][0]:
            raise DataMateError("Có dòng chưa giải được ngày; không kết luận kỳ ngoài dữ liệu hoặc trả tổng một phần")
        if period["end"]<coverage["start"] or period["start"]>(coverage["declared_end"] or coverage["end"]):
            return {"availability":"missing","missing":["period_outside_dataset"],"coverage":coverage,"period":period}
        allowed = ",".join("'"+s+"'" for s in self.policy.sales_statuses)
        bad_columns = " OR ".join(quote(c)+" IS NULL" for c in required) or "0"
        conflicts="_dm_issues LIKE '%conflicting_order:order_date%' OR _dm_issues LIKE '%conflicting_order:status%'"
        if customer_identity:
            conflicts+=" OR _dm_issues LIKE '%conflicting_order:customer%'"
        result = self.executor.execute(f"SELECT COUNT(*) FROM {table}_all WHERE order_date IS NULL OR (order_date BETWEEN ? AND ? AND (status='unknown' OR (status IN ({allowed}) AND ({conflicts} OR {bad_columns}))))",[period["start"],period["end"]])
        if result["rows"][0][0]:
            raise DataMateError("Dữ liệu trong phạm vi có trạng thái chưa rõ, lỗi parse, khóa xung đột hoặc thiếu trường cần tính; xem profile. Không xuất tổng một phần/0")
        return None

    def _entity(self, entity, value):
        if isinstance(value,dict) and isinstance(value.get("key"),str):
            key = value["key"]
            column = entity+"_key"
            exists = self.executor.execute(f"SELECT COUNT(*) FROM dm_sales_all WHERE {quote(column)}=?",[key])["rows"][0][0]
            if not exists:
                raise DataMateError("Entity key không tồn tại")
            return key
        found = self.lookup(entity,value)
        if found["ambiguous"] or found["truncated"]:
            raise DataMateError("Tên không có ứng viên duy nhất; dùng lookup và chọn key có bằng chứng, không đoán hoặc coi bằng 0")
        return found["candidates"][0]["key"]

    def _filter(self, action, alias="", table='dm_sales'):
        prefix = alias+"." if alias else ""
        clauses, parameters = [],[]
        for entity in ("product","customer"):
            if entity in action:
                clauses.append((f'EXISTS (SELECT 1 FROM dm_sales sx WHERE sx.order_id=dm_orders.order_id AND sx.product_key=?)') if entity=='product' and table=='dm_orders' else prefix+entity+"_key=?")
                parameters.append(self._entity(entity,action[entity]))
        if "unit" in action:
            if table=='dm_orders':
                raise DataMateError('Lọc đơn theo đơn vị cần điều kiện dòng hàng rõ; dùng SQL EXISTS trên dm_sales')
            clauses.append(f"search_key({prefix}base_unit)=?")
            parameters.append(search_key(action["unit"]))
        return (" AND "+" AND ".join(clauses) if clauses else ""),parameters

    def metric(self, action):
        name = action.get("name")
        common={'action','name','reason','period','product','customer','unit'}
        extras={'inventory':{'date'},'inventory_units':{'date','large_unit'},'debt_rank':{'date','limit'},
                'revenue_series':{'periods'},'continuous_growth':{'periods'},'rank_products':{'limit','by'},
                'rank_customers':{'limit','by'},'rank_employees':{'limit'},'customers_with_orders':{'threshold'},
                'customer_percent':{'threshold'},'day_count':{'threshold'},'channel_revenue':{'channels'},
                'revenue_growth':{'compare_period'},'quantity_growth':{'compare_period'}}
        if name in {'inventory','inventory_units'}:
            common={'action','name','reason','product'}
        elif name=='debt_rank':
            common={'action','name','reason'}
        elif name in {'revenue_series','continuous_growth'}:
            common.discard('period')
        unknown=set(action)-common-extras.get(name,set())
        if unknown:
            raise DataMateError('Metric không hỗ trợ các tham số này, không tự bỏ qua: '+str(sorted(unknown)))
        if name in {"inventory","inventory_units"}:
            return self._inventory(action)
        if name == "debt_rank":
            return self._debt(action)
        supported = {"revenue","revenue_series","orders","best_day","revenue_growth","quantity_growth","continuous_growth",'rank_employees',
                     "rank_products","rank_customers","customers_with_orders","customer_percent","quantity","day_count","channel_revenue","shift_growth"}
        if name not in supported:
            raise DataMateError("Metric chưa hỗ trợ; không thay bằng một đại lượng khác")
        if (name in {"orders","customers_with_orders","customer_percent"} or name=='rank_customers' and action.get('by')=='orders') and not self.semantic["capabilities"]["order_identity"]:
            raise DataMateError("Nguồn không có mã đơn để đếm đơn; không đếm dòng thay đơn")
        if name in {"rank_customers","customers_with_orders","customer_percent"} and not self.semantic["capabilities"]["customer_identity"]:
            return {"availability":"missing","missing":["customer_identity"]}
        periods = action.get("periods") if name in {"continuous_growth","revenue_series"} else [action.get("period")]
        if name in {"revenue_growth","quantity_growth"}:
            periods.append(action.get("compare_period"))
        if not isinstance(periods,list) or not 1<=len(periods)<=12:
            raise DataMateError("periods cần list 1..12 kỳ")
        resolved = [self._period(p) for p in periods]
        by = action.get("by","quantity" if name=="rank_products" else "revenue")
        if name in {'rank_products','rank_customers'} and by not in ({'quantity','revenue'} if name=='rank_products' else {'orders','revenue'}):
            raise DataMateError("Tiêu chí xếp hạng không hỗ trợ")
        if name=='rank_products' and by=='quantity' and not self.policy.quantity_ranking:
            raise DataMateError('Chưa xác nhận quy tắc bán chạy theo số lượng của split')
        table = "dm_orders" if name=="orders" or name in {"customers_with_orders","customer_percent"} or name=="rank_customers" and by=="orders" else "dm_sales"
        if table=='dm_orders':
            for period in resolved:
                statuses=self.policy.sales_statuses or []
                marks=','.join('?' for _ in statuses) or 'NULL'
                missing=self.executor.execute(f"SELECT COUNT(*) FROM dm_sales_all WHERE order_date BETWEEN ? AND ? AND status IN ({marks}) AND order_id IS NULL",[period['start'],period['end'],*statuses])['rows'][0][0]
                if missing:
                    raise DataMateError('Có dòng không xác định được mã đơn; không đếm/xếp hạng trên phần đơn còn lại')
        field = "base_quantity" if name in {"quantity","quantity_growth"} or name=="rank_products" and by=="quantity" else "line_amount"
        required = ["order_id"] if table=="dm_orders" else [field]
        for p in resolved:
            error = self._quality(table,p,required,name in {'rank_customers','customers_with_orders','customer_percent'})
            if error:
                return error
        filtering, filter_params = self._filter(action,table=table)
        p = resolved[0]
        where, params = "order_date BETWEEN ? AND ?"+filtering,[p["start"],p["end"],*filter_params]
        limit = action.get("limit",1)
        if type(limit) is not int or not 1<=limit<=100:
            raise DataMateError("limit phải trong 1..100")
        sql, kind = "", "number"
        if name in {"revenue","quantity","orders"}:
            sql = f"SELECT {'COUNT(*)' if name=='orders' else 'COALESCE(SUM('+field+'),0)'} AS value FROM {table} WHERE {where}"
        elif name == "best_day":
            sql,kind = f"SELECT order_date FROM dm_sales WHERE {where} GROUP BY order_date ORDER BY SUM(line_amount) DESC,order_date ASC LIMIT 1","date"
        elif name == "day_count":
            threshold = action.get("threshold")
            if type(threshold) not in (int,float):
                raise DataMateError("threshold phải là số")
            sql = f"SELECT COUNT(*) FROM (SELECT order_date FROM dm_sales WHERE {where} GROUP BY order_date HAVING SUM(line_amount)>=?)"
            params.append(threshold)
        elif name=='rank_employees':
            nulls=self.executor.execute(f'SELECT COUNT(*) FROM dm_sales WHERE {where} AND employee IS NULL',params)['rows'][0][0]
            if nulls:
                raise DataMateError('Không đủ thông tin nhân viên của đơn')
            sql=f'SELECT employee FROM dm_sales WHERE {where} GROUP BY employee ORDER BY SUM(line_amount) DESC,employee ASC LIMIT ?'
            params.append(limit);kind='string' if limit==1 else 'list'
        elif name in {"rank_products","rank_customers"}:
            entity = "product" if name=="rank_products" else "customer"
            key, display = entity+"_key",entity+"_name"
            value = "COUNT(*)" if by=="orders" else f"SUM({field})"
            if name=="rank_products" and self.policy.exclude_service_product:
                where += " AND search_key(product_name)!='phi phuc vu'"
            sql = f"SELECT COALESCE(MIN({display}),CAST({entity}_id AS TEXT)) AS name FROM {table} WHERE {where} AND {key} IS NOT NULL GROUP BY {key} ORDER BY {value} DESC,{key} ASC LIMIT ?"
            params.append(limit)
            kind = "string" if limit==1 else "list"
        elif name in {"customers_with_orders","customer_percent"}:
            threshold = action.get("threshold")
            if type(threshold) is not int or threshold<1:
                raise DataMateError("threshold số đơn phải là số nguyên dương")
            sub = f"SELECT customer_key,COUNT(*) n FROM dm_orders WHERE {where} AND customer_key IS NOT NULL GROUP BY customer_key"
            if name=="customers_with_orders":
                sql = f"SELECT COUNT(*) FROM ({sub}) WHERE n>=?"
                params.append(threshold)
            else:
                sql = f"SELECT round_half_up(SUM(CASE WHEN n>=? THEN 1 ELSE 0 END)*100.0/NULLIF(COUNT(*),0),2) FROM ({sub})"
                params=[threshold,*params];kind="percent"
        elif name in {"revenue_growth","quantity_growth"}:
            compare = resolved[1]
            sql = f"WITH totals AS (SELECT COALESCE(SUM(CASE WHEN order_date BETWEEN ? AND ? THEN {field} ELSE 0 END),0) current,COALESCE(SUM(CASE WHEN order_date BETWEEN ? AND ? THEN {field} ELSE 0 END),0) previous FROM dm_sales WHERE 1=1{filtering}) SELECT round_half_up((current-previous)*100.0/NULLIF(previous,0),2) FROM totals"
            params=[p['start'],p['end'],compare['start'],compare['end'],*filter_params];kind="percent"
        elif name in {"continuous_growth","revenue_series"}:
            if name=='continuous_growth' and len(resolved)<2:
                raise DataMateError("Xu hướng cần ít nhất 2 kỳ")
            expressions = [f"COALESCE(SUM(CASE WHEN order_date BETWEEN ? AND ? THEN line_amount ELSE 0 END),0) AS p{i}" for i in range(len(resolved))]
            condition = " AND ".join(f"p{i}>p{i-1}" for i in range(1,len(resolved)))
            projection=f'CASE WHEN {condition} THEN 1 ELSE 0 END' if name=='continuous_growth' else ','.join('p'+str(i) for i in range(len(resolved)))
            sql=f"WITH totals AS (SELECT {','.join(expressions)} FROM dm_sales WHERE 1=1{filtering}) SELECT {projection} FROM totals"
            params=[d for period in resolved for d in (period['start'],period['end'])]+filter_params;kind="boolean" if name=='continuous_growth' else 'list'
        elif name == "shift_growth":
            if not self.policy.shifts:
                raise DataMateError('Chưa xác nhận khung giờ các ca từ rulebook')
            missing_time=self.executor.execute(f"SELECT COUNT(*) FROM dm_sales WHERE {where} AND order_timestamp IS NULL",params)["rows"][0][0]
            if missing_time:
                raise DataMateError("Không đủ giờ đặt đơn cho phép so sánh ca")
            sql=f"WITH totals AS (SELECT SUM(CASE WHEN substr(order_timestamp,12,5)>=? AND substr(order_timestamp,12,5)<? THEN line_amount ELSE 0 END) morning,SUM(CASE WHEN substr(order_timestamp,12,5)>=? AND substr(order_timestamp,12,5)<? THEN line_amount ELSE 0 END) afternoon FROM dm_sales WHERE {where}) SELECT round_half_up((afternoon-morning)*100.0/NULLIF(morning,0),2) FROM totals";kind="percent"
            params=[*self.policy.shifts['morning'],*self.policy.shifts['afternoon'],*params]
        elif name == "channel_revenue":
            if not self.semantic["capabilities"]["channel"]:
                return {"availability":"missing","missing":["channel"]}
            channels=action.get("channels")
            if not isinstance(channels,list) or not channels or len(channels)>10 or not all(isinstance(x,str) for x in channels):
                raise DataMateError("channels cần list tên kênh theo thứ tự câu hỏi")
            aliases={"zalo oa":{"zalo oa","zalo"},"tai quay":{"tai quay","quay","store","offline"}}
            expr=[]; channel_params=[]
            for i,channel in enumerate(channels):
                folded=search_key(channel);values=next((v for v in aliases.values() if folded in v),{folded})
                expr.append(f"COALESCE(SUM(CASE WHEN search_key(channel) IN ({','.join('?' for _ in values)}) THEN line_amount ELSE 0 END),0) AS channel_{i}")
                channel_params.extend(sorted(values))
            sql=f"SELECT {','.join(expr)} FROM dm_sales WHERE {where}";params=channel_params+params;kind="list"
        if kind=='percent':
            places=self.policy.percent_places
            if type(places) is not int or not 0<=places<=6:
                raise DataMateError('Chưa xác nhận cách làm tròn phần trăm từ rulebook')
            # These are internal templates, never arbitrary/model-authored SQL.
            sql=sql.replace(',2)',f',{places})')
        result=self.executor.execute(sql,params)
        if not result['rows'] or any(value is None for row in result['rows'] for value in row):
            raise DataMateError("Kết quả không xác định/rỗng (ví dụ mẫu số bằng 0); không tự đổi thành 0 hay unanswerable")
        return {**result,"availability":"available","answer_type":kind,"metric":name,"periods":resolved,
                "verified":True,"sql":sql,"parameters":params,"basis":"typed_facts_current_split_policy"}

    def _inventory(self,action):
        if not self.semantic['capabilities']['inventory_snapshot']:
            inputs=any('purchase_id' in b['roles'] or 'purchase_date' in b['roles'] for b in self.semantic['bindings'])
            if inputs:
                raise DataMateError("Có sổ nhập/lô; cần tái dựng tồn theo lô, công cụ snapshot không thay thế phép tính này")
            return {"availability":"missing","missing":["inventory_balance"]}
        text=action.get('product')
        if not isinstance(text,str):
            raise DataMateError("inventory cần product tên/mã")
        day=action.get('date',self.as_of_date)
        date.fromisoformat(day)
        query="SELECT product_id,product_name,quantity,snapshot_date,_dm_issues FROM dm_inventory WHERE search_key(product_name)=? OR CAST(product_id AS TEXT)=?"
        rows=self.executor.execute(query,[search_key(text),text])['rows']
        if len(rows)!=1:
            raise DataMateError("Snapshot không có đúng 1 đối tượng phù hợp; xem profile/lookup")
        pid,pname,quantity,snapshot,issues=rows[0]
        if day!=snapshot:
            raise DataMateError("Snapshot chỉ có giá trị tại ngày cuối dữ liệu; cần lịch sử nhập/xuất cho ngày hỏi")
        if issues!='[]' or quantity is None:
            raise DataMateError("Snapshot có lỗi parse/thiếu số lượng")
        if quantity<0:
            raise DataMateError('Snapshot tồn âm; chưa xác định cách diễn giải từ nguồn')
        entity_where='product_id=?' if pid is not None else 'product_name=?'
        entity_value=pid if pid is not None else pname
        if action['name']=='inventory_units':
            large=action.get('large_unit')
            factors=self.executor.execute("SELECT DISTINCT factor FROM dm_conversions WHERE (product_id=? OR search_key(product_name)=?) AND search_key(large_unit)=?",[pid,search_key(pname),search_key(large)])['rows']
            if len(factors)!=1 or factors[0][0] is None or factors[0][0]<=0:
                raise DataMateError("Chưa có hệ số quy đổi duy nhất cho đơn vị lớn")
            factor=factors[0][0]
            sql=f"SELECT CAST(quantity/? AS INTEGER) AS whole,quantity-CAST(quantity/? AS INTEGER)*? AS remainder FROM dm_inventory WHERE {entity_where} AND snapshot_date=?"
            params=[factor,factor,factor,entity_value,day];kind='list'
        else:
            sql=f"SELECT quantity FROM dm_inventory WHERE {entity_where} AND snapshot_date=?"
            params=[entity_value,day];kind='number'
        return {**self.executor.execute(sql,params),'availability':'available','answer_type':kind,'metric':action['name'],'verified':True,'sql':sql,'parameters':params}

    def _debt(self,action):
        if not self.policy.debt_supported:
            raise DataMateError('Luật công nợ của split chưa được adapter xác nhận; không tự áp luật train')
        day=action.get('date',self.as_of_date);date.fromisoformat(day)
        limit=action.get('limit',3)
        if type(limit) is not int or not 1<=limit<=100:
            raise DataMateError("limit trong 1..100")
        caps=self.semantic['capabilities'];cutoff=self.semantic['coverage']['declared_end']
        if not caps['customer_identity']:
            return {'availability':'missing','missing':['customer_identity']}
        if not caps['balance_snapshot'] and not (caps['payments'] and caps['order_total']):
            potential=any(b['ambiguous'] or b['kind']=='other' and any(role in b['roles'] for role in ('payment_amount','paid','order_total')) for b in self.semantic['bindings'])
            if potential:
                raise DataMateError('Có nguồn tiền/khóa liên kết chưa giải được; đây là schema chưa hỗ trợ, không phải bằng chứng unanswerable')
            return {'availability':'missing','missing':['debt_balance_or_payments']}
        methods=self.executor.execute("SELECT COUNT(*) FROM dm_orders_all WHERE order_date<=? AND (payment_method IS NULL OR payment_method='unknown' OR status='unknown')",[day])['rows'][0][0]
        missing_dates=self.executor.execute('SELECT COUNT(*) FROM dm_orders_all WHERE order_date IS NULL')['rows'][0][0]
        if methods or missing_dates:
            raise DataMateError("Chưa xác định đủ trạng thái/hình thức thanh toán để loại COD khỏi công nợ")
        if caps['balance_snapshot'] and day==cutoff:
            missing=self.executor.execute("SELECT COUNT(*) FROM dm_orders_all WHERE order_date<=? AND status!='cancelled' AND payment_method NOT IN ('cod','marketplace') AND balance IS NULL",[day])['rows'][0][0]
            if missing:
                raise DataMateError('Có đơn thiếu/không parse được dư nợ; không xếp hạng một phần')
            sql="SELECT COALESCE(MIN(customer_name),customer_id) AS khach_hang,round_half_up(SUM(balance),0) AS so_tien_no FROM dm_orders_all WHERE order_date<=? AND status!='cancelled' AND payment_method NOT IN ('cod','marketplace') AND customer_key IS NOT NULL AND balance>0 GROUP BY customer_key ORDER BY SUM(balance) DESC,customer_key ASC LIMIT ?"
            params=[day,limit]
        elif caps['payments'] and caps['order_total']:
            missing=self.executor.execute("SELECT COUNT(*) FROM dm_payments WHERE payment_date IS NULL OR amount IS NULL")['rows'][0][0]
            missing+=self.executor.execute("SELECT COUNT(*) FROM dm_orders_all WHERE order_date<=? AND status!='cancelled' AND payment_method NOT IN ('cod','marketplace') AND order_total IS NULL",[day])['rows'][0][0]
            if missing:
                raise DataMateError('Không đủ ngày/số tiền trong sổ thu hoặc tổng phải trả của đơn')
            # Reconcile the ledger before using it for a past balance. A
            # cumulative 'paid today' value must not leak into a past date.
            mismatch=self.executor.execute("WITH p AS (SELECT order_id,SUM(amount) paid FROM dm_payments GROUP BY order_id) SELECT COUNT(*) FROM dm_orders_all o LEFT JOIN p ON p.order_id=o.order_id WHERE o.paid IS NOT NULL AND ABS(o.paid-COALESCE(p.paid,0))>0.01")['rows'][0][0]
            if mismatch:
                raise DataMateError("Sổ thu chưa đối soát được với tiền đã trả; không dùng để tái dựng công nợ quá khứ")
            sql="WITH p AS (SELECT order_id,SUM(amount) paid FROM dm_payments WHERE payment_date<=? GROUP BY order_id),d AS (SELECT o.*,MAX(o.order_total-COALESCE(p.paid,0),0) debt FROM dm_orders_all o LEFT JOIN p ON p.order_id=o.order_id WHERE o.order_date<=? AND o.status!='cancelled' AND o.payment_method NOT IN ('cod','marketplace') AND o.customer_key IS NOT NULL) SELECT COALESCE(MIN(customer_name),customer_id) AS khach_hang,round_half_up(SUM(debt),0) AS so_tien_no FROM d WHERE debt>0 GROUP BY customer_key ORDER BY SUM(debt) DESC,customer_key ASC LIMIT ?"
            params=[day,day,limit]
        else:
            raise DataMateError("Chỉ có dư nợ snapshot, chưa có lịch sử thanh toán để tính ngày quá khứ")
        result=self.executor.execute(sql,params)
        return {**result,'availability':'available','answer_type':'table','metric':'debt_rank','verified':True,'sql':sql,'parameters':params}
