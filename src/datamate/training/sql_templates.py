"""Auditable SQL templates for TRAIN curation, with source-driven identifiers."""
from __future__ import annotations
import re
from datetime import date,timedelta
from ..data.ingest import quote
from ..data.transforms import search_key
from ..domain import DataMateError


def binding(tools,roles,*,exclude=()):
    values=[b for b in tools.semantic['bindings'] if set(roles)<=set(b['roles']) and b['kind'] not in exclude]
    if len(values)!=1:raise DataMateError('Need one inspected source for roles '+str(roles)+'; got '+str(len(values)))
    return values[0]


def col(b,role,alias=None):
    return (alias+'.' if alias else '')+quote(b['roles'][role])


def find_column(b,labels):
    compact=lambda s:re.sub(r'[^a-z0-9]','',search_key(s) or '')
    names=list(b['roles'].values())+b.get('unmapped',[])
    wanted={compact(v) for v in labels}
    found=[c for c in names if compact(c) in wanted]
    if len(found)!=1:raise DataMateError('Ambiguous/missing source column '+str(labels)+' in '+b['table'])
    return found[0]


def date_expression(tools,b,column,alias=''):
    field=(alias+'.' if alias else '')+quote(column)
    meta=next(t for t in tools.catalog.tables if t['name']==b['table'])
    epoch=meta.get('excel_epoch')
    if not epoch:return 'iso_date('+field+')'
    epoch=epoch[:10]
    date.fromisoformat(epoch)
    return "CASE WHEN typeof("+field+") IN ('integer','real') THEN date('"+epoch+"',CAST("+field+" AS TEXT)||' days') ELSE iso_date("+field+') END'


def customer_display(expression):
    """Remove a trailing phone suffix, preserving the source's actual name."""
    result=expression
    for separator in (' 0',' (',' - '):
        tail='trim(substr('+expression+',instr('+expression+',\''+separator+'\')+'+str(len(separator))+'))'
        digits="replace(replace(replace("+tail+",' ',''),')',''),'-','')"
        sizes='(9,10)' if separator==' 0' else '(10,11)'
        condition='instr('+expression+',\''+separator+'\')>0 AND length('+digits+') IN '+sizes+' AND '+digits+" NOT GLOB '*[^0-9]*'"
        result='CASE WHEN '+condition+' THEN trim(substr('+expression+',1,instr('+expression+',\''+separator+'\')-1)) ELSE '+result+' END'
    return result


def customer_dimension(tools):
    cached=getattr(tools.catalog,'_sql_customer_dimension',None)
    if cached:return cached
    count=tools.executor.execute('SELECT COUNT(*) FROM dm_customers WHERE customer_name IS NOT NULL')['rows'][0][0]
    if count:result=('dm_customers','customer_id','customer_name')
    else:
        result=None
        for b in tools.semantic['bindings']:
            if 'customer_id' not in b['roles'] or b['kind'] in {'sales','orders'}:continue
            try:name=b['roles'].get('customer_name') or find_column(b,['Khách lẻ/sỉ','ten_kh','TenGoi','Tên khách hàng'])
            except DataMateError:continue
            result=(b['clean_table'],b['roles']['customer_id'],name);break
        if result is None:result=('dm_customers','customer_id','customer_name')
    tools.catalog._sql_customer_dimension=result
    return result


def purchase_relation(tools):
    enriched=[]
    for original in tools.semantic['bindings']:
        if 'purchase_id' not in original['roles']:continue
        b={**original,'roles':dict(original['roles'])}
        for role,labels in {'price':['DonGiaNhap','Giá nhập','Đơn giá'],'purchase_date':['NgayNhapKho','Ngày nhập','Thời gian']}.items():
            if role not in b['roles']:
                try:b['roles'][role]=find_column(b,labels)
                except DataMateError:pass
        enriched.append(b)
    options=[b for b in enriched if {'quantity','price'}<=set(b['roles'])]
    if len(options)!=1:raise DataMateError('Purchase details are not uniquely identified')
    lines=options[0]
    flat='purchase_date' in lines['roles']
    if flat:header=lines
    else:
        headers=[b for b in enriched if b['table']!=lines['table'] and 'purchase_date' in b['roles']]
        if len(headers)!=1:raise DataMateError('Purchase header not uniquely identified')
        header=headers[0]
    datecol=header['roles'].get('purchase_date') or find_column(header,['Thời gian','NgayNhapKho','Ngày nhập'])
    relation=quote(lines['clean_table'])+' AS l'
    if not flat:relation+=' JOIN '+quote(header['clean_table'])+' AS h ON '+col(lines,'purchase_id','l')+'='+col(header,'purchase_id','h')
    h='l' if flat else 'h'
    valid='1=1'
    if 'status' in header['roles']:
        valid="search_key("+col(header,'status',h)+") NOT IN ('huy','da huy','cancelled','canceled')"
    return lines,header,relation,h+'.'+quote(datecol),valid,h


def lot_stock(tools,day):
    """Reconstruct each lot from purchases minus net issued quantities.

    Purchase and issue quantities below are explicitly base-unit columns from
    the inspected exports. A lot identifier must be unique in its source.
    """
    candidates=[]
    for b in tools.semantic['bindings']:
        if 'purchase_id' not in b['roles'] or 'order_id' in b['roles']:continue
        try:lot=b['roles'].get('lot_id') or find_column(b,['Lô','Mã lô','MaLo'])
        except DataMateError:continue
        try:qty=find_column(b,['SLNhapQuyDoi','SL nhập (ĐVT cơ bản)','Số lượng (ĐVT cơ bản)','Số lượng nhập'])
        except DataMateError:continue
        candidates.append((b,lot,qty))
    if len(candidates)!=1:raise DataMateError('Lot opening source not uniquely identified')
    b,lot,qty=candidates[0]
    duplicates=tools.executor.execute('SELECT COUNT(*) FROM (SELECT '+quote(lot)+' FROM '+quote(b['clean_table'])+' GROUP BY '+quote(lot)+' HAVING COUNT(*)>1)')['rows'][0][0]
    if duplicates:raise DataMateError('Lot key is not unique')
    issue_candidates=[]
    for s in tools.semantic['bindings']:
        if 'order_id' not in s['roles']:continue
        try:lid=s['roles'].get('lot_id') or find_column(s,['Lô','Mã lô','MaLo'])
        except DataMateError:continue
        try:issued=s['roles'].get('base_quantity') or find_column(s,['SLQuyDoi','SL quy ra lẻ','Số lượng (ĐVT cơ bản)','Số lượng'])
        except DataMateError:continue
        returned=s['roles'].get('return_quantity')
        if returned is None:
            try:returned=find_column(s,['SLTraQuyDoi','SL trả (ĐVT cơ bản)','Số lượng trả (ĐVT cơ bản)','Số lượng trả'])
            except DataMateError:returned=None
        issue_candidates.append((s,lid,issued,returned))
    if not issue_candidates:raise DataMateError('Lot issue source unavailable')
    parts=[]
    for s,lid,issued,returned in issue_candidates:
        parts.append('SELECT '+col(s,'order_id')+' AS order_id,'+quote(lid)+' AS lot,'+quote(issued)+' AS issued,'+(quote(returned) if returned else 'NULL')+' AS returned FROM '+quote(s['clean_table']))
    sold=('SELECT x.lot,SUM(num(x.issued)-COALESCE(num(x.returned),0)) AS used FROM ('+' UNION '.join(parts)+') x JOIN dm_orders_all o ON o.order_id=CAST(x.order_id AS TEXT) '
          "WHERE o.order_date<=? AND o.status!='cancelled' GROUP BY x.lot")
    headers=[h for h in tools.semantic['bindings'] if 'purchase_id' in h['roles'] and h['table']!=b['table'] and
             ('purchase_date' in h['roles'] or any(re.sub(r'[^a-z0-9]','',search_key(c) or '') in {'thoigian','ngaynhap','ngaynhapkho'} for c in h['unmapped']))]
    relation=quote(b['clean_table'])+' l LEFT JOIN ('+sold+') u ON u.lot=l.'+quote(lot)
    where='1=1';params=[day]
    if len(headers)==1:
        h=headers[0];d=h['roles'].get('purchase_date') or find_column(h,['Thời gian','Ngày nhập','NgayNhapKho'])
        columns=[h['roles']['purchase_id'],d]+([h['roles']['status']] if 'status' in h['roles'] else [])
        header='(SELECT DISTINCT '+','.join(quote(c) for c in columns)+' FROM '+quote(h['clean_table'])+')'
        relation+=' JOIN '+header+' h ON '+col(h,'purchase_id','h')+'='+col(b,'purchase_id','l')
        where='iso_date(h.'+quote(d)+')<=?';params.append(day)
        if 'status' in h['roles']:where+=" AND search_key("+col(h,'status','h')+") NOT IN ('huy','da huy','cancelled','canceled')"
    elif 'purchase_date' in b['roles']:
        where='iso_date('+col(b,'purchase_date','l')+')<=?';params.append(day)
    else:raise DataMateError('No purchase dates for opening lots')
    expiry=b['roles'].get('expiry_date') or find_column(b,['HSD','Hạn sử dụng','HanSuDung'])
    product=col(b,'product_id','l')
    cost='NULL'
    try:
        lines,_,_,_,_,_=purchase_relation(tools)
        detail_lot=lines['roles'].get('lot_id') or find_column(lines,['Lô','Mã lô','MaLo'])
        cost='(SELECT 1.0*'+col(lines,'quantity','d')+'*'+col(lines,'price','d')+'/NULLIF(num(l.'+quote(qty)+'),0) FROM '+quote(lines['clean_table'])+' d WHERE d.'+quote(detail_lot)+'=l.'+quote(lot)+' LIMIT 1)'
    except DataMateError:pass
    sql=('SELECT CAST('+product+' AS TEXT) AS product_id,iso_date(l.'+quote(expiry)+') AS expiry_date,MAX(num(l.'+quote(qty)+')-COALESCE(u.used,0),0) AS quantity,'+cost+' AS unit_cost FROM '+relation+' WHERE '+where)
    return sql,params,b


def plan_sql(recipe,sql,params,*,source='dm_sales',measure='custom',kind='number',population='sales'):
    return {'action':'plan','measure':measure,'operation':'custom','type':kind,'population':population,
            'source':source,'evidence_id':'Q','periods':recipe.get('periods',[]),'filters':[],
            'query':{'sql':sql,'parameters':params}}


def dimension_filter(tools,role,text,*,identity='product',alias='s'):
    """Exact values from a real dimension; EXISTS cannot multiply fact rows."""
    text=' '+search_key(text)+' '
    candidates=[]
    for original in tools.semantic['bindings']:
        if identity+'_id' not in original['roles'] or original['kind'] in {'sales','orders'}:continue
        if role in original['roles']:candidates.append(original);continue
        for c in original['unmapped']:
            if not any(t in (search_key(c) or '') for t in ('nhom','loai','hang','phan loai')):continue
            b={**original,'roles':{**original['roles'],role:c}}
            candidates.append(b)
    for b in candidates:
        values=tools.executor.execute('SELECT DISTINCT '+col(b,role)+' FROM '+quote(b['clean_table']))
        found=sorted([v[0] for v in values['rows'] if isinstance(v[0],str) and (' '+search_key(v[0])+' ') in text],key=len,reverse=True)
        if found:
            return (alias+'.'+quote(identity+'_id')+' IN (SELECT CAST('+col(b,identity+'_id','d')+' AS TEXT) FROM '+quote(b['clean_table'])+' AS d WHERE search_key('+col(b,role,'d')+')=?)',[search_key(found[0])])
    values=tools.semantic.get('domains',{}).get(role,[])
    found=sorted([v for v in values if isinstance(v,str) and ' '+search_key(v)+' ' in text],key=len,reverse=True)
    if found:return ('search_key('+alias+'.'+quote(role)+')=?',[search_key(found[0])])
    raise DataMateError('No exact source value for dimension '+role)


def sales_filter(recipe,tools):
    q=recipe['question_key'];clauses=[];params=[]
    if recipe.get('product'):
        clauses.append('search_key(s.product_name)=?');params.append(search_key(recipe['product']))
    elif recipe.get('unit'):
        clauses.append('search_key(s.base_unit)=?');params.append(search_key(recipe['unit']))
    if recipe.get('size'):
        clauses.append('lower(substr(trim(s.product_name),-3))=?');params.append('('+recipe['size'].lower()+')')
    if recipe.get('shift'):
        lo,hi=tools.policy.shifts[recipe['shift']]
        clauses.append('substr(s.order_timestamp,12,5)>=? AND substr(s.order_timestamp,12,5)<?');params.extend([lo,hi])
    if 'nhom' in q and (not recipe.get('product') or 'nhom hang' in q):
        role='customer_group' if 'khach' in q else 'product_group'
        clause,p=dimension_filter(tools,role,q,identity='customer' if role=='customer_group' else 'product')
        clauses.append(clause);params.extend(p)
    if 'ke don' in q:
        clauses.append('COALESCE(s.prescription,0)<>1' if re.search(r'(?:khong|hok|ko|hong|k)\s+ke don|tru thuoc ke don',q) else 's.prescription=1')
    if 'hoat chat' in q:
        b=None
        for candidate in tools.semantic['bindings']:
            if 'product_id' not in candidate['roles']:continue
            try:column=find_column(candidate,['Hoạt chất','HoatChat','ThanhPhanChinh'])
            except DataMateError:continue
            b=candidate;break
        if b is None:raise DataMateError('No active ingredient source')
        values=tools.executor.execute('SELECT DISTINCT '+quote(column)+' FROM '+quote(b['clean_table']))['rows']
        canonical=' '+search_key(q)+' '
        found=sorted([v[0] for v in values if v[0] and ' '+search_key(v[0])+' ' in canonical],key=len,reverse=True)
        if not found:raise DataMateError('No exact active ingredient in question')
        clauses.append('s.product_id IN (SELECT CAST('+col(b,'product_id','d')+' AS TEXT) FROM '+quote(b['clean_table'])+' d WHERE search_key(d.'+quote(column)+')=?)')
        params.append(search_key(found[0]))
    if 'cuoi tuan' in q or 'thu bay' in q and 'chu nhat' in q:
        clauses.append("strftime('%w',s.order_date) IN ('0','6')")
    methods=[(q.find(label),value) for label,value in [('tien mat','cash'),('chuyen khoan','transfer'),('cod','cod'),('vi dien tu','ewallet')] if label in q]
    if len(methods)==1:clauses.append('s.payment_method=?');params.append(methods[0][1])
    channels=channel_filters(q,tools)
    if len(channels)==1:clauses.append(channels[0][1]);params.extend(channels[0][2])
    employee=re.search(r'\bnv\d+\b',q)
    if employee:
        clauses.append('s.employee=?')
        params.append(employee[0].upper())
    elif 'nhan vien' in q:
        emp=next((b for b in tools.semantic['bindings'] if 'nhan_vien' in b['table']),None)
        if emp:
            name=find_column(emp,['TenGoi','Tên nhân viên']);identifier=find_column(emp,['MaNhanVien','Mã nhân viên'])
            values=tools.executor.execute('SELECT '+quote(name)+' FROM '+quote(emp['clean_table']))['rows']
            found=[v[0] for v in values if v[0] and ' '+search_key(v[0])+' ' in q]
            if len(found)==1:
                h,field=employee_source(tools)
                clauses.append('s.order_id IN (SELECT CAST('+col(h,'order_id','h')+' AS TEXT) FROM '+quote(h['clean_table'])+' h JOIN '+quote(emp['clean_table'])+' e ON e.'+quote(identifier)+'=h.'+quote(field)+' WHERE search_key(e.'+quote(name)+')=?)')
                params.append(search_key(found[0]))
    return (' AND '.join(clauses) if clauses else '1=1'),params


def channel_filters(q,tools):
    requested=sorted([(q.find(label),code) for label,code in [('zalo oa','zalo oa'),('tai quay','store'),('dien thoai','phone')] if label in q])
    if not requested:return []
    if tools.semantic['capabilities'].get('channel'):
        return [(position,'search_key(s.channel)=?',[code]) for position,code in requested]
    headers=[b for b in tools.semantic['bindings'] if b['kind']=='orders']
    if len(headers)!=1:raise DataMateError('Channel order source not uniquely identified')
    h=headers[0];field=h['roles'].get('channel') or find_column(h,['Bán qua','Kênh bán','Nguồn đơn','KenhBanHang'])
    return [(position,'s.order_id IN (SELECT CAST('+col(h,'order_id','h')+' AS TEXT) FROM '+quote(h['clean_table'])+' h WHERE search_key(h.'+quote(field)+')=?)',[code]) for position,code in requested]


def employee_source(tools):
    h=binding(tools,['order_id','order_date'],exclude=('sales',))
    return h,h['roles'].get('employee') or find_column(h,['MaNhanVien','Mã NV bán','Người bán'])


def price_effect_periods(recipe,question,tools):
    q=recipe['question_key'];n=re.search(r'(\d+) ngay',q)
    price=re.search(r'(?:len gia|gia len|doi gia.*?len)\s*([\d.,]+)',q)
    if not n or not price:raise DataMateError('Price-effect anchor absent and price not explicitly stated')
    from ..data.transforms import parse_number
    amount=parse_number(price[1]+'đ','vi')
    source=None
    for b in tools.semantic['bindings']:
        if not ('product_id' in b['roles'] or 'product_name' in b['roles']):continue
        try:d=find_column(b,['Ngày áp dụng','NgayApDung']);v=find_column(b,['Giá mới','GiaBanLe'])
        except DataMateError:continue
        source=(b,d,v);break
    if source is None:raise DataMateError('Price history unavailable')
    b,d,v=source;table=quote(b['clean_table'])
    meta=next(t for t in tools.catalog.tables if t['name']==b['table'])
    epoch=meta.get('excel_epoch','1899-12-30T00:00:00')[:10]
    # The date modifier expression is kept explicit for offline SQLite.
    day='CASE WHEN typeof('+quote(d)+") IN ('integer','real') THEN date(?,CAST("+quote(d)+" AS TEXT)||' days') ELSE iso_date("+quote(d)+') END'
    val='parse_number_locale('+quote(v)+",'vi')"
    if 'product_name' in b['roles']:where='search_key('+col(b,'product_name')+')=?'
    else:where='EXISTS (SELECT 1 FROM dm_products p WHERE p.product_id=CAST('+col(b,'product_id')+' AS TEXT) AND search_key(p.product_name)=?)'
    sql=('SELECT effective_date FROM (SELECT effective_date,price,LAG(price) OVER (ORDER BY effective_date) AS previous_price FROM (SELECT '+day+' AS effective_date,'+val+' AS price FROM '+table+' WHERE '+where+')) WHERE price=? AND price>previous_price ORDER BY effective_date LIMIT 1')
    params=[epoch,search_key(recipe['product']),amount]
    res=tools.executor.execute(sql,params)
    if len(res['rows'])!=1:raise DataMateError('No unique earliest increase to requested price')
    anchor=date.fromisoformat(res['rows'][0][0]);days=int(n[1])
    recipe['periods']=[{'kind':'range','start':anchor.isoformat(),'end':(anchor+timedelta(days=days-1)).isoformat()},
                       {'kind':'range','start':(anchor-timedelta(days=days)).isoformat(),'end':(anchor-timedelta(days=1)).isoformat()}]
    recipe['source_discovery_actions']=[{'action':'schema','table':b['table']},{'action':'sql','sql':sql,'parameters':params}]
    return recipe


def compile_custom(recipe,question,tools):
    task=recipe['task'];ps=recipe.get('periods',[]);q=recipe['question_key'];checks=[]
    if task in {'rank_customers','average_order_revenue'}:
        p=ps[0];where,filters=sales_filter(recipe,tools)
        if task=='average_order_revenue':
            sql='SELECT round_half_up(SUM(s.line_amount)*1.0/NULLIF(COUNT(DISTINCT s.order_id),0),0) AS value FROM dm_sales s WHERE s.order_date BETWEEN ? AND ? AND '+where
            kind='number';params=[p['start'],p['end'],*filters]
        else:
            dimension,identifier,name=customer_dimension(tools)
            aggregate='COUNT(DISTINCT s.order_id)' if recipe.get('by')=='orders' else 'SUM(s.line_amount)'
            sql=('SELECT COALESCE(MIN('+customer_display('c.'+quote(name))+'),MIN('+customer_display('s.customer_name')+'),s.customer_id) AS name FROM dm_sales s LEFT JOIN '+quote(dimension)+' c ON CAST(c.'+quote(identifier)+' AS TEXT)=s.customer_id '
                 'WHERE s.order_date BETWEEN ? AND ? AND s.customer_key IS NOT NULL AND '+where+' GROUP BY s.customer_key ORDER BY '+aggregate+' DESC,s.customer_key ASC LIMIT ?')
            kind='string' if recipe['limit']==1 else 'list';params=[p['start'],p['end'],*filters,recipe['limit']]
        return [plan_sql(recipe,sql,params,kind=kind,measure='revenue' if recipe.get('by')!='orders' else 'orders')],checks
    if task=='rank_employees':
        p=ps[0]
        sql='SELECT s.employee AS name FROM dm_sales s WHERE s.order_date BETWEEN ? AND ? AND s.employee IS NOT NULL GROUP BY s.employee ORDER BY SUM(s.line_amount) DESC,s.employee ASC LIMIT ?'
        return [plan_sql(recipe,sql,[p['start'],p['end'],recipe['limit']],kind='string' if recipe['limit']==1 else 'list',measure='revenue')],checks
    if task=='shift_growth':
        p=ps[0];count='so don' in q or 'don ca' in q or 'thi so don' in q
        table='dm_orders' if count else 'dm_sales';measure='orders' if count else 'revenue'
        def aggregate(shift):
            lo,hi=tools.policy.shifts[shift]
            cond='substr(order_timestamp,12,5)>=? AND substr(order_timestamp,12,5)<?'
            expr='COUNT(DISTINCT CASE WHEN '+cond+' THEN order_id END)' if count else 'COALESCE(SUM(CASE WHEN '+cond+' THEN line_amount ELSE 0 END),0)'
            return 'SELECT '+expr+' AS value FROM '+table+' WHERE order_date BETWEEN ? AND ?',[lo,hi,p['start'],p['end']]
        a,ap=aggregate('afternoon');m,mp=aggregate('morning')
        sql='SELECT round_half_up((a.value-m.value)*100.0/NULLIF(m.value,0),2) AS value FROM ('+a+') a CROSS JOIN ('+m+') m'
        return [plan_sql(recipe,sql,[*ap,*mp],source=table,measure=measure,kind='percent')],checks
    if task in {'revenue','revenue_ratio','quantity','gross_profit','rank_profit','unsold','product_order_threshold'}:
        if len(ps)!=1:raise DataMateError('One sales period required for '+task)
        p=ps[0];where,filters=sales_filter(recipe,tools);params=[p['start'],p['end'],*filters]
        if task=='unsold':
            sql=('SELECT COUNT(*) AS value FROM (SELECT DISTINCT product_key FROM dm_sales WHERE order_date<? AND product_key IS NOT NULL '
                 "AND search_key(product_name)!='phi phuc vu') a WHERE NOT EXISTS (SELECT 1 FROM dm_sales b WHERE b.product_key=a.product_key AND b.order_date BETWEEN ? AND ?)")
            params=[p['start'],p['start'],p['end']]
        elif task=='product_order_threshold':
            threshold=recipe.get('threshold')
            if threshold is None:
                m=re.search(r'(\d+)\s+don',q)
                if not m:raise DataMateError('Product frequency threshold missing')
                threshold=int(m[1])
            sql=("SELECT COUNT(*) AS value FROM (SELECT product_key FROM dm_sales WHERE order_date BETWEEN ? AND ? AND product_key IS NOT NULL AND search_key(product_name)!='phi phuc vu' GROUP BY product_key HAVING COUNT(DISTINCT order_id)>=?)")
            params=[p['start'],p['end'],threshold]
        elif task=='revenue_ratio':
            if where=='1=1':raise DataMateError('Missing numerator predicate')
            sql='SELECT round_half_up(SUM(CASE WHEN '+where+' THEN s.line_amount ELSE 0 END)*100.0/NULLIF(SUM(s.line_amount),0),2) AS value FROM dm_sales s WHERE s.order_date BETWEEN ? AND ?'
            params=[*filters,p['start'],p['end']]
        elif task=='rank_profit':
            sql=("SELECT MIN(s.product_name) AS name FROM dm_sales s WHERE s.order_date BETWEEN ? AND ? AND search_key(s.product_name)!='phi phuc vu' GROUP BY s.product_key ORDER BY SUM(s.line_amount-s.quantity*s.cost) DESC,s.product_key ASC LIMIT ?")
            params=[p['start'],p['end'],recipe['limit']]
        else:
            value={'revenue':'s.line_amount','quantity':'s.base_quantity','gross_profit':'s.line_amount-s.quantity*s.cost'}[task]
            money=task!='quantity'
            aggregate='COALESCE(SUM('+value+'),0)'
            if money:aggregate='round_half_up('+aggregate+',0)'
            sql='SELECT '+aggregate+' AS value FROM dm_sales s WHERE s.order_date BETWEEN ? AND ? AND '+where
            methods=sorted([(q.find(label),value) for label,value in [('tien mat','cash'),('chuyen khoan','transfer'),('cod','cod'),('vi dien tu','ewallet')] if label in q])
            if task=='revenue' and len(methods)>1:
                expressions=['round_half_up(COALESCE(SUM(CASE WHEN s.payment_method=? THEN s.line_amount ELSE 0 END),0),0) AS value_'+str(i) for i in range(len(methods))]
                sql='SELECT '+','.join(expressions)+' FROM dm_sales s WHERE s.order_date BETWEEN ? AND ?'
                params=[v for _,v in methods]+[p['start'],p['end']]
        kind='percent' if task=='revenue_ratio' else 'string' if task=='rank_profit' and recipe['limit']==1 else 'list' if task=='rank_profit' else 'number'
        if task=='revenue' and len(methods)>1:kind='list'
        channels=channel_filters(q,tools) if task in {'revenue','revenue_ratio'} else []
        if task=='revenue' and len(channels)>1:
            sql='SELECT '+','.join('round_half_up(COALESCE(SUM(CASE WHEN '+clause+' THEN s.line_amount ELSE 0 END),0),0) AS value_'+str(i) for i,(_,clause,_) in enumerate(channels))+' FROM dm_sales s WHERE s.order_date BETWEEN ? AND ?'
            params=[v for _,_,args in channels for v in args]+[p['start'],p['end']];kind='list'
        measure={'revenue':'revenue','quantity':'quantity','gross_profit':'gross_profit','revenue_ratio':'revenue','rank_profit':'gross_profit'}.get(task,'custom')
        # Check exactly the fields used, and never substitute an empty fact table.
        required=['line_amount','quantity','cost'] if task in {'gross_profit','rank_profit'} else ['base_quantity'] if task=='quantity' else ['product_key','order_id'] if task in {'unsold','product_order_threshold'} else ['line_amount']
        proof=tools._quality('dm_sales',p,required)
        if proof:raise DataMateError('Source quality not established: '+str(proof))
        return [plan_sql(recipe,sql,params,measure=measure,kind=kind)],checks
    if task=='debt_flow':
        from ..agent.planning import validate_plan,calculate
        plan={'action':'plan','measure':'debt','operation':'debt_flow','type':'list' if recipe['flow']=='both' else 'number','population':'debt',
              'source':'dm_orders_all','evidence_id':'Q','periods':ps,'filters':[],'flow':recipe['flow']}
        candidates=tools.executor.execute('SELECT DISTINCT o.customer_key,COALESCE(c.customer_name,o.customer_name) FROM dm_orders_all o LEFT JOIN dm_customers c ON c.customer_id=o.customer_id WHERE o.customer_key IS NOT NULL GROUP BY o.customer_key,COALESCE(c.customer_name,o.customer_name)')['rows']
        clean=lambda n:re.sub(r'\s*(?:-\s*\(?0[\d\s.-]+\)?|\(0[\d\s.-]+\))\s*$','',n).strip()
        names=[(k,clean(n)) for k,n in candidates if n and ' '+search_key(clean(n))+' ' in q]
        prefix=[]
        if names:
            names.sort(key=lambda x:len(x[1]),reverse=True)
            if len({k for k,n in names if search_key(n)==search_key(names[0][1])})!=1:
                raise DataMateError('Customer name matches multiple identities')
            plan['entities']={'customer':{'key':names[0][0]}}
            prefix=[{'action':'lookup','entity':'customer','text':names[0][1]}]
        accepted=validate_plan(plan,question,tools);res=calculate(accepted,tools)
        plan['query']={'sql':res['sql'],'parameters':res['parameters'],'columns':res['columns']}
        return [*prefix,plan],checks
    if task in {'debt_total','debt_count','debt_rank'}:
        day=ps[0]['end'] if len(ps)==1 else question.as_of_date
        caps=tools.semantic['capabilities']
        if not caps['payments'] or not caps['order_total']:raise DataMateError('Historical debt requires payment ledger')
        body=("SELECT o.order_id,o.order_date,o.customer_id,o.customer_name,o.customer_key,MAX(o.order_total-COALESCE(p.amount,0),0) AS debt FROM dm_orders_all o "
              "LEFT JOIN (SELECT order_id,SUM(amount) amount FROM dm_payments WHERE payment_date<=? GROUP BY order_id) p ON p.order_id=o.order_id "
              "WHERE o.order_date<=? AND o.status!='cancelled' AND o.payment_method NOT IN ('cod','marketplace')")
        if task=='debt_rank':
            dimension,identifier,name=customer_dimension(tools)
            sql=('SELECT COALESCE(MIN('+customer_display('c.'+quote(name))+'),MIN('+customer_display('d.customer_name')+'),d.customer_id) AS khach_hang,round_half_up(SUM(d.debt),0) AS so_tien_no '
                 'FROM ('+body+') d LEFT JOIN '+quote(dimension)+' c ON CAST(c.'+quote(identifier)+' AS TEXT)=d.customer_id WHERE d.debt>0 AND d.customer_key IS NOT NULL GROUP BY d.customer_key ORDER BY SUM(d.debt) DESC,d.customer_key ASC LIMIT ?')
            return [plan_sql(recipe,sql,[day,day,recipe['limit']],source='dm_orders_all',population='debt',kind='table')],checks
        groups=re.findall(r'(\d+)\s*[–-]\s*(\d+)',q)
        if groups:
            expressions=[];params=[]
            for low,high in groups:
                expressions.append('COALESCE(SUM(CASE WHEN julianday(?)-julianday(order_date) BETWEEN ? AND ? THEN debt ELSE 0 END),0)')
                params.extend([day,int(low),int(high)])
            above=re.search(r'(?:tren|hon|qua)\s+(\d+)\s+ngay',q)
            if above:expressions.append('COALESCE(SUM(CASE WHEN julianday(?)-julianday(order_date)>? THEN debt ELSE 0 END),0)');params.extend([day,int(above[1])])
            sql='SELECT '+','.join(e+' AS value_'+str(i) for i,e in enumerate(expressions))+' FROM ('+body+') WHERE debt>0'
            params.extend([day,day]);kind='list'
        else:
            aggregate='COUNT(DISTINCT customer_key)' if task=='debt_count' else 'round_half_up(COALESCE(SUM(debt),0),0)'
            sql='SELECT '+aggregate+' AS value FROM ('+body+') WHERE debt>0';params=[day,day];kind='number'
            above=re.search(r'(?:tren|hon|qua)\s+(\d+)\s+ngay',q)
            if above:sql+=' AND julianday(?)-julianday(order_date)>?';params.extend([day,int(above[1])])
        return [plan_sql(recipe,sql,params,source='dm_orders_all',kind=kind,population='debt')],checks
    if task=='inventory_value':
        day=question.as_of_date
        if not tools.semantic['capabilities']['inventory_snapshot']:
            if not tools.semantic['capabilities'].get('stock_ledger'):raise DataMateError('No stock snapshot or ledger')
            sql=('SELECT round_half_up(COALESCE(SUM(i.quantity*p.cost),0),0) AS value FROM '
                 '(SELECT product_id,quantity FROM (SELECT product_id,quantity,ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY occurred_at DESC,_dm_row DESC) AS n FROM dm_stock_ledger WHERE substr(occurred_at,1,10)<=?) WHERE n=1) i JOIN dm_products p ON p.product_id=i.product_id WHERE i.quantity>0')
            return [plan_sql(recipe,sql,[day],source='dm_stock_ledger',population='custom')],checks
        sql=('SELECT round_half_up(COALESCE(SUM(i.quantity*p.cost),0),0) AS value FROM dm_inventory i JOIN dm_products p '
             'ON i.product_id=p.product_id OR (i.product_id IS NULL AND norm(i.product_name)=norm(p.product_name)) WHERE i.snapshot_date=? AND i.quantity>0')
        return [plan_sql(recipe,sql,[day],source='dm_inventory',population='custom')],checks
    if task in {'inventory','inventory_units','expiry'}:
        day=question.as_of_date
        stock,params,b=lot_stock(tools,day)
        relation='('+stock+') i JOIN dm_products p ON p.product_id=i.product_id'
        product=recipe.get('product');where='1=1'
        if product:where='search_key(p.product_name)=?';params.append(search_key(product))
        if task=='inventory':
            sql='SELECT COALESCE(SUM(i.quantity),0) AS value FROM '+relation+' WHERE '+where;kind='number'
        elif task=='inventory_units':
            factor=recipe.get('large_unit')
            if not factor:raise DataMateError('Large inventory unit missing')
            # Parent aggregate is materialized once before applying conversion.
            sql=('SELECT CAST(v.quantity/c.factor AS INTEGER) AS whole,v.quantity-CAST(v.quantity/c.factor AS INTEGER)*c.factor AS remainder '
                 'FROM (SELECT p.product_id,SUM(i.quantity) AS quantity FROM '+relation+' WHERE '+where+' GROUP BY p.product_id) v JOIN dm_conversions c ON c.product_id=v.product_id WHERE search_key(c.large_unit)=?')
            params.append(search_key(factor));kind='list'
        else:
            if 'het hang' in q:
                sql='SELECT COUNT(*) AS value FROM (SELECT i.product_id FROM '+relation+' GROUP BY i.product_id HAVING SUM(i.quantity)=0)';kind='number'
            elif product and 'gan het han nhat' in q:
                sql='SELECT MIN(i.expiry_date) AS value FROM '+relation+' WHERE '+where+' AND i.quantity>0 AND i.expiry_date>=?';params.append(day);kind='date'
            else:
                bounds={'start':day,'end':max(p['end'] for p in ps)} if ps else None
                if bounds is None:raise DataMateError('Expiry window absent')
                where+=' AND i.quantity>0 AND i.expiry_date BETWEEN ? AND ?';params.extend([bounds['start'],bounds['end']])
                if 'tien von' in q:
                    sql='SELECT round_half_up(SUM(i.quantity*i.unit_cost),0) AS value FROM '+relation+' WHERE '+where;kind='number'
                elif re.search(r'(?:may|bao nhieu)\s+(?:mat hang|lo hang)',q):
                    sql='SELECT COUNT('+('DISTINCT i.product_id' if 'mat hang' in q else '*')+') AS value FROM '+relation+' WHERE '+where;kind='number'
                else:
                    sql=('SELECT MIN(p.product_name) AS name FROM '+relation+' WHERE '+where+' GROUP BY i.product_id ORDER BY MIN(i.expiry_date),i.product_id');kind='list'
        return [plan_sql(recipe,sql,params,source=b['clean_table'],population='custom',kind=kind)],checks
    if task=='delivery_count':
        b=next((b for b in tools.semantic['bindings'] if any(search_key(c) in {'ngay giao hang','ngay giao','ngay cho'} for c in b['unmapped'])),None)
        if b is None:raise DataMateError('No delivery source')
        rawdate=next(c for c in b['unmapped'] if search_key(c) in {'ngay giao hang','ngay giao','ngay cho'})
        sequence=next(c for c in b['unmapped'] if search_key(c) in {'lan giao','dot giao','luot giao'})
        table=quote(b['clean_table']);p=ps[0]
        sql='SELECT COUNT(*) AS value FROM (SELECT DISTINCT '+col(b,'order_id')+','+quote(sequence)+' FROM '+table+' WHERE iso_date('+quote(rawdate)+') BETWEEN ? AND ?)'
        return [plan_sql(recipe,sql,[p['start'],p['end']],source=b['clean_table'],population='custom')],checks
    if task in {'purchase_price','purchase_total'}:
        if len(ps)!=1:raise DataMateError('One purchase period required')
        p=ps[0];lines,header,relation,datecol,valid,h=purchase_relation(tools)
        quantity=col(lines,'quantity','l');price=col(lines,'price','l')
        value=col(lines,'line_amount','l') if 'line_amount' in lines['roles'] else quantity+'*'+price
        params=[p['start'],p['end']];where='iso_date('+datecol+') BETWEEN ? AND ? AND '+valid
        if task=='purchase_price':
            product=recipe.get('product')
            if not product:raise DataMateError('Purchase product not resolved')
            if 'product_name' in lines['roles']:where+=' AND search_key('+col(lines,'product_name','l')+')=?'
            else:where+=' AND EXISTS (SELECT 1 FROM dm_products p WHERE p.product_id=CAST('+col(lines,'product_id','l')+' AS TEXT) AND search_key(p.product_name)=?)'
            params.append(search_key(product))
            sql='SELECT round_half_up(1.0*SUM('+quantity+'*'+price+')/NULLIF(SUM('+quantity+'),0),0) AS value FROM '+relation+' WHERE '+where
            kind='number'
        elif 'nha cung cap' in q or 'cung cap nao' in q:
            vendorcol=find_column(header,['Nhà cung cấp','MaNhaCungCap','Mã NCC'])
            vendor=h+'.'+quote(vendorcol)
            sql='SELECT '+vendor+' AS name FROM '+relation+' WHERE '+where+' GROUP BY '+vendor+' ORDER BY SUM('+value+') DESC,'+vendor+' ASC LIMIT 1'
            kind='string'
        else:
            sql='SELECT round_half_up(COALESCE(SUM('+value+'),0),0) AS value FROM '+relation+' WHERE '+where
            kind='number'
        return [plan_sql(recipe,sql,params,source=lines['clean_table'],population='custom',kind=kind)],checks
    if task=='purchase_sales_difference':
        p=ps[0];lines,header,relation,datecol,valid,h=purchase_relation(tools)
        product=recipe.get('product')
        if not product:raise DataMateError('No product for purchase/sales difference')
        unit=lines['roles'].get('unit') or find_column(lines,['ĐVT nhập','Đơn vị nhập','DonViTinh'])
        if 'product_id' in lines['roles']:
            identity='c.product_id=CAST('+col(lines,'product_id','l')+' AS TEXT)'
            product_where='CAST('+col(lines,'product_id','l')+' AS TEXT) IN (SELECT product_id FROM dm_products WHERE search_key(product_name)=?)'
        else:
            identity='search_key(c.product_name)=search_key('+col(lines,'product_name','l')+')'
            product_where='search_key('+col(lines,'product_name','l')+')=?'
        factor='COALESCE((SELECT c.factor FROM dm_conversions c WHERE '+identity+' AND search_key(c.large_unit)=search_key(l.'+quote(unit)+') LIMIT 1),1)'
        purchased='SELECT COALESCE(SUM('+col(lines,'quantity','l')+'*'+factor+'),0) AS value FROM '+relation+' WHERE iso_date('+datecol+') BETWEEN ? AND ? AND '+valid+' AND '+product_where
        sold='SELECT COALESCE(SUM(base_quantity),0) AS value FROM dm_sales WHERE order_date BETWEEN ? AND ? AND search_key(product_name)=?'
        sql='SELECT a.value-b.value AS value FROM ('+purchased+') a CROSS JOIN ('+sold+') b'
        params=[p['start'],p['end'],search_key(product),p['start'],p['end'],search_key(product)]
        return [plan_sql(recipe,sql,params,source=lines['clean_table'],population='custom')],checks
    if task=='month_rank':
        if not ps:raise DataMateError('Month candidates absent')
        where=' OR '.join('order_date BETWEEN ? AND ?' for _ in ps)
        sql="SELECT substr(order_date,1,7) AS month FROM dm_sales WHERE ("+where+") GROUP BY month ORDER BY SUM(line_amount) DESC,month ASC LIMIT 1"
        params=[v for p in ps for v in (p['start'],p['end'])]
        return [plan_sql(recipe,sql,params,kind='date',measure='revenue')],checks
    if task=='net_profit':
        if len(ps)!=1:raise DataMateError('One net-profit period required')
        p=ps[0]
        expense=next((b for b in tools.semantic['bindings'] if 'chi_phi' in b['table'] or 'chiphi' in b['table']),None)
        if not expense:raise DataMateError('No expense source')
        d=find_column(expense,['Ngày chi','NgayChi','NgayPhatSinh']);money=find_column(expense,['Số tiền','So tien','SoTien'])
        sql=('SELECT round_half_up((SELECT COALESCE(SUM(line_amount-quantity*cost),0) FROM dm_sales WHERE order_date BETWEEN ? AND ?)'
             '-(SELECT COALESCE(SUM(CASE WHEN instr('+quote(money)+",'.')>0 AND instr("+quote(money)+",',')=0 THEN parse_number_locale("+quote(money)+",'vi') ELSE parse_number("+quote(money)+') END),0) FROM '+quote(expense['clean_table'])+' WHERE '+date_expression(tools,expense,d)+' BETWEEN ? AND ?),0) AS value')
        return [{'action':'schema','table':expense['table']},plan_sql(recipe,sql,[p['start'],p['end'],p['start'],p['end']],measure='gross_profit')],checks
    raise DataMateError('Custom template not implemented: '+task)
