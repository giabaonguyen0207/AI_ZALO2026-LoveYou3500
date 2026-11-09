"""TRAIN curation: text/schema driven recipes, never selected by question ID.

This is an offline annotation aid, not a new inference strategy. References
must only be read by the subsequent scoring stage, never by this module.
"""
from __future__ import annotations

import calendar
from copy import deepcopy
from datetime import date, timedelta
import re

from ..agent.planning import calculate, validate_plan
from ..agent.verification import question_key, question_periods
from ..data.transforms import search_key
from ..data.ingest import quote
from ..domain import DataMateError
from .seed import compile_recipe, inline_generated_ctes

VERSION = 1


def key(text):
    text = question_key(text)
    replacements = {'bua nay':'hom nay', 'bua qua':'hom qua', 'ngay hom nay':'hom nay',
                    'thang lien truoc':'thang truoc', 'thang lien trc':'thang truoc',
                    'th truoc do':'thang truoc', 'th truoc':'thang truoc',
                    'thang truoc do':'thang truoc', 'den heets':'den het',
                    'tienf':'tien', 'tieenf':'tien', 'tong cong':'tong cong',
                    'hown':'hon', 'cungf':'cung', 'duowc':'duoc', 'ngay nao':'ngay nao',
                    'trung binh mooix':'trung binh moi', 'trung binh moiix':'trung binh moi',
                    'ban dc':'ban duoc', 'bnhieu':'bao nhieu', 'bao nhju':'bao nhieu',
                    'co may':'co bao nhieu', 'tuan lien truoc':'tuan truoc',
                    'bill':'don','soos':'so','ddown':'don','hoom':'hom','thangs':'thang',
                    'mawtj':'mat','hangf':'hang','naof':'nao','xuaats':'xuat','its':'it',
                    'ban dduwowcj':'ban duoc','bans':'ban','bao nhieeu':'bao nhieu',
                    'ees':'e','dduwowcj':'duoc','toongr':'tong'}
    replacements.update({'phan trawm':'phan tram','phan tramf':'phan tram','tuwf':'tu','tro leen':'tro len',
                         'gaanf':'gan','la nhieu':'la bao nhieu','nhoms':'nhom','quys':'quy',
                         'binh quaan':'binh quan','mooix':'moi','ngayf':'ngay','naof':'nao','nhaats':'nhat',
                         'nawm':'nam','nhaapj':'nhap','bowts':'bot','trar':'tra','cuoois':'cuoi',
                         'maays':'may','toonf':'ton','nhieeuf':'nhieu','ddienj':'dien','ddieenj':'dien',
                         'thoaij':'thoai','nayf':'nay','gias':'gia','trowr leen':'tro len',
                         'tra bang tm':'tra bang tien mat','tra bang ck':'tra bang chuyen khoan',
                         'tien tuoi':'tien mat','khong can don bac si':'khong ke don','nam ngoai':'nam truoc'})
    for a,b in replacements.items():
        text = text.replace(a,b)
    return ' '+' '.join(text.split())+' '


def periods(question, tools):
    q = key(question.question)
    q = q.replace(' toi ', ' den ')
    recent=re.search(r'(\d+) ngay gan nhat',q)
    if recent:
        end=date.fromisoformat(question.as_of_date)
        return [{'kind':'range','start':(end-timedelta(days=int(recent[1])-1)).isoformat(),'end':end.isoformat()}]
    if '1/1 nam nay' in q or '01/01 nam nay' in q or '1/1 den' in q:
        return [{'kind':'range','start':question.as_of_date[:4]+'-01-01','end':question.as_of_date}]
    if 'tet' in q:
        match=re.search(r'tet\s+(?:am lich\s+)?(?:nam\s+)?(20\d\d)',q)
        year=int(match[1]) if match else int(question.as_of_date[:4])
        result=tools.executor.execute('SELECT MIN(date),MAX(date) FROM calendar WHERE (CAST(lunar_year AS INTEGER)=? AND CAST(lunar_month AS INTEGER)=12 AND CAST(lunar_day AS INTEGER)>=23) OR (CAST(lunar_year AS INTEGER)=? AND CAST(lunar_month AS INTEGER)=1 AND CAST(lunar_day AS INTEGER)<=10)',[year-1,year])
        start,end=result['rows'][0]
        if start is None or end is None:raise DataMateError('Calendar does not cover requested Tet')
        specs=[{'kind':'range','start':start,'end':end}]
        if 'nam truoc' in q:
            previous=tools.executor.execute('SELECT MIN(date),MAX(date) FROM calendar WHERE (CAST(lunar_year AS INTEGER)=? AND CAST(lunar_month AS INTEGER)=12 AND CAST(lunar_day AS INTEGER)>=23) OR (CAST(lunar_year AS INTEGER)=? AND CAST(lunar_month AS INTEGER)=1 AND CAST(lunar_day AS INTEGER)<=10)',[year-2,year-1])['rows'][0]
            if any(v is None for v in previous):raise DataMateError('Previous Tet calendar missing')
            specs.append({'kind':'range','start':previous[0],'end':previous[1]})
        elif 'truoc tet' in q:
            first=date.fromisoformat(start);n=(date.fromisoformat(end)-first).days+1
            specs.append({'kind':'range','start':(first-timedelta(days=n)).isoformat(),'end':(first-timedelta(days=1)).isoformat()})
        return specs
    q=q.replace('nam ngoai','nam truoc').replace('cung thang nam truoc','cung ky nam truoc')
    resolved=question_periods(q,question.as_of_date,tools.policy)
    if not resolved:
        # Incomplete year expressions use the split rulebook's nearest past period.
        match=re.search(r'\bthang\s+(\d{1,2})(?![\d/])',q)
        if match:
            month=int(match[1]);today=date.fromisoformat(question.as_of_date)
            year=today.year-(month>today.month)
            resolved=[{'start':date(year,month,1).isoformat(),'end':date(year,month,calendar.monthrange(year,month)[1]).isoformat()}]
    if len(resolved)==1 and re.search(r'3 thang (?:lien nhau|lien tiep|gan nhat)',q):
        last=date.fromisoformat(resolved[0]['start']);idx=last.year*12+last.month-1
        resolved=[]
        for i in (-2,-1,0):
            y,m=divmod(idx+i,12);m+=1
            resolved.append({'start':date(y,m,1).isoformat(),'end':date(y,m,calendar.monthrange(y,m)[1]).isoformat()})
    return [{'kind':'range','start':p['start'],'end':p['end']} for p in resolved]


def _f(column,value,op='eq',fold=False):
    return {'column':column,'op':op,'value':value,'evidence_id':'Q',**({'fold_accents':True} if fold else {})}


def _product(question, tools):
    cache=getattr(tools.catalog,'_sql_sft_products',None)
    if cache is None:
        result=tools.executor.execute('SELECT product_name FROM dm_products UNION SELECT product_name FROM dm_sales_all WHERE product_name IS NOT NULL GROUP BY product_name',remaining_seconds=30)
        if result['truncated']:
            raise DataMateError('Product dictionary truncated')
        cache=sorted({r[0] for r in result['rows'] if r[0]},key=lambda s:len(search_key(s) or ''),reverse=True)
        tools.catalog._sql_sft_products=cache
    text=' '+(search_key(question.question) or '')+' '
    found=[p for p in cache if ' '+search_key(p)+' ' in text]
    if not found:return None
    longest=search_key(found[0]);equivalent=[p for p in found if search_key(p)==longest]
    # Normalized spelling matches are accepted only after exact entity lookup.
    return equivalent[0]


def annotate(question, tools):
    """Return a fixed proposal before any reference is decoded."""
    q=key(question.question);ps=periods(question,tools)
    recipe={'periods':ps}
    product=_product(question,tools)
    percent=bool(re.search(r'%\s*(?:[,?.!]|$|so voi|tinh|vay|nhi|ha|do|z)|(?:phan tram|ti le|ty le|ti trong|ty trong)',q) or
                 '%' in q and (not product or re.search(r'\b(?:chiem|bang|tang|giam|chenh|hon|kem)\b',q)))
    k=re.search(r'\b(?:top|liet ke|cho (?:toi|minh))\s+(\d+)\b',q)
    if not k:k=re.search(r'(?<![\d/])(\d{1,2})\s+(?:khach|mat hang|san pham|hang hoa|best seller)\b',q)
    limit=int(k[1]) if k else 1
    numeric_threshold=re.search(r'(?:it nhat|tu|>=|≥)\s*(\d+)\s+don',q)
    money='doanh thu' in q or 'tien' in q or 'chi tieu' in q or 'gia tri' in q
    if any(t in q for t in ('dia chi','nha o','o duong','duong hai ba trung','gioi tinh','khach nu','khach nam','ly do tra','nguyen nhan tra','tra hang vi','hay khen','tan tam','le phep','chu dao','lanh nghe','tay nghe','nen uong')):
        recipe.update(task='missing_attribute')
    elif any(t in q for t in ('han su dung','het han','gan het han','qua han','han dung')):
        recipe.update(task='expiry')
    elif any(t in q for t in ('no them','thu no','tra no','thu hoi','tra bot no')):
        recipe.update(task='debt_flow',flow='both' if 'no them' in q and ('tra' in q or 'thu no' in q) else 'added' if 'no them' in q else 'paid')
    elif any(t in q for t in ('con no','cong no','dang no','khach no','no nhieu','no qua','chua tra het no','con thieu','tuoi no','khoan no')):
        task='debt_rank' if (k or 'nhieu nhat' in q or 'nhieu tien nhat' in q) else 'debt_count' if re.search(r'(?:so|may|bao nhieu) khach',q) else 'debt_total'
        recipe.update(task=task,date=ps[0]['end'] if len(ps)==1 else question.as_of_date,limit=limit)
    elif any(t in q for t in ('ton kho','trong kho','con ton','kho con','con hang','ton cuoi')) or product and ('cuoi ngay' in q or 'cuoi hom nay' in q):
        task='inventory_value' if ('gia tri' in q or 'tien von' in q or 'gia von' in q and not product) else 'inventory_units' if any(t in q for t in ('nguyen','cuon le','vien le','chai le','chan')) else 'inventory'
        recipe.update(task=task,date=question.as_of_date)
        if product:recipe['product']=product
    elif 'nhap' in q and any(t in q for t in ('binh quan','trung binh')):
        recipe.update(task='purchase_price')
    elif 'nhap tru' in q or 'nhap tr' in q:
        recipe.update(task='purchase_sales_difference')
    elif 'nhap' in q and any(t in q for t in ('tien','gia tri','nha cung cap','chenh','tru ban')):
        recipe.update(task='purchase_total')
    elif any(t in q for t in ('gia ban','don gia','ban voi gia','ban gia','gia bao nhieu','gia la bao nhieu')) and not percent:
        recipe.update(task='historical_price',date=ps[0]['start'] if ps else question.as_of_date)
    elif any(t in q for t in ('dot giao','lan giao','dot hang')):
        recipe.update(task='delivery_count')
    elif 'khong ban duoc' in q or re.search(r'\b(?:hang|mon) e\b',q) or 'bi e' in q or 'khong ai mua' in q or 'khong ban ra duoc' in q or 'ban khong duoc' in q:
        recipe.update(task='unsold')
    elif 'lai rong' in q or 'loi nhuan rong' in q:
        recipe.update(task='net_profit')
    elif re.search(r'\bhuy\b',q):
        recipe.update(task='cancel_ratio' if percent else 'cancel_count')
        if any(t in q for t in ('vi giao','ly do','nguyen nhan')):recipe.update(task='cancel_reason_count',reason_text='giao')
    elif 'ca sang' in q and 'ca chieu' in q and percent:
        recipe.update(task='shift_growth')
    elif 'khach' in q and numeric_threshold:
        recipe.update(task='customer_percent' if percent else 'customers_with_orders',threshold=int(numeric_threshold[1]))
    elif any(t in q for t in ('tang lien tuc','luon cao hon','thang sau co luon','xu huong doanh thu')):
        recipe.update(task='continuous_growth')
    elif 'thang nao' in q and any(t in q for t in ('nhieu tien nhat','cao nhat','ban dat nhat','lon nhat')):
        recipe.update(task='month_rank')
    elif ('ngay nao' in q or 'ngay ban dat nhat' in q) and ('cao nhat' in q or 'nhieu tien nhat' in q or 'ban dat nhat' in q):
        recipe.update(task='best_day')
    elif 'ngay' in q and any(t in q for t in ('so ngay','bao nhieu ngay','may ngay','nhieu ngay')) and any(t in q for t in ('>=','≥','tro len','it nhat')):
        recipe.update(task='day_count')
    elif ('don' in q and money and any(t in q for t in ('tro len','it nhat','>=','≥'))) and not percent:
        recipe.update(task='order_revenue_threshold')
    elif ('nhan vien' in q or 'ai la nguoi ban' in q) and any(t in q for t in ('gioi nhat','nhieu nhat','cao nhat')):
        recipe.update(task='rank_employees',limit=limit)
    elif ('khach' in q or 'ai mua' in q) and any(t in q for t in ('nhieu nhat','nhat','top','liet ke')) and not 'nhom' in q:
        recipe.update(task='rank_customers',limit=limit,by='orders' if 'don' in q and not money else 'revenue')
    elif any(t in q for t in ('ban chay','hot nhat','ban nhieu nhat','ban duoc nhieu nhat','chay nhat','best seller')):
        recipe.update(task='rank_products',limit=limit)
    elif 'loi nhat' in q or 'loi nhuan gop lon nhat' in q:
        recipe.update(task='rank_profit',limit=limit)
    elif any(t in q for t in ('trung binh','binh quan')) and 'don' in q and money:
        recipe.update(task='average_order_revenue')
    elif percent and (len(ps)==2 or any(t in q for t in ('so voi','so vs','so so','tang hay giam','tang truong','thay doi','hon hay kem','hon hoac kem','tang bao nhieu','tang may','giam may','giam bao nhieu','nhieu hay it hon'))):
        recipe.update(task='quantity_growth' if product or ('so luong' in q) else 'revenue_growth')
    elif percent and (money or any(t in q for t in ('nhom','ke don','hoat chat','chiem','ti trong','ty trong'))):
        recipe.update(task='revenue_ratio')
    elif (any(t in q for t in ('ghi lan luot','theo thu tu','lan luot la','moi thang')) or len(ps)>1 and 'thang' in q) and money:
        recipe.update(task='revenue_series')
    elif ('mat hang' in q or 'san pham' in q) and 'don' in q and any(t in q for t in ('it nhat','tro len','≥','>=')):
        recipe.update(task='product_order_threshold')
    elif any(t in q for t in ('lai gop','loi gop')):
        recipe.update(task='rank_profit' if 'nhat' in q else 'gross_profit',limit=limit)
    elif re.search(r'\b(?:so|bao nhieu|may|nhieu) don\b',q) or 'don hang' in q:
        recipe.update(task='orders')
    elif product and not money or (any(t in q for t in ('so luong','size m','size l')) or any(re.search(r'\b(?:bao nhieu|may|so)\s+'+re.escape(search_key(u))+r'\b',q) for u in tools.semantic.get('domains',{}).get('base_unit',[]))) and not money:
        recipe.update(task='quantity')
    elif money:
        recipe.update(task='revenue')
    else:
        raise DataMateError('Unclassified question: '+q)
    if product:recipe['product']=product
    if recipe['task'] in {'day_count','order_revenue_threshold'}:
        match=re.search(r'(?:tu|it nhat|>=|≥|dat)\s*([\d.,]+)',q)
        if not match:raise DataMateError('No numeric threshold')
        recipe['threshold']=int(match[1].replace('.','').replace(',',''))
    if recipe['task']=='quantity' and not product:
        unit=next((u for u in tools.semantic.get('domains',{}).get('base_unit',[]) if re.search(r'\b(?:bao nhieu|may|so)\s+'+re.escape(search_key(u))+r'\b',q)),None)
        if unit:recipe['unit']=unit
    size=re.search(r'\bsize\s+([ml])\b',q)
    if size:recipe['size']=size[1].upper()
    if recipe['task']=='inventory_units' and product:
        options=tools.executor.execute('SELECT DISTINCT large_unit FROM dm_conversions WHERE search_key(product_name)=?',[search_key(product)])['rows']
        matches=[u[0] for u in options if re.search(r'(?:may|bao nhieu)\s+'+re.escape(search_key(u[0]))+r'\b|\b'+re.escape(search_key(u[0]))+r'\s+(?:chan|nguyen)',q)]
        if len(matches)!=1:raise DataMateError('Inventory large unit not uniquely specified')
        recipe['large_unit']=matches[0]
    if not percent:
        if 'ca sang' in q:recipe['shift']='morning'
        elif 'ca chieu' in q:recipe['shift']='afternoon'
    if recipe['task']=='product_order_threshold' and numeric_threshold:recipe['threshold']=int(numeric_threshold[1])
    recipe['question_key']=q
    if recipe['task']=='quantity_growth' and len(recipe['periods'])!=2 and product:
        from .sql_templates import price_effect_periods
        recipe=price_effect_periods(recipe,question,tools)
    return recipe


def compile_proposal(recipe,question,tools):
    task=recipe['task'];r=deepcopy(recipe);r.pop('question_key',None)
    # A real check action supplies e0. Do not teach a broken plan preceding
    # an unanswerable final, and do not infer missingness from a gold label.
    q=recipe['question_key']
    if task=='revenue_series' and any(t in q for t in ('zalo','tai quay','dien thoai','tien mat','cod','chuyen khoan')):
        from .sql_templates import compile_custom
        return compile_custom({**recipe,'task':'revenue'},question,tools)
    required=[]
    if 'nhom' in q and 'tuoi no' not in q and (not recipe.get('product') or 'nhom hang' in q):
        group='customer_group' if 'khach' in q else 'product_group'
        from .sql_templates import dimension_filter
        try:dimension_filter(tools,group,q,identity='customer' if group=='customer_group' else 'product')
        except DataMateError:required.append(group)
    if 'ke don' in q:required.append('prescription')
    if task in {'customers_with_orders','customer_percent','rank_customers'}:required.append('customer_identity')
    for p in r.get('periods',[])[:1] or [None]:
        proof=tools.check(question.question,p,plan={'requirements':required,'measure':'custom'})
        if proof['availability']=='missing':
            missing_plan={'action':'plan','measure':'custom','operation':'custom','type':'number','population':'sales','source':'dm_sales',
                          'evidence_id':'Q','periods':r.get('periods',[]),'filters':[],'requirements':required}
            return [missing_plan,{'action':'final','type':'unanswerable','evidence_id':'e0'}],[{'tool_missing_evidence':proof}]
    if task in {'revenue','revenue_ratio','quantity','gross_profit','rank_profit','unsold','product_order_threshold','debt_total','debt_count','debt_flow','inventory_value','delivery_count','purchase_total','purchase_price','net_profit','month_rank','purchase_sales_difference','shift_growth','rank_employees','rank_customers','average_order_revenue','debt_rank'}:
        from .sql_templates import compile_custom
        return compile_custom(recipe,question,tools)
    if task in {'inventory','inventory_units','expiry'} and any('purchase_id' in b['roles'] and ('lot_id' in b['roles'] or 'expiry_date' in b['roles']) for b in tools.semantic['bindings']):
        from .sql_templates import compile_custom
        return compile_custom(recipe,question,tools)
    if task=='expiry':
        from .sql_templates import compile_custom
        return compile_custom(recipe,question,tools)
    if task in {'missing_attribute','expiry','debt_flow','debt_count','debt_total','inventory_value','purchase_price','purchase_total','delivery_count','unsold','net_profit','revenue_ratio','product_order_threshold'}:
        raise DataMateError('Pending compiler: '+task)
    plan,prefixes,checks=compile_recipe(r,question,tools)
    filters=[];q=recipe['question_key']
    if task in {'revenue','revenue_ratio','gross_profit'}:
        for field in ('product_group','customer_group'):
            values=tools.semantic.get('domains',{}).get(field,[])
            found=sorted([v for v in values if ' '+search_key(v)+' ' in q.replace('?',' ').replace(',',' ')],key=len,reverse=True)
            if found and 'nhom' in q:
                filters.append(_f(field,found[0],fold=True))
        if 'ke don' in q:
            filters.append(_f('prescription',None,'not_true') if re.search(r'(?:khong|hok|ko)\s+ke don',q) else _f('prescription',1))
    if filters:plan['filters'].extend(filters)
    if not plan.get('query') or filters:
        accepted=validate_plan(plan,question,tools)
        first=accepted['resolved_periods'][0] if accepted['resolved_periods'] else None
        proof=tools.check(question.question,{'kind':'range','start':first['start'],'end':first['end']} if first else None,plan=accepted)
        if proof['availability']=='missing':
            return [*prefixes,plan,{'action':'final','type':'unanswerable','evidence_id':'e0'}],checks
        result=calculate(accepted,tools)
        if result.get('availability')=='missing':
            return [*prefixes,plan,{'action':'final','type':'unanswerable','evidence_id':'e0'}],checks
        if 'sql' not in result:raise DataMateError('Calculation returned no SQL')
        sql=trusted_inline(result['sql'])
        rendered=tools.executor.execute(sql,result.get('parameters',[]))
        if rendered['rows']!=result['rows']:raise DataMateError('Compiler rewrite changed results')
        plan['query']={'sql':sql,'parameters':result.get('parameters',[]),'columns':rendered['columns']}
    return [*prefixes,plan],checks


def trusted_inline(sql):
    """Rewrite only internally compiled CTEs, including comma cross joins."""
    if not sql.startswith('WITH '):return sql
    names=re.findall(r'(?:WITH |\),\s*)(\w+) AS \(',sql)
    for name in names:
        sql=re.sub(r',\s*'+re.escape(name)+r'(?!\s+AS\s*\()(?=\s|,|$)',' CROSS JOIN '+name,sql)
    return inline_generated_ctes(sql)
