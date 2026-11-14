"""Model-proposed question plans and composable, parameterized calculations.

The model interprets language. Code validates the plan and its data dependencies,
and compiles supported operations. No business-specific metric definitions live
here. Citation checks establish provenance, not semantic correctness.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import re

from ..data.profiling import quote
from ..data.rules import resolve_period
from ..data.transforms import search_key
from ..execution_audit import audit_execution
from ..domain import ANSWER_TYPES, DataMateError
from ..data.interpretation import evidence_in

MEASURES = {'revenue':'line_amount', 'quantity':'base_quantity', 'orders':'order_id',
            'gross_profit':None, 'price':'price', 'debt':None, 'inventory':None, 'custom':None}
OPERATIONS = {'sum', 'count', 'rank', 'ratio', 'growth', 'series', 'day_count', 'best_day', 'custom',
              'average','order_count','continuous_growth','shift_growth','debt_flow','inventory','value'}
POPULATIONS = {'sales', 'all', 'debt', 'custom'}
FILTER_OPERATORS = {'eq','ne','gt','ge','lt','le','in','not_in','contains','endswith','is_null','not_null','not_true','shift'}
PLAN_FIELDS = {'action','reason','measure','operation','type','population','evidence','periods','filters','numerator_filters',
               'series_filters','output_order','date','source','requirements','group_by','display','limit','threshold','joins',
               'query','evidence_id','entities','unit','flow','large_unit'}


def source_schema(catalog):
    schema = {t['name']:set(t['columns']) for t in catalog.tables}
    schema.update({t['name']:set(t['columns']) for t in catalog.semantic['tables']})
    for binding in catalog.semantic['bindings']:
        schema[binding['clean_table']] = schema[binding['table']] | {'_dm_issues','_dm_status','_dm_source_table'}
    return schema


def column_ref(column, source, schema):
    if isinstance(column,dict) and set(column)=={'table','column'}:
        table,name=column['table'],column['column']
    elif isinstance(column,str):
        table, name = column.rsplit('.',1) if '.' in column else (source,column)
    else:
        raise DataMateError('column cần chuỗi tên cột')
    if table not in schema or name not in schema[table]:
        raise DataMateError('Cột không tồn tại trong nguồn đã chọn: ' + str(column))
    return table, name


def _resolve_plan_period(spec,as_of_date,policy):
    # Older models sometimes repeat the anchor or resolved date. These are
    # assertions to check, never alternate anchors and never silently ignored.
    spec=deepcopy(spec)
    if 'as_of_date' in spec and spec.pop('as_of_date')!=as_of_date:
        raise DataMateError('period.as_of_date khác mốc của câu hỏi')
    asserted_date=spec.pop('date',None) if spec.get('kind') in {'today','yesterday'} else None
    resolved=resolve_period(spec,as_of_date,policy)
    if asserted_date is not None and asserted_date!=resolved['start']:
        raise DataMateError('period.date mâu thuẫn kind theo as_of_date')
    return resolved


def validate_plan(action, question, tools, readme=''):
    if action.get('action') != 'plan':
        raise DataMateError('Cần action plan trước khi tính/chốt để ghi rõ đại lượng, kỳ, điều kiện và type')
    plan = deepcopy(action)
    for field in ('filters','numerator_filters','series_filters'):
        if field in plan and not isinstance(plan[field],list):raise DataMateError(field+' cần list')
    if set(plan)-PLAN_FIELDS:
        raise DataMateError('Plan chưa hỗ trợ các trường này, không tự bỏ qua: '+str(sorted(set(plan)-PLAN_FIELDS))
                            + '. Trường hợp lệ: '+', '.join(sorted(PLAN_FIELDS)))
    for field, choices in (('measure', MEASURES), ('operation', OPERATIONS),
                           ('type', ANSWER_TYPES - {'abstain','unanswerable'}), ('population', POPULATIONS)):
        if not isinstance(plan.get(field), str) or plan[field] not in choices:
            raise DataMateError(f'plan.{field}={plan.get(field)!r}: cần một chuỗi trong '+', '.join(sorted(choices)))
    if plan.get('evidence_id') == 'Q' and 'evidence' not in plan:
        plan['evidence'] = question.question
    elif 'evidence_id' in plan and plan['evidence_id'] != 'Q':
        raise DataMateError('plan.evidence_id chỉ hỗ trợ Q = câu hỏi gốc')
    if not evidence_in(plan.get('evidence'), question.question):
        raise DataMateError('plan.evidence cần trích đúng phần câu hỏi xác định đại lượng')
    if plan['population']=='custom' and plan['operation'] not in {'custom','inventory'}:
        raise DataMateError('Population custom cần SQL custom có căn cứ; calculate dùng sales hoặc all với điều kiện rõ')
    for field,operations in {'flow':{'debt_flow'},'large_unit':{'inventory'},'unit':{'sum','count','rank','ratio','growth','series','day_count','best_day','average','order_count','continuous_growth','shift_growth'}}.items():
        if plan.get(field) is not None and plan['operation'] not in operations:raise DataMateError(field+' không được operation này thực thi')
    if not isinstance(plan.get('requirements',[]),list) or any(x not in set(tools.semantic['capabilities']) | {'debt','inventory','payment_method'} for x in plan.get('requirements',[])):
        raise DataMateError('requirements chỉ gồm tên capability của catalog')
    default = 'dm_orders' if plan['measure'] in {'orders','debt'} else 'dm_sales'
    if plan['measure']=='inventory':default='dm_inventory'
    if plan['population'] in {'all','debt'} and default in {'dm_orders','dm_sales'}:
        default += '_all'
    plan.setdefault('source', default)
    schema = source_schema(tools.catalog)
    if plan['source'] not in schema:
        raise DataMateError('plan.source không có trong catalog')
    if plan['population'] in {'all','debt'} and plan['source'] in {'dm_sales','dm_orders'}:
        raise DataMateError('population all/debt cần nguồn *_all; nguồn sales đã bỏ trạng thái')
    if plan['population'] == 'sales' and plan['source'] in {'dm_sales_all','dm_orders_all'}:
        raise DataMateError('population sales dùng dm_sales/dm_orders đã áp policy hiện tại')
    periods = plan.get('periods', [])
    if not isinstance(periods, list) or len(periods) > 24:
        raise DataMateError('plan.periods cần list tối đa 24 kỳ theo thứ tự câu hỏi')
    plan['resolved_periods'] = []
    for index, period in enumerate(periods):
        if not isinstance(period, dict):
            raise DataMateError(f'plan.periods[{index}] cần object có kind, không phải chuỗi/list')
        try:
            plan['resolved_periods'].append(_resolve_plan_period(period,question.as_of_date,tools.policy))
        except (DataMateError, KeyError, TypeError, ValueError) as exc:
            raise DataMateError(f'plan.periods[{index}]: {exc}') from exc
    op=plan['operation']
    if plan.get('date') is not None and op not in {'custom','inventory'}:raise DataMateError('date không được operation này thực thi; dùng periods')
    if op=='count' and plan['measure']!='orders':raise DataMateError('count đếm đơn với measure orders; đếm nhóm theo ngưỡng dùng day_count/order_count, phép khác dùng custom')
    if op in {'sum','count','rank','ratio','best_day','day_count','average','order_count','shift_growth','debt_flow','value'} and len(periods)!=1:
        raise DataMateError(f'{op} thực thi đúng một kỳ; không bỏ qua kỳ dư. So hai kỳ dùng growth, nhiều kết quả dùng series')
    if plan.get('numerator_filters') and op!='ratio':raise DataMateError('numerator_filters chỉ được thực thi bởi ratio; dùng filters/series_filters đúng mục đích')
    for field in ('group_by','display','limit'):
        if plan.get(field) is not None and op not in {'rank','average'}:
            if op=='best_day':
                if field=='limit' and type(plan[field]) is int and plan[field]==1:continue
                if field!='limit' and column_ref(plan[field],plan['source'],schema)==(plan['source'],'order_date'):continue
            raise DataMateError(f'plan.{field} không được {op} thực thi; chọn rank/average/custom hoặc sửa plan, không âm thầm bỏ trường')
    if op=='average' and (plan.get('display') is not None or plan.get('limit') is not None):
        raise DataMateError('average chỉ dùng group_by để lấy trung bình tổng theo nhóm; display/limit không được thực thi')
    if plan.get('threshold') is not None and op not in {'day_count','order_count'}:raise DataMateError('threshold chỉ dùng sau tổng hợp day_count/order_count')
    if op in {'day_count','order_count'} and any(p.get('column')=='line_amount' for p in plan.get('filters',[]) if isinstance(p,dict)):
        raise DataMateError('Ngưỡng theo ngày/đơn đặt ở threshold sau tổng hợp; không lọc line_amount từng dòng trước tổng hợp')
    if op=='growth' and len(periods)==2 and periods[1].get('kind')=='previous_month_to_date' and periods[0].get('kind')!='this_month':
        raise DataMateError('previous_month_to_date neo theo as_of_date; tháng cụ thể cần month_offset anchor đúng kỳ hỏi hoặc month cụ thể')
    if plan['operation'] == 'growth' and len(periods) != 2:
        raise DataMateError('growth cần hai kỳ: kỳ hỏi, kỳ gốc; tháng đang chạy phải giải rõ cùng kỳ')
    if plan['operation'] == 'ratio' and not isinstance(plan.get('numerator_filters'), list):
        raise DataMateError('ratio cần numerator_filters; tỷ trọng khác tăng trưởng giữa hai kỳ')
    if plan['operation'] in {'ratio','growth'} and plan['type'] != 'percent':
        raise DataMateError('Tỷ lệ/tăng trưởng phần trăm cần type percent')
    if plan['operation'] == 'series' and plan['type'] != 'list':
        raise DataMateError('series cần type list theo thứ tự periods')
    if plan['operation'] == 'best_day' and plan['type'] != 'date':
        raise DataMateError('best_day cần type date')
    if plan['operation'] in {'sum','count','day_count','order_count','average','value'} and plan['type'] != 'number':
        raise DataMateError('Phép tổng/đếm cần type number; nhiều thành phần dùng series hoặc SQL custom')
    if 'date' in plan:
        date.fromisoformat(plan['date'])
    if 'query' in plan:
        query=plan['query']
        if not isinstance(query,dict) or set(query)-{'sql','parameters','columns'} or not isinstance(query.get('sql'),str):
            raise DataMateError('query cần sql, parameters và columns tùy chọn')
        if not isinstance(query.get('parameters',[]),list) or any(type(v) not in {str,int,float,bool,type(None)} for v in query.get('parameters',[])):
            raise DataMateError('query.parameters cần list giá trị đơn')
        if 'columns' in query and (not isinstance(query['columns'],list) or not query['columns'] or not all(isinstance(c,str) for c in query['columns'])):
            raise DataMateError('query.columns cần list tên cột không rỗng')
    plan.setdefault('filters', [])
    entities=plan.get('entities',{})
    if not isinstance(entities,dict) or set(entities)-{'product','customer'}:raise DataMateError('entities chỉ gồm product/customer tên hoặc key từ lookup')
    for entity,value in entities.items():
        if op=='inventory' and entity=='product':continue
        key=tools._entity(entity,value)
        if entity+'_key' not in schema[plan['source']]:raise DataMateError('Nguồn không có khóa '+entity+'; chọn facts phù hợp/SQL có join')
        predicate={'column':entity+'_key','op':'eq','value':key,'evidence_id':'Q','evidence':question.question}
        if not any(all(f.get(k)==predicate[k] for k in ('column','op','value')) for f in plan['filters'] if isinstance(f,dict)):plan['filters'].append(predicate)
    if 'unit' in plan:
        if not isinstance(plan['unit'],str) or not plan['unit'].strip():raise DataMateError('unit cần tên đơn vị cơ bản')
        predicate={'column':'base_unit','op':'eq','value':plan['unit'],'evidence_id':'Q','evidence':question.question}
        if not any(all(f.get(k)==predicate[k] for k in ('column','op','value')) for f in plan['filters'] if isinstance(f,dict)):
            plan['filters'].append(predicate)
    context = '\n'.join([question.question,tools.rulebook,readme])
    series=plan.get('series_filters',[])
    if not isinstance(series,list) or len(series)>24 or series and plan['operation']!='series':
        raise DataMateError('series_filters chỉ dùng cho series, tối đa 24 nhóm theo thứ tự câu hỏi')
    if series and len(periods) not in {1,len(series)}:
        raise DataMateError('series_filters cần một kỳ dùng chung hoặc mỗi nhóm một kỳ')
    if 'output_order' in plan:
        count=len(series) if series else len(periods)
        if plan['operation']!='series' or not isinstance(plan['output_order'],list) or len(plan['output_order'])!=count or not all(isinstance(v,str) for v in plan['output_order']):
            raise DataMateError('output_order cần list nhãn tương ứng các phần tử series')
    groups={k:plan[k] for k in ('filters','numerator_filters') if k in plan}
    groups.update({'series_filters['+str(i)+']':p for i,p in enumerate(series)})
    for key,predicates in groups.items():
        if not isinstance(predicates, list) or len(predicates) > 20:
            raise DataMateError(key + ' cần list điều kiện tối đa 20 phần tử')
        for index, predicate in enumerate(predicates):
            path = f'plan.{key}[{index}]'
            if not isinstance(predicate, dict):
                raise DataMateError(path + ' cần object column/op/value/evidence_id hoặc evidence')
            if set(predicate)-{'column','op','value','evidence','evidence_id','fold_accents'}:
                raise DataMateError(path + ': điều kiện có tham số chưa hỗ trợ, không tự bỏ qua')
            if predicate.get('evidence_id') == 'Q' and 'evidence' not in predicate:
                predicate['evidence'] = question.question
            elif 'evidence_id' in predicate and predicate['evidence_id'] != 'Q':
                raise DataMateError('Điều kiện chỉ hỗ trợ evidence_id Q hoặc trích nguồn nguyên văn')
            if 'fold_accents' in predicate and type(predicate['fold_accents']) is not bool:
                raise DataMateError('fold_accents cần boolean')
            try:
                column_ref(predicate.get('column'), plan['source'], schema)
            except DataMateError as exc:
                raise DataMateError(path + '.column: ' + str(exc)) from exc
            if not isinstance(predicate.get('op'), str) or predicate['op'] not in FILTER_OPERATORS:
                raise DataMateError(path + '.op cần một chuỗi trong '+', '.join(sorted(FILTER_OPERATORS))
                                    + '; khoảng giá trị dùng hai filter ge và le')
            if not evidence_in(predicate.get('evidence'), context):
                raise DataMateError(path + ' cần căn cứ câu hỏi/README/rulebook; dùng evidence_id="Q" hoặc evidence trích nguồn: ' + str(predicate.get('column')))
            value=predicate.get('value');operator=predicate['op']
            if operator in {'in','not_in'} and (not isinstance(value,list) or not value or len(value)>100 or any(isinstance(v,(dict,list)) or v is None for v in value)):
                raise DataMateError(path+'.value cần list giá trị đơn không rỗng, không nhét SQL vào value')
            if operator not in {'is_null','not_null','not_true','in','not_in'} and (value is None or isinstance(value,(dict,list))):
                raise DataMateError(path+'.value cần một giá trị đơn')
            column_name=column_ref(predicate['column'],plan['source'],schema)[1]
            domain=tools.semantic.get('domains',{}).get(column_name,[])
            if domain and operator in {'eq','ne','in','not_in'}:
                values=value if operator in {'in','not_in'} else [value]
                canonical=[]
                for member in values:
                    matches=[v for v in domain if member==v or isinstance(member,str) and isinstance(v,str) and search_key(member)==search_key(v)]
                    canonical.append(matches[0] if len(matches)==1 else member)
                predicate['value']=canonical if operator in {'in','not_in'} else canonical[0]
        _check_contradictions(predicates,key)
    if plan['operation']=='ratio':
        _check_contradictions(plan['filters']+plan['numerator_filters'],'filters + numerator_filters')
        signatures=lambda fs:{(str(f['column']),f['op'],str(f.get('value'))) for f in fs}
        if plan['numerator_filters'] and signatures(plan['numerator_filters'])<=signatures(plan['filters']):
            raise DataMateError('Điều kiện riêng tử số đã nằm toàn bộ trong mẫu số; kiểm tra phạm vi để tránh tự chia cùng một tập')
    if plan['operation'] == 'rank':
        if type(plan.get('limit')) is not int or not 1 <= plan['limit'] <= 100:
            raise DataMateError('rank cần limit theo câu hỏi, trong 1..100')
        if plan['type'] not in ({'string','date'} if plan['limit']==1 else {'list'}):
            raise DataMateError('rank: limit=1 trả string; nhiều đối tượng trả list')
        for field in ('group_by','display'):
            column = plan.get(field,plan.get('group_by'))
            if not isinstance(column, str):
                raise DataMateError(f'plan.{field} cần một chuỗi tên cột; không dùng list/object')
            column_ref(column, plan['source'], schema)
    if plan['operation'] in {'day_count','order_count'} and type(plan.get('threshold')) not in (int,float):
        raise DataMateError('day_count cần ngưỡng số threshold')
    if op=='average':column_ref(plan.get('group_by'),plan['source'],schema)
    if op in {'continuous_growth','shift_growth'} and plan['type']!=('boolean' if op=='continuous_growth' else 'percent'):
        raise DataMateError(op+' cần type '+('boolean' if op=='continuous_growth' else 'percent'))
    if op=='continuous_growth' and len(periods)<2:raise DataMateError('continuous_growth cần ít nhất hai kỳ theo thứ tự')
    if op=='inventory' and (plan['measure']!='inventory' or plan['type']!=('list' if plan.get('large_unit') else 'number')):
        raise DataMateError('inventory cần measure inventory; có large_unit trả list, không có trả number')
    if op=='inventory':
        if plan['source'] not in {'dm_inventory','dm_stock_ledger'}:raise DataMateError('inventory cần nguồn kho, không dùng nguồn bán hàng')
        if set(entities)!={'product'} or plan['filters'] or plan.get('joins') or plan.get('query'):
            raise DataMateError('inventory cần entities.product và date; không bỏ qua filters/joins/query, nghiệp vụ khác dùng custom')
        if periods and (len(periods)!=1 or plan['resolved_periods'][0]['start']!=plan['resolved_periods'][0]['end']):
            raise DataMateError('inventory dùng ngày chốt date, không dùng khoảng phát sinh')
        if periods and plan.get('date') and plan['date']!=plan['resolved_periods'][0]['end']:
            raise DataMateError('inventory date và periods mâu thuẫn ngày chốt')
        plan.setdefault('date',plan['resolved_periods'][0]['end'] if periods else question.as_of_date)
    if op=='debt_flow' and (plan['measure']!='debt' or plan.get('flow') not in {'added','paid','both'} or plan['type']!=('list' if plan.get('flow')=='both' else 'number')):
        raise DataMateError('debt_flow cần measure debt, flow added/paid/both, both trả list còn lại trả number')
    if op=='debt_flow' and (plan['population']!='debt' or plan.get('joins') or any(f['column']!='customer_key' or f['op']!='eq' for f in plan['filters'])):
        raise DataMateError('debt_flow dùng population debt và entities.customer tùy chọn; không bỏ qua filter khác')
    if op=='debt_flow' and plan['source']!='dm_orders_all':raise DataMateError('debt_flow cần nguồn dm_orders_all cùng dm_payments')
    if op=='value' and plan['measure']!='price':raise DataMateError('value hiện hỗ trợ price; phép khác dùng custom')
    plan.setdefault('joins', [])
    if not isinstance(plan['joins'], list) or len(plan['joins']) > 6:
        raise DataMateError('joins cần list tối đa sáu quan hệ many-to-one')
    joined = {plan['source']}
    for join in plan['joins']:
        if not isinstance(join, dict) or join.get('table') not in schema or join['table'] in joined:
            raise DataMateError('Bảng nối không tồn tại hoặc bị lặp')
        if set(join)-{'table','left','right','evidence'}:
            raise DataMateError('Quan hệ nối có tham số chưa hỗ trợ')
        left = column_ref(join.get('left'),plan['source'],schema)
        right = column_ref(join.get('right'),join['table'],schema)
        if left[0] not in joined or right[0] != join['table']:
            raise DataMateError('joins phải nối từ nguồn đã có tới khóa bảng đích')
        remote = quote(right[1]); table = quote(right[0])
        duplicate = tools.executor.execute(f'SELECT COUNT(*) FROM (SELECT {remote} FROM {table} WHERE {remote} IS NOT NULL GROUP BY {remote} HAVING COUNT(*)>1)')['rows'][0][0]
        if duplicate:
            raise DataMateError('Khóa bảng đích không duy nhất; joins sẽ nhân số tiền/số lượng')
        joined.add(join['table'])
    for predicates in groups.values():
        for predicate in predicates:
            if column_ref(predicate['column'],plan['source'],schema)[0] not in joined:
                raise DataMateError('Điều kiện dùng bảng chưa có trong joins')
    for field in ('group_by','display'):
        if plan.get(field) is not None and column_ref(plan[field],plan['source'],schema)[0] not in joined:
            raise DataMateError(field+' dùng bảng chưa có trong joins')
    if op=='rank' and plan['type']=='date':
        _,column=column_ref(plan.get('display',plan['group_by']),plan['source'],schema)
        date_columns={'order_date','snapshot_date','expiry_date'}|{b['roles'][role] for b in tools.semantic['bindings'] for role in ('order_date','expiry_date') if role in b['roles']}
        if column not in date_columns:raise DataMateError('rank type date cần cột ngày đã giải; không chuyển tên đối tượng thành ngày')
    return plan


def _check_contradictions(predicates,label):
    equal={}
    for predicate in predicates:
        if predicate['op']!='eq':continue
        column=str(predicate['column']);value=predicate.get('value')
        if column in equal and equal[column]!=value:raise DataMateError(label+': điều kiện eq loại trừ nhau trên '+column+'; nhiều kết quả dùng series_filters')
        equal[column]=value


def _col(column, plan, schema):
    table, name = column_ref(column,plan['source'],schema)
    return quote(table)+'.'+quote(name)


def _filters(predicates, plan, schema, tools):
    clauses, params = [], []
    for item in predicates:
        column = _col(item['column'],plan,schema)
        op, value = item['op'], item.get('value')
        if item.get('fold_accents') and op in {'eq','ne','in','not_in'}:
            column=f'search_key({column})'
            if op in {'in','not_in'}:
                if not isinstance(value,list) or not all(isinstance(v,str) for v in value):
                    raise DataMateError('fold_accents cho in/not_in cần list chuỗi')
                value=[search_key(v) for v in value]
            elif isinstance(value,str):
                value=search_key(value)
            else:
                raise DataMateError('fold_accents chỉ dùng cho giá trị chuỗi')
        if op in {'is_null','not_null'}:
            clauses.append(column + (' IS NULL' if op == 'is_null' else ' IS NOT NULL'))
        elif op == 'not_true':
            clauses.append(f"COALESCE(LOWER(CAST({column} AS TEXT)),'') NOT IN ('1','true')")
        elif op == 'shift':
            interval = (tools.policy.shifts or {}).get(value)
            if not interval:
                raise DataMateError('Ca chưa được rulebook xác nhận')
            clauses.append(f'substr({column},12,5)>=? AND substr({column},12,5)<?')
            params.extend(interval)
        elif op in {'in','not_in'}:
            if not isinstance(value,list) or not value or len(value)>100:
                raise DataMateError('in/not_in cần list không rỗng tối đa 100 giá trị')
            clauses.append(column + (' IN (' if op=='in' else ' NOT IN (') + ','.join('?' for _ in value)+')')
            params.extend(value)
        elif op in {'contains','endswith'}:
            if not isinstance(value,str) or not value:
                raise DataMateError('contains/endswith cần chuỗi không rỗng')
            # Accent folding is used only when explicitly requested by the plan.
            expr = f'search_key({column})' if item.get('fold_accents') else f'LOWER({column})'
            value = search_key(value) if item.get('fold_accents') else value.lower()
            value = value.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')
            clauses.append(expr + " LIKE ? ESCAPE '\\'")
            params.append('%'+value+('%' if op=='contains' else ''))
        else:
            operator = {'eq':'=','ne':'!=','gt':'>','ge':'>=','lt':'<','le':'<='}[op]
            if value is None or isinstance(value,(dict,list)):
                raise DataMateError('So sánh cần giá trị đơn khác null; dùng is_null nếu cần')
            clauses.append(f'{column}{operator}?'); params.append(value)
    return (' AND '.join('('+c+')' for c in clauses) or '1=1'), params


@audit_execution
def calculate(plan, tools):
    """Compose common relational operations; unsupported concepts keep SQL."""
    schema = source_schema(tools.catalog)
    source, measure, operation = plan['source'],plan['measure'],plan['operation']
    if operation=='debt_flow':
        from .ledger import debt_flow
        return debt_flow(plan,tools)
    if operation=='inventory':
        product=plan['entities']['product'];product=product.get('key') if isinstance(product,dict) else product
        result=tools.metric({'name':'inventory_units' if plan.get('large_unit') else 'inventory',
            'product':product,'date':plan['date'],**({'large_unit':plan['large_unit']} if plan.get('large_unit') else {})})
        return {**result,'plan':plan}
    if measure not in {'revenue','quantity','orders','gross_profit','price'} or operation == 'custom':
        raise DataMateError('Phép tính này cần SQL có kế hoạch và kiểm tra nguồn; calculate chưa biểu diễn được')
    if source not in {'dm_sales','dm_sales_all','dm_orders','dm_orders_all'}:
        raise DataMateError('calculate cần facts giao dịch; nguồn khác dùng SQL có kiểm tra')
    if not tools.semantic['capabilities'].get('order_identity' if measure == 'orders' else 'sales'):
        raise DataMateError('Nguồn chưa ánh xạ được giao dịch; không coi bảng chuẩn hóa rỗng là số 0')
    required = ['order_id'] if measure == 'orders' else ['line_amount','quantity','cost'] if measure == 'gross_profit' else [MEASURES[measure]]
    if operation=='order_count':required+=['order_id']
    if operation=='average':
        group_table,group_column=column_ref(plan['group_by'],source,schema)
        if group_table==source:required+=[group_column]
    if operation=='value':required+=['quantity','base_quantity']
    if operation=='shift_growth':required+=['order_timestamp']
    if operation=='rank':
        group_table,group_column=column_ref(plan['group_by'],source,schema)
        if group_table==source and group_column not in {'customer_key','customer_id'}:
            required.append(group_column)
        if group_column in {'product_key','customer_key'}:
            capability='product_identity' if group_column=='product_key' else 'customer_identity'
            if not tools.semantic['capabilities'].get(capability):
                raise DataMateError('Chưa xác định được danh tính để xếp hạng; không coi bảng rỗng là không có đối tượng')
            if group_column=='product_key' and group_table==source:
                required.append(group_column)
    if measure == 'gross_profit' and not tools.policy.gross_profit_line_cost:
        raise DataMateError('Chưa xác nhận quy tắc giá vốn đơn vị bán trên dòng để tính lãi gộp')
    if plan['population'] == 'debt':
        raise DataMateError('Công nợ cần dòng tiền đến ngày chốt; không dùng tổng/snapshot tùy ý')
    periods = plan['resolved_periods']
    if not periods:
        raise DataMateError('calculate cần periods đã giải; không tự chọn toàn thời gian')
    all_predicates=plan['filters']+plan.get('numerator_filters',[])+[f for group in plan.get('series_filters',[]) for f in group]
    for predicate in all_predicates:
        table, column = column_ref(predicate['column'],source,schema)
        if table == source and predicate['op'] not in {'is_null','not_null','not_true'}:
            required.append(column)
    for period in periods:
        missing = tools._quality(source.removesuffix('_all'),period,list(set(required)),
                                 statuses='all' if plan['population'] == 'all' else None)
        if missing:
            return missing
    if measure=='price' and (operation!='value' or plan['population']!='all' or not tools.policy.price_all_states):
        raise DataMateError('Giá bán cần value/all và rulebook xác nhận trạng thái; công thức khác dùng custom')
    base = quote(source)
    for join in plan['joins']:
        target = join['table']
        right_table, right_col = column_ref(join['right'],target,schema)
        base += ' LEFT JOIN '+quote(target)+' ON '+_col(join['left'],plan,schema)+'='+quote(right_table)+'.'+quote(right_col)
    if measure == 'orders':
        aggregate = 'COUNT(DISTINCT '+_col('order_id',plan,schema)+')'
    else:
        value = (_col('line_amount',plan,schema)+'-'+_col('quantity',plan,schema)+'*'+_col('cost',plan,schema)) if measure == 'gross_profit' else _col(MEASURES[measure],plan,schema)
        aggregate = 'COALESCE(SUM('+value+'),0)'
    filtering, filter_params = _filters(plan['filters'],plan,schema,tools)
    if operation == 'rank' and column_ref(plan['group_by'],source,schema)[1]=='product_key' and tools.policy.exclude_service_product:
        filtering += ' AND search_key('+_col('product_name',plan,schema)+")!='phi phuc vu'"
    day = _col('order_date',plan,schema)
    # A parsed attribute that became NULL due to an error must not accidentally
    # satisfy NOT TRUE or be dropped by a dimension filter.
    checked=set()
    for predicate in all_predicates:
        table,column=column_ref(predicate['column'],source,schema)
        spans=' OR '.join(f'{day} BETWEEN ? AND ?' for _ in periods)
        args=[v for p in periods for v in (p['start'],p['end'])]
        if table!=source and predicate['op'] not in {'not_true','is_null','not_null'}:
            missing=tools.executor.execute(f'SELECT COUNT(*) FROM {base} WHERE ({spans}) AND '+_col(predicate['column'],plan,schema)+' IS NULL',args)['rows'][0][0]
            if missing:
                raise DataMateError('Thiếu thuộc tính/bản ghi danh mục cần lọc trong kỳ; không tính tổng một phần: '+table+'.'+column)
        if '_dm_issues' not in schema[table] or (table,column) in checked:
            continue
        checked.add((table,column))
        issues=quote(table)+'.'+quote('_dm_issues')
        issue_keys=['parse:'+column,'conflicting_dimension:'+column,'invalid_dimension:'+column]
        if column in {'product_group','prescription','base_unit'}:issue_keys+=['ambiguous_product_dimension']
        if column in {'customer_group','gender'}:issue_keys+=['ambiguous_customer_dimension']
        bad=tools.executor.execute(f'SELECT COUNT(*) FROM {base} WHERE ({spans}) AND ('+' OR '.join(f'instr({issues},?)>0' for _ in issue_keys)+')',[*args,*issue_keys])['rows'][0][0]
        if bad:
            raise DataMateError('Thuộc tính lọc có lỗi parse trong kỳ: '+table+'.'+column)
        if table==source and predicate['op'] in {'eq','contains','endswith','in'} and column in {
                'product_name','customer_name','product_key','customer_key','product_group','customer_group','base_unit','payment_method','status','gender'}:
            clause,values=_filters([predicate],plan,schema,tools)
            domain_table=source
            if column in {'product_group','customer_group','base_unit'}:
                domain_table='dm_customers' if column=='customer_group' else 'dm_products'
                if not tools.executor.execute('SELECT COUNT(*) FROM '+quote(domain_table))['rows'][0][0]:domain_table=source
                clause=clause.replace(quote(source)+'.',quote(domain_table)+'.')
            count=tools.executor.execute('SELECT COUNT(*) FROM '+quote(domain_table)+' WHERE '+clause,values)['rows'][0][0]
            if not count and column in {'product_key','product_name'} and predicate['op']=='eq':
                name_column='COALESCE(CAST(product_id AS TEXT),norm(product_name))' if column=='product_key' else 'product_name'
                count=tools.executor.execute('SELECT COUNT(*) FROM dm_products WHERE '+name_column+'=?',[predicate['value']])['rows'][0][0]
            if not count:
                raise DataMateError('Bộ lọc không khớp đối tượng/giá trị nào trong nguồn: '+column+'; dùng lookup/profile hoặc cột nhóm thật, không suy kết quả bằng 0')
    def total(period, more=None):
        where, more_params = _filters(more or [],plan,schema,tools)
        return f'SELECT {aggregate} AS value FROM {base} WHERE {day} BETWEEN ? AND ? AND {filtering} AND {where}', [period['start'],period['end'],*filter_params,*more_params]
    sql, params = total(periods[0])
    if operation=='value':
        if not plan.get('entities',{}).get('product'):
            raise DataMateError('value giá cần entities.product duy nhất, không lấy trung bình nhiều sản phẩm')
        sql=f'SELECT DISTINCT 1.0*'+_col('price',plan,schema)+'*'+_col('quantity',plan,schema)+'/NULLIF('+_col('base_quantity',plan,schema)+f',0) AS value FROM {base} WHERE {day} BETWEEN ? AND ? AND {filtering}'
        params=[periods[0]['start'],periods[0]['end'],*filter_params]
    elif operation in {'ratio','growth','shift_growth'}:
        places = tools.policy.percent_places
        if type(places) is not int:
            raise DataMateError('Chưa xác nhận số chữ số làm tròn phần trăm')
        if operation=='shift_growth':
            morning,mp=total(periods[0],[{'column':'order_timestamp','op':'shift','value':'morning'}])
            afternoon,ap=total(periods[0],[{'column':'order_timestamp','op':'shift','value':'afternoon'}])
            sql=f'WITH a AS ({afternoon}),m AS ({morning}) SELECT round_half_up((a.value-m.value)*100.0/NULLIF(m.value,0),?) FROM a,m'
            params=[*ap,*mp,places]
        elif operation == 'ratio':
            numerator, np = total(periods[0],plan['numerator_filters'])
            sql = f'WITH numerator AS ({numerator}), denominator AS ({sql}) SELECT round_half_up(numerator.value*100.0/NULLIF(denominator.value,0),?) FROM numerator,denominator'
            params = [*np,*params,places]
        else:
            previous, pp = total(periods[1])
            sql = f'WITH current AS ({sql}), previous AS ({previous}) SELECT round_half_up((current.value-previous.value)*100.0/NULLIF(previous.value,0),?) FROM current,previous'
            params += [*pp,places]
    elif operation == 'series':
        groups=plan.get('series_filters',[])
        pieces = [total(periods[0] if len(periods)==1 else periods[i],predicates) for i,predicates in enumerate(groups)] if groups else [total(p) for p in periods]
        sql = ' UNION ALL '.join('SELECT '+str(i)+' AS ordinal,value FROM ('+piece+')' for i,(piece,_) in enumerate(pieces))
        rounding=measure in {'revenue','gross_profit','price'} and tools.policy.money_places is not None
        sql = ('SELECT round_half_up(value,?) AS value FROM (' if rounding else 'SELECT value FROM (')+sql+') ORDER BY ordinal'
        params = [v for _,ps in pieces for v in ps]
        if rounding:params=[tools.policy.money_places,*params]
    elif operation=='continuous_growth':
        pieces=[total(p) for p in periods]
        ctes=','.join('p'+str(i)+' AS ('+s+')' for i,(s,_) in enumerate(pieces))
        sql='WITH '+ctes+' SELECT CASE WHEN '+' AND '.join(f'p{i}.value>p{i-1}.value' for i in range(1,len(pieces)))+' THEN 1 ELSE 0 END FROM '+','.join('p'+str(i) for i in range(len(pieces)))
        params=[value for _,ps in pieces for value in ps]
    elif operation in {'rank','best_day','day_count','order_count','average'}:
        if operation == 'best_day':
            group, display = day,day
        elif operation in {'day_count','order_count','average'}:
            group=day if operation=='day_count' else _col('order_id' if operation=='order_count' else plan['group_by'],plan,schema)
            display=group
        else:
            group = _col(plan['group_by'],plan,schema)
            display = 'COALESCE(MIN('+_col(plan.get('display',plan['group_by']),plan,schema)+'),CAST('+group+' AS TEXT))'
        body = f'SELECT {display} AS name,{aggregate} AS value FROM {base} WHERE {day} BETWEEN ? AND ? AND {filtering} AND {group} IS NOT NULL GROUP BY {group}'
        params = [periods[0]['start'],periods[0]['end'],*filter_params]
        if operation in {'day_count','order_count'}:
            sql = 'SELECT COUNT(*) FROM ('+body+') WHERE value>=?'
            params.append(plan['threshold'])
        elif operation=='average':
            places=tools.policy.money_places if measure in {'revenue','gross_profit'} else tools.policy.percent_places
            if places is None:raise DataMateError('Rulebook chưa xác nhận làm tròn trung bình')
            sql='SELECT round_half_up(AVG(value),?) AS value FROM ('+body+')'
            params=[places,*params]
        else:
            # Ties use the actual group key, independently of display spelling.
            body += f' ORDER BY value DESC,{group} ASC LIMIT ?'
            params.append(1 if operation=='best_day' else plan['limit'])
            sql = 'SELECT name FROM ('+body+')'
    elif operation not in {'sum','count'}:
        raise DataMateError('operation chưa hỗ trợ')
    if measure in {'revenue','gross_profit','price'} and plan['type'] in {'number','list'} and operation in {'sum','average','value'} and tools.policy.money_places is not None:
        sql='SELECT round_half_up(value,?) AS value FROM ('+sql+')'
        params=[tools.policy.money_places,*params]
    result=tools.executor.execute(sql,params)
    if (not result['rows'] and not (operation=='rank' and plan['type']=='list')) or any(v is None for row in result['rows'] for v in row):
        raise DataMateError('Kết quả không xác định/rỗng; kiểm tra mẫu số, điều kiện và dữ liệu')
    return {**result, 'verified':True, 'verification_scope':'validated_plan_compiled_query_and_source_quality',
            'answer_type':plan['type'], 'plan':plan, 'availability':'available'}


def verify_metric_plan(action, plan):
    measures = {'revenue':'revenue','revenue_series':'revenue','revenue_growth':'revenue','best_day':'revenue',
                'orders':'orders','quantity':'quantity','quantity_growth':'quantity','rank_products':action.get('by','quantity'),
                'rank_customers':'orders' if action.get('by')=='orders' else 'revenue','day_count':'revenue',
                'debt_rank':'debt','inventory':'inventory','inventory_units':'inventory'}
    if measures.get(action.get('name')) not in {None,plan['measure']}:
        raise DataMateError('Metric khác đại lượng trong plan; sửa metric hoặc lập lại plan có căn cứ')
    operations = {'revenue':'sum','quantity':'sum','orders':'count','best_day':'best_day',
                  'revenue_growth':'growth','quantity_growth':'growth','revenue_series':'series','day_count':'day_count',
                  'rank_products':'rank','rank_customers':'rank','rank_employees':'rank'}
    if action.get('name') in operations and plan['operation'] != operations[action['name']]:
        raise DataMateError('Metric khác phép toán trong plan; tỷ trọng khác tăng trưởng, đếm ngày khác cộng tiền')
    if plan['filters'] or plan.get('numerator_filters') or plan.get('series_filters') or plan['population'] == 'all':
        raise DataMateError('Metric không biểu diễn đủ điều kiện/population của plan; dùng calculate hoặc SQL giữ đủ điều kiện')
    if 'limit' in plan and action.get('limit',1) != plan['limit']:
        raise DataMateError('limit phải khớp plan')


def verify_sql_plan(result, plan, tools):
    """Conservative structural checks; not a proof of arbitrary SQL meaning."""
    tools.verify_fact_sql(result,'',plan=plan)
    refs = {(r['table'],r['column']) for r in result.get('references',[])}
    # View reads expose the underlying *_all columns in SQLite's authorizer.
    expanded = refs | {(t.removesuffix('_all'),c) for t,c in refs}
    if plan['source'] not in result.get('logical_tables',result.get('direct_tables',[])) and not any(t==plan['source'] for t,_ in expanded):
        raise DataMateError('SQL chưa đọc nguồn giao dịch được chọn trong plan')
    schema = source_schema(tools.catalog)
    if plan['measure'] in {'revenue','quantity','orders','gross_profit'}:
        names = ['line_amount','quantity','cost'] if plan['measure']=='gross_profit' else [MEASURES[plan['measure']]]
        if plan['measure']=='orders' and 'count' in {f.lower() for f in result.get('functions',[])}:
            names = []  # COUNT(*) on one-row-per-order facts is valid.
        for name in names:
            if (plan['source'],name) not in expanded:
                raise DataMateError('SQL thiếu trường đại lượng của plan: '+name)
    for predicates in [plan['filters'],plan.get('numerator_filters',[]),*plan.get('series_filters',[])]:
        for predicate in predicates:
            table,column=column_ref(predicate['column'],plan['source'],schema)
            if (table,column) not in expanded:raise DataMateError('SQL chưa đọc trường điều kiện đã khai báo: '+table+'.'+column)
    if plan['operation'] != 'custom':
        expected = calculate(plan,tools)
        if expected.get('availability') != 'available':
            raise DataMateError('Chưa kiểm chứng được nguồn/kỳ của SQL theo plan')
        return {'scope':'agreement_with_compiled_plan_and_source_quality', 'expected_result':expected,
                'semantic_meaning_proven':False}
    return {'scope':'source_lineage_and_required_fields_only', 'semantic_meaning_proven':False}
