"""Curate a small TRAIN-only SFT seed using source-first task annotations.

No answer/reference is passed to the recipe compiler or replay backend. The
current agent executes the proposed actions before references are decoded.
These are curated examples, not newly collected model trajectories.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from ..agent.planning import validate_plan
from ..agent.protocol import parse_action
from ..agent.runner import Agent
from ..agent.tools import BusinessTools
from ..config import Settings
from ..data.split import DatasetSplit, ensure_outside_dataset, read_jsonl
from ..data.transforms import search_key
from ..domain import DataMateError, json_text
from ..evaluation.scoring import matches
from ..models.artifacts import sha256
from .dataset import business_holdout, require_approved_source

SEED_VERSION=1
TARGET_MODEL_KEY='gemma4-12b'
DEFAULT_RUNS=('outputs/diagnosis-2026-11-03/gemma4-12b-stress72-pipeline-v2-01',
              'outputs/diagnosis-2026-11-03/gemma4-12b-stress100-baseline-v3')


def _digest(value):
    return hashlib.sha256(json_text(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _write_rows(path,rows):
    path.write_text(''.join(json_text(row)+'\n' for row in rows),encoding='utf-8')


def load_seed_sources(split,runs,approved):
    questions={q.id:q for q in split.questions()}
    records=defaultdict(list); provenance=[]
    for directory in map(Path,runs):
        manifest=json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
        identity={k:v for k,v in manifest.items() if k!='identity_hash'}
        if hashlib.sha256(json_text(identity,sort_keys=True).encode()).hexdigest()!=manifest.get('identity_hash'):
            raise DataMateError('Identity hash bundle không khớp: '+str(directory))
        if manifest.get('source_fingerprint')!=approved['fingerprint']:
            raise DataMateError('Bundle không thuộc nguồn TRAIN đã đăng ký')
        selected=read_jsonl(directory/'selected_questions.jsonl')
        wanted=[row['id'] for row in selected]
        if len(wanted)!=len(set(wanted)) or wanted!=manifest['question_ids']:
            raise DataMateError('Danh sách câu bundle bị thay đổi/trùng')
        log_paths=sorted((directory/'logs').glob('*.json'))
        loaded=[json.loads(path.read_text(encoding='utf-8')) for path in log_paths]
        if {row['id'] for row in loaded}!=set(wanted) or len(loaded)!=len(wanted):
            raise DataMateError('Bundle thiếu/thừa/trùng log')
        for row in [*selected,*loaded]:
            q=questions.get(row.get('id'))
            if not q or any(row.get(key)!=getattr(q,key) for key in ('business_id','question','as_of_date')):
                raise DataMateError('Câu/log khác dữ liệu TRAIN gốc')
        for path,row in zip(log_paths,loaded):
            if 'rulebook_sha256' in row and row['rulebook_sha256']!=split.rulebook_hash:
                raise DataMateError('Log khác rulebook hiện tại')
            records[row['id']].append({'run':directory.name,'log_sha256':sha256(path),'record':row})
        provenance.append({'run':directory.name,'manifest_sha256':sha256(directory/'run_manifest.json'),
                           'selection_sha256':sha256(directory/'selected_questions.jsonl'),
                           'source_fingerprint':manifest['source_fingerprint'],
                           'model_manifest_sha256':manifest.get('model_manifest_sha256'),
                           'records':len(loaded),'missing_historical_prompt':sum(not r.get('initial_messages') for r in loaded),
                           'missing_per_record_rulebook_hash':sum('rulebook_sha256' not in r for r in loaded)})
    return questions,records,provenance


def _filter(column,value,op='eq',fold=False):
    return {'column':column,'op':op,'value':value,'evidence_id':'Q',**({'fold_accents':True} if fold else {})}


def inline_generated_ctes(sql):
    """Inline the metric compiler's named CTEs, retaining placeholder order.

    SQLite reports a CTE name as an authorizer context. The current agent's
    direct-table guard rejects those forms. Use equivalent derived tables;
    do not weaken that guard or apply this to untrusted model-written SQL.
    """
    if not sql.startswith('WITH '): return sql
    # Number anonymous parameters before moving nested SELECTs. A placeholder
    # in an outer SELECT can otherwise move before one in an earlier CTE.
    numbered=[];quote_char=None;index=0;parameter=0
    while index<len(sql):
        char=sql[index]
        if quote_char:
            numbered.append(char)
            if char==quote_char:
                if index+1<len(sql) and sql[index+1]==quote_char:
                    numbered.append(sql[index+1]);index+=1
                else: quote_char=None
        elif char in {'\'', '"'}:
            quote_char=char;numbered.append(char)
        elif char=='?':
            if index+1<len(sql) and sql[index+1].isdigit():
                raise DataMateError('Metric compiler cần anonymous parameters trước chuẩn hóa')
            parameter+=1;numbered.append('?'+str(parameter))
        else: numbered.append(char)
        index+=1
    sql=''.join(numbered)
    rest=sql[5:];ctes={}
    while True:
        match=re.match(r'(\w+) AS \(',rest)
        if not match: raise DataMateError('CTE compiler chưa hỗ trợ dạng này')
        name=match[1];start=match.end();depth=1;quote_char=None;index=start
        while index<len(rest) and depth:
            char=rest[index]
            if quote_char:
                if char==quote_char:
                    if index+1<len(rest) and rest[index+1]==quote_char: index+=1
                    else: quote_char=None
            elif char in {'\'', '"'}: quote_char=char
            elif char=='(': depth+=1
            elif char==')': depth-=1
            index+=1
        if depth: raise DataMateError('CTE compiler thiếu dấu đóng')
        body=rest[start:index-1]
        for previous,text in ctes.items():
            body=re.sub(r'\b(FROM|JOIN) '+re.escape(previous)+r'\b',lambda m:m[1]+' ('+text+') AS '+previous,body)
        ctes[name]=body
        rest=rest[index:].lstrip()
        if rest.startswith(','): rest=rest[1:].lstrip();continue
        for previous,text in ctes.items():
            rest=re.sub(r'\b(FROM|JOIN) '+re.escape(previous)+r'\b',lambda m:m[1]+' ('+text+') AS '+previous,rest)
        return rest


def compile_recipe(recipe,question,tools):
    """Uses annotations + source data only. Reference answers are not an input."""
    task=recipe['task']
    if task=='pending':
        raise DataMateError(recipe.get('category','needs_review')+': '+recipe.get('note','Chưa được duyệt'))
    measures={'orders':'orders','cancel_count':'orders','cancel_ratio':'orders','quantity':'quantity',
              'quantity_growth':'quantity','gross_profit':'gross_profit','debt_rank':'debt',
              'inventory':'inventory','inventory_units':'inventory','historical_price':'price'}
    operations={'orders':'count','cancel_count':'count','cancel_ratio':'ratio','revenue_growth':'growth',
                'quantity_growth':'growth','best_day':'best_day','day_count':'day_count',
                'rank_customers':'rank','rank_products':'rank','rank_employees':'rank',
                'revenue_series':'series','payment_series':'series','channel_revenue':'series'}
    custom={'continuous_growth','customers_with_orders','customer_percent','shift_growth','debt_rank',
            'inventory','inventory_units','historical_price','average_order_revenue','order_revenue_threshold','month_rank'}
    measure=measures.get(task,'revenue');operation=operations.get(task,'custom' if task in custom else 'sum')
    kind='number'
    if task in {'cancel_ratio','revenue_growth','quantity_growth','customer_percent','shift_growth'}: kind='percent'
    elif task in {'best_day','month_rank'}: kind='date'
    elif task=='continuous_growth': kind='boolean'
    elif task in {'revenue_series','payment_series','channel_revenue','inventory_units'}: kind='list'
    elif task=='debt_rank': kind='table'
    elif task.startswith('rank_'): kind='string' if recipe['limit']==1 else 'list'
    population='all' if task in {'cancel_count','cancel_ratio','historical_price','cancel_reason_count'} else 'debt' if task=='debt_rank' else 'custom' if task.startswith('inventory') else 'sales'
    plan={'action':'plan','measure':measure,'operation':operation,'type':kind,'population':population,
          'evidence_id':'Q','periods':deepcopy(recipe.get('periods',[])),'filters':[]}
    prefixes=[]; checks=[]
    if task=='debt_rank': plan.update(source='dm_orders_all',date=recipe['date'],requirements=['debt'])
    if task.startswith('inventory'): plan.update(source='dm_inventory',date=recipe['date'],requirements=['inventory'])
    if task=='historical_price':
        plan.update(source='dm_sales_all',periods=[{'kind':'range','start':recipe['date'],'end':recipe['date']}])
    if task.startswith('rank_'):
        entity='customer' if task=='rank_customers' else 'product' if task=='rank_products' else 'employee'
        if task=='rank_customers':
            plan['measure']='orders' if recipe.get('by')=='orders' else 'revenue'
            plan['requirements']=['customer_identity']
        elif task=='rank_products':
            plan['measure']='quantity';plan['requirements']=['product_identity']
        plan.update(group_by=entity+'_key' if entity!='employee' else 'employee',
                    display=entity+'_name' if entity!='employee' else 'employee',limit=recipe['limit'])
    if task in {'cancel_count','cancel_ratio'}:
        plan['source']='dm_orders_all'
        conditions=[_filter('status','cancelled')]
        plan['numerator_filters' if task=='cancel_ratio' else 'filters']=conditions
    if task=='cancel_reason_count':
        plan.update(measure='orders',operation='count',source='dm_orders_all',requirements=['cancellation_reason'])
        plan['filters']=[_filter('status','cancelled'),_filter('cancel_reason',recipe['reason_text'],'contains',True)]
    if recipe.get('shift'):
        plan['filters'].append(_filter('order_timestamp',recipe['shift'],'shift'))
    if recipe.get('size'):
        plan['filters'].append(_filter('product_name','('+recipe['size']+')','endswith'))
    if recipe.get('unit') and not task.startswith('inventory'):
        plan['filters'].append(_filter('base_unit',recipe['unit'],fold=True))
    if recipe.get('product') and not task.startswith('inventory'):
        product=recipe['product']
        if search_key(product) not in search_key(question.question):
            raise DataMateError('Tên sản phẩm chú thích không nằm trong câu hỏi')
        candidates=tools.lookup('product',product)['candidates']
        exact=[item for item in candidates if search_key(item.get('name',''))==search_key(product)]
        if len({item['key'] for item in exact})!=1:
            raise DataMateError('Không có đúng một tên sản phẩm khớp nguồn; cần xem lại lookup')
        plan['filters'].append(_filter('product_name',product,fold=True))
        checks.append({'entity_literal':product,'exact_source_match':True})
    if 'threshold' in recipe and task=='day_count': plan['threshold']=recipe['threshold']
    if task=='payment_series':
        plan['requirements']=['payment_method']
        plan['series_filters']=[[_filter('payment_method',value)] for value in recipe['methods']]
        plan['output_order']=recipe['methods']
    if task=='channel_revenue':
        aliases={'zalo oa':['zalo oa','zalo'],'tai quay':['tai quay','quay','store','offline']}
        plan['requirements']=['channel']
        plan['series_filters']=[[_filter('channel',aliases[search_key(value)],'in',True)] for value in recipe['channels']]
        plan['output_order']=recipe['channels']
    if task in {'customers_with_orders','customer_percent'}:
        plan.update(measure='custom',source='dm_orders',requirements=['customer_identity'])
    # Missing-data evidence must come from the current check tool, not the
    # historical final answer. A deferred operation lets the model inspect it.
    preliminary=tools.check(question.question,plan['periods'][0] if plan['periods'] else None,plan=plan)
    if preliminary['availability']=='missing':
        plan['operation']='custom'
        plan.pop('series_filters',None);plan.pop('output_order',None)
        return plan,prefixes,checks+[{'missing_data_check':preliminary}]
    if task in {'continuous_growth','customers_with_orders','customer_percent','shift_growth','debt_rank','inventory','inventory_units'}:
        metric={'action':'metric','name':task}
        if task=='debt_rank': metric.update(date=recipe['date'],limit=recipe['limit'])
        elif task.startswith('inventory'):
            metric.update(date=recipe['date'],product=recipe['product'])
            if recipe.get('large_unit'): metric['large_unit']=recipe['large_unit']
            if recipe.get('unit'):
                units=tools.executor.execute('SELECT DISTINCT base_unit FROM dm_sales_all WHERE search_key(product_name)=?',
                                             [search_key(recipe['product'])])['rows']
                units={search_key(row[0]) for row in units if row[0] is not None}
                if units!={search_key(recipe['unit'])}:
                    raise DataMateError('Đơn vị hỏi tồn kho chưa khớp duy nhất với đơn vị cơ bản trong nguồn')
        else:
            metric.update({'periods':plan['periods']} if task=='continuous_growth' else {'period':plan['periods'][0]})
            if 'threshold' in recipe: metric['threshold']=recipe['threshold']
        result=tools.metric(metric)
        if result.get('availability')!='available':
            raise DataMateError('Metric chưa tạo được kết quả kiểm chứng: '+str(result.get('missing')))
        sql=result['sql']
        if task=='debt_rank':
            # Project only the fields used by the result. Reading o.* would
            # also read paid/balance snapshots, correctly rejected by the
            # agent's past-date guard even though this query uses the ledger.
            sql=sql.replace('SELECT o.*,MAX(',
                'SELECT o.order_id,o.customer_id,o.customer_name,o.customer_key,MAX(')
        sql=inline_generated_ctes(sql);params=result.get('parameters',[])
        # A rewrite is accepted only if both source queries agree, before any
        # reference answer is loaded. Parameter order must remain identical.
        if sql!=result['sql']:
            rewritten=tools.executor.execute(sql,params)
            if rewritten['rows']!=result['rows'] or rewritten['columns']!=result['columns']:
                raise DataMateError('Derived-table rewrite khác metric gốc')
            checks.append({'derived_query_equivalent_on_source':True})
        if task.startswith('inventory'):
            # Query the named product and conversion in SQL. Internal IDs and
            # conversion factors are source data, never invented target tokens.
            if task=='inventory_units':
                sql=('SELECT DISTINCT CAST(i.quantity/c.factor AS INTEGER) AS whole,'
                     'i.quantity-CAST(i.quantity/c.factor AS INTEGER)*c.factor AS remainder '
                     'FROM dm_inventory i JOIN dm_conversions c ON (i.product_id=c.product_id OR search_key(i.product_name)=search_key(c.product_name)) '
                     'WHERE search_key(i.product_name)=? AND i.snapshot_date=? AND search_key(c.large_unit)=?')
                params=[search_key(recipe['product']),recipe['date'],search_key(recipe['large_unit'])]
            else:
                sql='SELECT quantity FROM dm_inventory WHERE search_key(product_name)=? AND snapshot_date=?'
                params=[search_key(recipe['product']),recipe['date']]
            rewritten=tools.executor.execute(sql,params)
            if rewritten['rows']!=result['rows']:
                raise DataMateError('SQL tồn kho theo tên/quy đổi khác metric đã kiểm chứng')
            checks.append({'inventory_sql_reads_conversion_instead_of_literal_factor':True})
        plan['query']={'sql':sql,'parameters':params}
        checks.append({'compiled_metric':metric,'source_validation':result.get('basis',result.get('availability'))})
    elif task in {'historical_price','average_order_revenue','order_revenue_threshold','month_rank'}:
        accepted=validate_plan(plan,question,tools)
        p=accepted['resolved_periods'][0]
        required=['price','quantity','base_quantity'] if task=='historical_price' else ['order_id','line_amount']
        missing=tools._quality('dm_sales',p,required,statuses='all' if task=='historical_price' else None)
        if missing:
            raise DataMateError('Nguồn/kỳ chưa đủ để tạo SQL custom')
        params=[p['start'],p['end']]
        if task=='historical_price':
            sql='SELECT DISTINCT price*quantity/NULLIF(base_quantity,0) AS value FROM dm_sales_all WHERE order_date BETWEEN ? AND ? AND search_key(product_name)=?'
            params.append(search_key(recipe['product']))
        elif task=='average_order_revenue':
            sql='SELECT round_half_up(SUM(line_amount)*1.0/NULLIF(COUNT(DISTINCT order_id),0),0) AS value FROM dm_sales WHERE order_date BETWEEN ? AND ?'
        elif task=='order_revenue_threshold':
            plan.update(measure='orders',source='dm_sales')
            sql='SELECT COUNT(*) AS value FROM (SELECT order_id,SUM(line_amount) AS revenue FROM dm_sales WHERE order_date BETWEEN ? AND ? GROUP BY order_id HAVING SUM(line_amount)>=?)'
            params.append(recipe['threshold'])
        else:
            sql="SELECT strftime('%Y-%m',order_date) AS month FROM dm_sales WHERE order_date BETWEEN ? AND ? GROUP BY month ORDER BY SUM(line_amount) DESC,month ASC LIMIT 1"
        plan['query']={'sql':sql,'parameters':params}
        checks.append({'all_row_quality_fields':required,'custom_sql_semantics':task})
    return plan,prefixes,checks


class CuratedReplay:
    """Runs the production model-control path without invoking model weights."""
    counts_as_model=True
    def __init__(self,actions): self.actions=iter(deepcopy(actions))
    def generate(self,messages,remaining_seconds):
        try: return json_text(next(self.actions))
        except StopIteration: pass
        feedback=json.loads(messages[-1]['content']).get('tool_result',{})
        proof=feedback if feedback.get('availability')=='missing' else feedback.get('answerability',{})
        if proof.get('availability')=='missing' and proof.get('missing') and proof.get('evidence_id'):
            return json_text({'action':'final','type':'unanswerable','evidence_id':proof['evidence_id'],
                              'reason':'Công cụ xác nhận thiếu: '+', '.join(proof['missing'])})
        raise DataMateError('Không có bước tiếp theo đã kiểm chứng; chuyển hàng chờ')


def training_messages(result):
    messages=deepcopy(result.initial_messages)
    for event in result.trace:
        if event.get('phase')=='preparation':
            raise DataMateError('Seed v1 không học chuẩn bị schema bằng model')
        if event.get('error') or event.get('observation',{}).get('error'):
            raise DataMateError('Trace mới có lỗi; không đưa action sai vào SFT')
        action=parse_action(event['raw'])
        if 'answer' in action:
            raise DataMateError('Action target chứa đáp án số chép lại')
        messages.append({'role':'assistant','content':json_text(action)})
        if not event.get('accepted'):
            messages.append({'role':'user','content':json_text({'tool_result':event.get('model_observation',event.get('observation',{}))})})
    return messages


def prepare_seed(split_root,output,reviews='configs/seed72_100_reviews.json',runs=DEFAULT_RUNS,
                 sources='configs/training_sources.json',cache_dir='.cache/datamate-v3-probes',
                 tokenizer_dirs=None,max_length=16384,max_target_tokens=768):
    if not tokenizer_dirs or set(tokenizer_dirs)!={TARGET_MODEL_KEY}:
        raise DataMateError('Bộ seed này chỉ chuẩn bị và kiểm tokenizer Gemma 4 12B')
    split=DatasetSplit(split_root);approved=require_approved_source(split.root,sources)
    destination=ensure_outside_dataset(output,split.root)
    if destination.exists() and any(destination.iterdir()):
        raise DataMateError('Output seed phải mới/rỗng; không ghi đè dataset đã niêm phong')
    questions,records,provenance=load_seed_sources(split,runs,approved)
    train_businesses,validation_businesses=business_holdout(list(questions.values()),.2,42)
    review_path=Path(reviews);review_rows=json.loads(review_path.read_text(encoding='utf-8'))['reviews']
    review_map={r['question_id']:r for r in review_rows}
    training_ids={qid for qid in records if questions[qid].business_id in train_businesses}
    if len(review_map)!=len(review_rows) or set(review_map)!=training_ids:
        raise DataMateError('Review phải đủ và chỉ gồm các câu seed phía train')
    destination.mkdir(parents=True,exist_ok=True)
    settings=Settings(split_root=str(split.root),cache_dir=str(cache_dir),backend='transformers',
                      model_key=TARGET_MODEL_KEY,model_dir='models/'+TARGET_MODEL_KEY,schema_mode='code_only',adaptive_preparation=False,
                      require_plan=True,auto_execute_plan=True,max_steps=4,max_new_tokens=768,
                      question_timeout_seconds=120)
    agent=Agent(settings,CuratedReplay([]),split)
    executions=[];candidates={};failure={}
    # Freeze annotations/proposals and execute BEFORE decoding answer references.
    (destination/'review_annotations.json').write_bytes(review_path.read_bytes())
    for index,qid in enumerate(sorted(training_ids),1):
        q=questions[qid];review=review_map[qid]
        if review['question_sha256']!=_digest(q.prompt_dict()):
            raise DataMateError('Câu hỏi đã đổi sau khi review: '+qid)
        entry={'question_id':qid,'business_id':q.business_id,'recipe':review['recipe']}
        try:
            catalog=agent.ingestor.catalog(q.business_id)
            tools=BusinessTools(catalog,split.rulebook,q.as_of_date)
            plan,prefixes,checks=compile_recipe(review['recipe'],q,tools)
            agent.backend=CuratedReplay([*prefixes,plan])
            result=agent.run(q)
            entry.update(proposed_actions=[*prefixes,plan],source_checks=checks,
                         submission=result.answer.to_dict(qid),elapsed_cpu_seconds=result.elapsed_seconds,
                         scripted_backend_calls=result.model_calls,trace=result.trace,initial_messages=result.initial_messages)
            if result.answer.type=='abstain':
                cause=next((e.get('observation',{}).get('error') for e in result.trace if e.get('observation',{}).get('error')),None)
                raise DataMateError(cause or result.answer.reason or 'Không có kết quả')
            messages=training_messages(result)
            candidates[qid]={'question_id':qid,'business_id':q.business_id,'messages':messages,
                'kind':'curated_source_first_fastv5','metadata':{'review_recipe':review['recipe'],
                'provenance':'Task annotations + current tool execution; not a new model-generated trajectory',
                'reference_used_for_target_creation':False,'historical_prompt_reconstructed':False,
                'source_runs':[{'run':r['run'],'log_sha256':r['log_sha256']} for r in records[qid]]}}
        except (DataMateError,KeyError,ValueError,TypeError) as exc:
            failure[qid]=str(exc);entry['pre_reference_error']=str(exc)
        executions.append(entry)
        print(f'[{index}/{len(training_ids)}] {qid} '+('executed' if qid in candidates else 'review_queue'),flush=True)
    _write_rows(destination/'execution_audit.jsonl',executions)
    execution_sha=sha256(destination/'execution_audit.jsonl')
    # Scoring is a separate stage; it never changes a recipe or tries parameters
    # until a reference value happens to match.
    references={r['id']:r for r in read_jsonl(split.root/'answers.jsonl')}
    execution_map={r['question_id']:r for r in executions}
    train=[];queue=[];inventory=[]
    for qid,history in sorted(records.items()):
        q=questions[qid];outcomes=[];clean=False
        for source in history:
            row=source['record'];ok=matches(row['submission'],references[qid])
            outcomes.append('correct' if ok else 'abstain' if row['submission']['type']=='abstain' else 'wrong')
            steps=[e for e in row['trace'] if e.get('phase')!='preparation']
            try: [parse_action(e['raw']) for e in steps];parsed=True
            except (DataMateError,KeyError,TypeError): parsed=False
            clean |= bool(ok and steps and steps[-1].get('accepted') and parsed and not any(e.get('error') or e.get('observation',{}).get('error') for e in steps))
        partition='validation' if q.business_id in validation_businesses else 'train'
        item={'question_id':qid,'business_id':q.business_id,**q.prompt_dict(),'partition':partition,
              'historical_outcomes':outcomes,'historical_clean_candidate':clean,'source_records':len(history)}
        if partition=='validation': item['status']='reserved_validation'
        elif qid in candidates and matches(execution_map[qid]['submission'],references[qid]):
            candidates[qid]['metadata'].update(historical_clean_candidate=clean,
                curation_category='converted_clean_history' if clean else 'repaired_history')
            train.append(candidates[qid]);item['status']='verified_before_tokenization'
        else:
            error=failure.get(qid,'reference_mismatch_after_fixed_recipe_execution')
            item.update(status='review_queue',reason=error)
            queue.append({**item,'recipe':review_map[qid]['recipe'],'new_submission':execution_map[qid].get('submission'),
                          'reference_type':references[qid]['type']})
        inventory.append(item)
    tokenizers={};token_report=[]
    if tokenizer_dirs:
        from transformers import AutoTokenizer
        tokenizers={name:AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
                    for name,path in tokenizer_dirs.items()}
    from .finetune import tokenize_conversations
    fitting=[]
    for row in train:
        item={'question_id':row['question_id'],'models':{}};fits=True
        for name,tokenizer in tokenizers.items():
            encoded,skipped=tokenize_conversations([row],tokenizer,max_length)
            item['models'][name]={'sequence_lengths':[len(r['input_ids']) for r in encoded],
                                 'target_tokens':[sum(x!=-100 for x in r['labels']) for r in encoded],
                                 'skipped':skipped}
            fits &= not skipped and bool(encoded) and all(sum(x!=-100 for x in r['labels'])<=max_target_tokens for r in encoded)
        item['fits']=bool(fits);token_report.append(item)
        if fits: fitting.append(row)
        else:
            queue.append({'question_id':row['question_id'],'business_id':row['business_id'],
                          'status':'review_queue','reason':'context_or_target_token_budget_exceeded; no truncation',
                          'token_report':item,'historical_clean_candidate':next(x['historical_clean_candidate'] for x in inventory if x['question_id']==row['question_id'])})
            next(x for x in inventory if x['question_id']==row['question_id']).update(status='review_queue',reason='token_budget_exceeded')
    train=fitting
    if not train: raise DataMateError('Không có mẫu train qua kiểm chứng')
    accepted_ids={row['question_id'] for row in train}
    context_statistics={}
    for model in tokenizers:
        lengths=[n for row in token_report if row['question_id'] in accepted_ids for n in row['models'][model]['sequence_lengths']]
        targets=[n for row in token_report if row['question_id'] in accepted_ids for n in row['models'][model]['target_tokens']]
        context_statistics[model]={'sequences':len(lengths),'min_sequence_tokens':min(lengths),
                                  'max_sequence_tokens':max(lengths),'max_target_tokens':max(targets)}
    selected_validation=[qid for qid in sorted(records) if questions[qid].business_id in validation_businesses]
    files={'train.jsonl':train,'validation.jsonl':[], 'inventory.jsonl':inventory,'repair_queue.jsonl':queue,
           'token_report.jsonl':token_report,'validation_questions.jsonl':[
               {'id':qid,'business_id':questions[qid].business_id,**questions[qid].prompt_dict()} for qid in selected_validation]}
    for name,rows in files.items(): _write_rows(destination/name,rows)
    ids={'train_ids.json':[r['question_id'] for r in train],
         'validation_ids.json':[q.id for q in questions.values() if q.business_id in validation_businesses],
         'validation_seed_ids.json':selected_validation,'seed_ids.json':sorted(records)}
    for name,rows in ids.items(): (destination/name).write_text(json_text(rows,indent=2),encoding='utf-8')
    summary={'target_model_key':TARGET_MODEL_KEY,'seed_questions':len(records),'source_logs':sum(len(v) for v in records.values()),
        'initial_clean_candidates':sum(r['historical_clean_candidate'] for r in inventory),
        'initial_clean_train_candidates':sum(r['historical_clean_candidate'] and r['partition']=='train' for r in inventory),
        'train_examples':len(train),'train_questions':len({r['question_id'] for r in train}),
        'training_assistant_decisions':sum(sum(m['role']=='assistant' for m in r['messages']) for r in train),
        'review_queue':len(queue),'validation_seed_questions':len(selected_validation),
        'validation_full_questions':len(ids['validation_ids.json']),
        'accepted_from_initial_repair_queue':sum(not next(x['historical_clean_candidate'] for x in inventory if x['question_id']==r['question_id']) for r in train),
        'answer_type_counts':dict(Counter(references[r['question_id']]['type'] for r in train)),
        'task_counts':dict(Counter(r['metadata']['review_recipe']['task'] for r in train)),
        'actual_model_inference_calls':0,'gpu_training_performed':False,
        'tokenizer_checks':list(tokenizers),'max_length':max_length,'max_target_tokens':max_target_tokens,
        'context_statistics':context_statistics}
    (destination/'report.json').write_text(json_text(summary,indent=2),encoding='utf-8')
    type_rows='\n'.join('| '+kind+' | '+str(count)+' |' for kind,count in sorted(summary['answer_type_counts'].items()))
    (destination/'data_report.md').write_text(
        '# Dataset seed 72+100 — v1\n\n'
        f"Nguồn: {summary['source_logs']} log, {summary['seed_questions']} câu riêng biệt. "
        f"Đã nhận {summary['train_examples']} câu / {summary['training_assistant_decisions']} quyết định assistant; "
        f"{summary['review_queue']} câu phía train còn trong hàng chờ. "
        f"Giữ ngoài train {summary['validation_seed_questions']} câu seed thuộc 6 doanh nghiệp validation (đầy đủ {summary['validation_full_questions']} câu).\n\n"
        '| Type đã kiểm chứng | Số câu |\n|---|---:|\n'+type_rows+'\n\n'
        'Mẫu được biên tập từ câu hỏi/rulebook và chạy lại qua agent fast v5. Không phải trajectory mới do model sinh; '
        'không tái dựng giả prompt lịch sử bị thiếu. Nhãn tham chiếu chỉ được đọc sau khi niêm phong đề xuất/thực thi. '
        'Kết quả kiểm chứng là kiểm tra dữ liệu/công cụ, không phải PA mới của SLM.\n\n'
        f"Tokenizer offline: {', '.join(tokenizers)}; giới hạn {max_length} token ngữ cảnh và {max_target_tokens} token mỗi target; không cắt mẫu.\n\n"
        'validation.jsonl để trống vì không tạo gold action trên holdout. validation_seed_ids.json dùng cho pilot generation; '
        'validation_ids.json giữ 630 câu của split doanh nghiệp hiện có. Đây là validation phát triển đã có tiếp xúc khi debug.\n\n'
        'Không nhân bản diễn đạt, không huấn luyện GPU, không có lời giải reasoning nội bộ. Mẫu đúng về số vẫn có thể còn sai diễn giải; '
        'chưa có kiểm định ngữ nghĩa độc lập bởi người thứ hai. Xem execution_audit.jsonl và repair_queue.jsonl để rà soát.\n',encoding='utf-8')
    manifest={'kind':'curated_seed','dataset_format_version':2,'seed_version':SEED_VERSION,'target_model_key':TARGET_MODEL_KEY,
        'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'source_role':approved['role'],
        'source_fingerprint':approved['fingerprint'],'rulebook_sha256':split.rulebook_hash,
        'split_strategy':'business_holdout','seed':42,'train_businesses':sorted(train_businesses),
        'validation_businesses':sorted(validation_businesses),'train_examples':len(train),'validation_examples':0,
        'target_pipeline_version':Agent.pipeline_version,'schema_mode':'code_first',
        'generation_method':'source_first_curated_actions_replayed_with_code_only_schema; not LLM generation',
        'historical_prompt_missing':True,'validation_exposure':'development; 18 selected questions already inspected during debugging',
        'execution_audit_sha256_before_reference_scoring':execution_sha,
        'reference_used_for_target_creation':False,'source_runs':provenance,
        'review_sha256':sha256(review_path),'max_length':max_length,'max_target_tokens':max_target_tokens,'tokenizer_checks':list(tokenizers),
        'runtime_source_hashes':{p.relative_to(Path(__file__).parents[1]).as_posix():sha256(p)
                                 for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
        'files':{p.name:sha256(p) for p in sorted(destination.iterdir()) if p.is_file() and p.name!='manifest.json'}}
    (destination/'manifest.json').write_text(json_text(manifest,indent=2),encoding='utf-8')
    return summary


def verify_seed_dataset(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind')!='curated_seed' or manifest.get('source_role')!='competition_training':
        raise DataMateError('Cần dataset seed TRAIN đã kiểm chứng')
    if manifest.get('target_model_key')!=TARGET_MODEL_KEY or manifest.get('tokenizer_checks')!=[TARGET_MODEL_KEY]:
        raise DataMateError('Seed này chỉ dùng Gemma 4 12B')
    if manifest.get('reference_used_for_target_creation') is not False:
        raise DataMateError('Dataset thiếu kiểm tokenizer hoặc nguồn tạo target không hợp lệ')
    if set(manifest['train_businesses']) & set(manifest['validation_businesses']):
        raise DataMateError('Rò rỉ doanh nghiệp validation')
    for name,digest in manifest['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path)!=digest:
            raise DataMateError('Hash/path dataset không hợp lệ: '+name)
    train=read_jsonl(directory/'train.jsonl');seen=set()
    for row in train:
        if row['business_id'] not in manifest['train_businesses'] or row['question_id'] in seen:
            raise DataMateError('Mẫu train trùng hoặc thuộc validation')
        seen.add(row['question_id'])
        if row['messages'][-1]['role']!='assistant': raise DataMateError('Mẫu phải kết thúc ở assistant target')
        if row.get('metadata',{}).get('reference_used_for_target_creation') is not False:
            raise DataMateError('Mẫu không có provenance source-first')
        for message in row['messages']:
            if message['role']=='assistant':
                action=parse_action(message['content'])
                if 'answer' in action: raise DataMateError('Target chứa đáp án chép lại')
    if len(train)!=manifest['train_examples']: raise DataMateError('Số mẫu khác manifest')
    return manifest
