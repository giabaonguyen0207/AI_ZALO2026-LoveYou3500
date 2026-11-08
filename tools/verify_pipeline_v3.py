"""Replay manually specified TRAIN calculations and measure prompts offline.

Diagnostic case IDs select regression inputs only. They are never imported by
the agent. These supplied plans are not model predictions or a PA benchmark.
Reference labels are loaded only after every calculation has been attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from datamate.agent.planning import calculate, validate_plan
from datamate.agent.prompts import initial_messages
from datamate.agent.protocol import answer_from_query
from datamate.agent.tools import BusinessTools
from datamate.data.ingest import Ingestor
from datamate.data.split import DatasetSplit, read_jsonl
from datamate.domain import json_text
from datamate.evaluation.scoring import matches
from datamate.training.dataset import require_approved_source


def month(y,m):
    return {'kind':'month','year':y,'month':m}


CASES = [
    ('q_3234281bf31d','orders','count',[month(2026,5)],'all','cancel'),
    ('q_6d57349294ec','orders','count',[month(2025,7)],'all','cancel'),
    ('q_255991af3110','orders','count',[month(2026,3)],'all','cancel'),
    ('q_46d72d387508','orders','ratio',[month(2026,4)],'all','cancel_ratio'),
    ('q_5348170a927c','orders','ratio',[month(2025,6)],'all','cancel_ratio'),
    ('q_2ca8fd946a09','revenue','sum',[{'kind':'year_to_date'}],'sales',None),
    ('q_5a4149617845','revenue','sum',[{'kind':'year_to_date'}],'sales',None),
    ('q_3d7beff53344','revenue','sum',[{'kind':'year_to_date'}],'sales',None),
    ('q_137ebe6e6af4','orders','count',[{'kind':'year_to_date'}],'sales',None),
    ('q_55e037221ed7','orders','count',[{'kind':'year_to_date'}],'sales',None),
    ('q_ece3c46e24d3','revenue','growth',[{'kind':'this_month'},{'kind':'previous_month_to_date'}],'sales',None),
    ('q_576da3a138ac','revenue','sum',[month(2025,10)],'sales','afternoon'),
    ('q_f4a89e1d7ba0','revenue','sum',[month(2026,8)],'sales','afternoon'),
    ('q_2674d035b36f','orders','count',[month(2025,8)],'sales','morning'),
    ('q_4bbd72655e8e','quantity','sum',[month(2025,9)],'sales','dish'),
    ('q_b0ff97ca1db9','quantity','sum',[{'kind':'this_week'}],'sales','cup'),
    ('q_b5c0715af111','quantity','sum',[month(2025,2)],'sales','size_l'),
    ('q_81cc91f86744','gross_profit','sum',[month(2025,9)],'sales',None),
    ('q_72f43cc7a0bc','revenue','best_day',[month(2025,4)],'sales',None),
    ('q_7c0a4b5f5196','revenue','day_count',[month(2026,5)],'sales','threshold'),
    ('q_b3e5c2fb4852','revenue','sum',[month(2026,4)],'sales','non_prescription'),
]


def supplied_plan(q,measure,operation,periods,population,condition,catalog):
    p={'action':'plan','measure':measure,'operation':operation,'type':'percent' if operation in {'ratio','growth'} else 'date' if operation=='best_day' else 'number',
       'population':population,'periods':periods,'filters':[],'evidence':q.question}
    def predicate(column,op,value,**extra):
        return {'column':column,'op':op,'value':value,'evidence':q.question,**extra}
    if condition=='cancel':
        p['filters']=[predicate('status','eq','cancelled')]
    if condition=='cancel_ratio':
        p['numerator_filters']=[predicate('status','eq','cancelled')]
    if condition in {'morning','afternoon'}:
        p['filters']=[predicate('order_timestamp','shift',condition)]
    if condition in {'dish','cup','size_l'}:
        p['filters']=[predicate('base_unit','eq','đĩa' if condition=='dish' else 'ly',fold_accents=True)]
    if condition=='dish':
        p['filters'].append(predicate('product_id','eq','SP0037'))  # Manually checked entity for this TRAIN regression, not runtime routing.
    if condition=='size_l':
        p['filters'].append(predicate('product_name','endswith','(L)'))
    if condition=='threshold':
        p['threshold']=11000000
    if condition=='non_prescription':
        raw=next(t for t in catalog.tables if 'Thuốc kê đơn' in t['columns'])
        binding=next(b for b in catalog.semantic['bindings'] if b['table']==raw['name'])
        dimension=binding['clean_table']
        p['joins']=[{'table':dimension,'left':'product_id','right':dimension+'.'+binding['roles']['product_id']}]
        p['filters']=[predicate(dimension+'.Thuốc kê đơn','not_true',True)]
    return p


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split-root',default='train/train')
    parser.add_argument('--output',default='outputs/verification-pipeline-v3-2026-11-05')
    parser.add_argument('--tokenizers',action='store_true')
    args=parser.parse_args()
    split=DatasetSplit(args.split_root)
    source=require_approved_source(split.root,'configs/training_sources.json')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    ingestor=Ingestor(split,'.cache/datamate-v3-probes')
    questions={q.id:q for q in split.questions()}
    report={'scope':'CPU regression calculations with manually supplied plans; no model inference',
            'source_fingerprint':source['fingerprint'],'rulebook_sha256':split.rulebook_hash,
            'source_hashes':{p.relative_to(Path('src/datamate')).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('src/datamate').rglob('*.py'))},
            'cases':[]}
    for qid,measure,operation,periods,population,condition in CASES:
        q=questions[qid];start=time.monotonic()
        item={'id':qid,'business_id':q.business_id}
        try:
            catalog=ingestor.catalog(q.business_id)
            tools=BusinessTools(catalog,split.rulebook,q.as_of_date)
            proposal=supplied_plan(q,measure,operation,periods,population,condition,catalog)
            accepted=validate_plan(proposal,q,tools,split.readme(q.business_id))
            result=calculate(accepted,tools)
            item.update(plan=accepted,tool=result,answer=answer_from_query(accepted['type'],result).to_dict(qid))
        except Exception as exc:
            item['error']=f'{type(exc).__name__}: {exc}'
        item['seconds']=round(time.monotonic()-start,3)
        report['cases'].append(item)
        print(qid,item.get('answer',item.get('error')),flush=True)
    # Unsold products need a set difference; free SQL remains supported. Nullable
    # product IDs are deliberately replaced with available stable product keys.
    q=questions['q_ddd87e1d24ed'];catalog=ingestor.catalog(q.business_id)
    tools=BusinessTools(catalog,split.rulebook,q.as_of_date)
    sql="""SELECT COUNT(*) FROM (SELECT DISTINCT before.product_key FROM dm_sales before
           WHERE before.order_date<? AND before.product_key IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM dm_sales during WHERE during.product_key=before.product_key AND during.order_date BETWEEN ? AND ?))"""
    result=tools.executor.execute(sql,['2025-10-07','2025-10-07','2025-10-13'])
    tools.verify_fact_sql(result,q.question)
    report['cases'].append({'id':q.id,'business_id':q.business_id,'tool':result,'answer':answer_from_query('number',result).to_dict(q.id)})
    refs={r['id']:r for r in read_jsonl(split.root/'answers.jsonl')}
    for item in report['cases']:
        item['reference']=refs[item['id']]
        item['matches_reference']='answer' in item and matches(item['answer'],item['reference'])
    report['summary']={'cases':len(report['cases']),'matches':sum(c['matches_reference'] for c in report['cases']),
                       'errors':sum('error' in c for c in report['cases']),'not_an_llm_benchmark':True}
    (output/'regressions.json').write_text(json_text(report,indent=2),encoding='utf-8')
    print(json_text(report['summary']),flush=True)
    if args.tokenizers:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        from transformers import AutoTokenizer
        token_report=[]
        for model in ('qwen35-9b','gemma4-12b'):
            tokenizer=AutoTokenizer.from_pretrained(Path('.cache/tokenizers')/model,local_files_only=True,trust_remote_code=False)
            lengths=[]
            for business in sorted({q.business_id for q in questions.values()}):
                q=max((q for q in questions.values() if q.business_id==business),key=lambda q:len(q.question))
                catalog=ingestor.catalog(business)
                messages=initial_messages(q,split.rulebook,split.readme(business),catalog.summary(),{'plan_required':True,'lexical_hints_are_advisory':True})
                encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                tokens=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
                tokens=tokens.tolist() if hasattr(tokens,'tolist') else tokens
                if tokens and isinstance(tokens[0],list):
                    tokens=tokens[0]
                lengths.append({'business_id':business,'initial_tokens':len(tokens),'fits_16k_with_768_output':len(tokens)+768<=16384})
            token_report.append({'model':model,'businesses':lengths,'min_tokens':min(x['initial_tokens'] for x in lengths),'max_tokens':max(x['initial_tokens'] for x in lengths)})
            print(model,token_report[-1]['min_tokens'],token_report[-1]['max_tokens'],flush=True)
        (output/'tokenizers.json').write_text(json_text(token_report,indent=2),encoding='utf-8')
        from datamate.agent.preparation import RULE_PROMPT, TABLE_PROMPT, table_context
        from datamate.data.interpretation import survey_catalog
        from datamate.data.binding import bind_table
        from datamate.data.rules import read_policy
        # Largest initial prompts plus the most fragmented schema. Selection is
        # based on input size, with no reference labels or model outputs.
        selected={r['business_id'] for result in token_report for r in sorted(result['businesses'],key=lambda r:r['initial_tokens'],reverse=True)[:3]}
        selected.add(max(split.businesses(),key=lambda b:len(ingestor.catalog(b).semantic['bindings'])))
        preparation=[]
        for business in sorted(selected):
            catalog=ingestor.catalog(business)
            survey=survey_catalog(catalog)
            schema=[{'name':t['name'],'file':t['file'],'sheet':t['sheet'],'columns':list(t['columns'])} for t in survey['tables']]
            contexts=[table_context(t,schema,split.readme(business),read_policy(split.rulebook).to_dict(),
                       bind_table(next(raw for raw in catalog.tables if raw['name']==t['name']),split.readme(business))) for t in survey['tables']]
            for model in ('qwen35-9b','gemma4-12b'):
                tokenizer=AutoTokenizer.from_pretrained(Path('.cache/tokenizers')/model,local_files_only=True,trust_remote_code=False)
                sizes=[]
                for context in contexts:
                    encoded=tokenizer.apply_chat_template([{'role':'system','content':TABLE_PROMPT},{'role':'user','content':json_text(context)}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
                    ids=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
                    ids=ids.tolist() if hasattr(ids,'tolist') else ids
                    if ids and isinstance(ids[0],list):ids=ids[0]
                    sizes.append(len(ids))
                preparation.append({'business_id':business,'model':model,'tables':len(contexts),'max_input_tokens':max(sizes),
                                    'fits_16k_with_2048_output':max(sizes)+2048<=16384})
                print('PREPARATION',preparation[-1],flush=True)
        (output/'preparation-tokenizers.json').write_text(json_text(preparation,indent=2),encoding='utf-8')
        if not all(r['fits_16k_with_2048_output'] for r in preparation) or not all(r['fits_16k_with_768_output'] for result in token_report for r in result['businesses']):
            raise SystemExit('Prompt budget exceeded; review token reports')
    if not all(c['matches_reference'] for c in report['cases']):
        raise SystemExit('Some regression calculations need review; inspect regressions.json')


if __name__=='__main__':
    main()
