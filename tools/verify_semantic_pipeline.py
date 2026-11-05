"""CPU-only verification of schema/value handling on development businesses.

Manual tool calls below are regression checks, not an LLM benchmark and not
production routing. Reference answers are read only after tool execution.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from datamate.agent.protocol import answer_from_query
from datamate.agent.tools import BusinessTools
from datamate.agent.verification import verify_metric_request
from datamate.data.ingest import Ingestor
from datamate.data.split import DatasetSplit, read_jsonl
from datamate.evaluation.scoring import matches
from datamate.training.dataset import business_holdout, require_approved_source

CASES = [
    ('q_7e974a3f4552',{'name':'best_day','period':{'kind':'month','year':2026,'month':1}}),
    ('q_1cc70c12aa27',{'name':'revenue','period':{'kind':'today'}}),
    ('q_489bb0e084b6',{'name':'orders','period':{'kind':'quarter','year':2025,'quarter':3}}),
    ('q_93e78b3a3a98',{'name':'orders','period':{'kind':'quarter','year':2025,'quarter':4}}),
    ('q_b3b45f9adceb',{'name':'revenue','period':{'kind':'quarter','year':2025,'quarter':3}}),
    ('q_a536e1622568',{'name':'revenue_growth','period':{'kind':'month','year':2025,'month':5},'compare_period':{'kind':'month','year':2025,'month':4}}),
    ('q_db64c1b7436d',{'name':'rank_products','period':{'kind':'month','year':2025,'month':4},'limit':1}),
    ('q_b0728619f5b4',{'name':'quantity_growth','product':'Cà phê muối (M)','period':{'kind':'days_from','date':'2025-08-23','days':7},'compare_period':{'kind':'days_before','date':'2025-08-23','days':7}}),
    ('q_8036e34515e9',{'name':'customer_percent','period':{'kind':'month','year':2025,'month':9},'threshold':3}),
    ('q_27eebcccdf1f',{'name':'revenue_growth','period':{'kind':'month','year':2026,'month':6},'compare_period':{'kind':'month','year':2026,'month':5}}),
    ('q_51d12dd67b24',{'name':'inventory','product':'Efferalgan 500mg viên sủi hộp 4 vỉ x 4 viên'}),
    ('q_8659862171ab',{'name':'debt_rank','date':'2026-08-14','limit':5}),
    ('q_a9305e207d35',{'name':'orders','period':{'kind':'quarter','year':2025,'quarter':3}}),
    ('q_8c6b3bb85449',{'name':'rank_products','period':{'kind':'month','year':2025,'month':8}}),
    ('q_4bdb1a8972ce',{'name':'day_count','period':{'kind':'month','year':2025,'month':11},'threshold':9600000}),
    ('q_1837da83c4a5',{'name':'rank_customers','period':{'kind':'month','year':2026,'month':1}}),
    ('q_afedc107ddb1',{'name':'rank_customers','period':{'kind':'month','year':2026,'month':1},'by':'orders'}),
    ('q_151d139d77ca',{'name':'rank_customers','period':{'kind':'month','year':2026,'month':3},'limit':3}),
    ('q_e35d8fac5233',{'name':'customers_with_orders','period':{'kind':'month','year':2026,'month':1},'threshold':3}),
    ('q_8aeed472de30',{'name':'revenue_growth','period':{'kind':'month','year':2026,'month':1},'compare_period':{'kind':'month','year':2025,'month':1}}),
    ('q_14ea21a5b6a1',{'name':'channel_revenue','period':{'kind':'this_month'},'channels':['Zalo OA','tại quầy']}),
    ('q_9fd096dfdb84',{'name':'shift_growth','period':{'kind':'last_week'}}),
    ('q_b622d7d743bc',{'name':'revenue_series','periods':[{'kind':'month','month':m,'year':2025} for m in (10,11,12)]}),
    ('q_e9930d87abdd',{'name':'debt_rank','date':'2026-03-27','limit':5}),
    ('q_153f4e995558',{'name':'debt_rank','date':'2025-10-27','limit':3}),
    ('q_8ae4326131f8',{'name':'debt_rank','date':'2025-10-05','limit':3}),
    ('q_8f503c0b707e',{'name':'debt_rank','date':'2025-07-07','limit':3}),
    ('q_8423c5cc4a42',{'name':'debt_rank','date':'2026-08-05','limit':3}),
]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split-root',default='train/train')
    parser.add_argument('--output',default='outputs/verification-2026-11-03')
    parser.add_argument('--sources',default='configs/training_sources.json')
    parser.add_argument('--businesses',nargs='*')
    parser.add_argument('--regressions-only',action='store_true')
    parser.add_argument('--tokenizers',action='store_true',help='Use already cached tokenizers offline; no weights/GPU')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    split=DatasetSplit(args.split_root)
    approved=require_approved_source(split.root,args.sources)
    development,holdout=business_holdout(split.questions(),.2,42)
    selected=set(args.businesses or development)
    if not selected<=development:
        raise ValueError('This audit deliberately preserves held-out businesses')
    ingest=Ingestor(split,'.cache/datamate')
    report={'scope':'CPU-only, no model inference','source_fingerprint':approved['fingerprint'],
            'development':sorted(selected),'reserved_holdout':sorted(holdout),'businesses':[],'regressions':[]}
    previous_path=output/'semantic-pipeline.json'
    if args.regressions_only and previous_path.is_file():
        previous=json.loads(previous_path.read_text(encoding='utf-8'))
        if previous.get('development')==sorted(selected):
            report['businesses']=previous.get('businesses',[])
    def save():
        (output/'semantic-pipeline.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not args.regressions_only:
        for business in sorted(selected):
            started=time.monotonic();catalog=ingest.catalog(business);s=catalog.semantic
            item={'business_id':business,'seconds':round(time.monotonic()-started,3),'fingerprint':catalog.fingerprint,
                  'coverage':s['coverage'],'capabilities':s['capabilities'],'quality':s['quality'],
                  'status_profiles':s['status_profiles'],'bindings':s['bindings'],'deduplicated_rows':s['deduplicated_rows']}
            report['businesses'].append(item);save()
            print(business,item['seconds'],'seconds',json.dumps(item['quality'],ensure_ascii=False),flush=True)
    questions={q.id:q for q in split.questions()}
    for qid,action in CASES:
        q=questions[qid]
        if q.business_id not in selected:
            continue
        try:
            catalog=ingest.catalog(q.business_id)
            tool=BusinessTools(catalog,split.rulebook,q.as_of_date)
            request={'action':'metric',**action}
            verify_metric_request(request,q.question,q.as_of_date,tool.policy)
            result=tool.metric(request)
            answer=answer_from_query(result['answer_type'],result).to_dict(qid) if 'rows' in result else {'type':'unanswerable'}
            record={'id':qid,'business_id':q.business_id,'question':q.question,'action':action,'answer':answer,'tool':result}
        except Exception as exc:
            record={'id':qid,'business_id':q.business_id,'action':action,'error':f'{type(exc).__name__}: {exc}'}
        report['regressions'].append(record);save()
    # Labels cannot influence the inputs, binding, normalization, SQL or tools.
    refs={r['id']:r for r in read_jsonl(split.root/'answers.jsonl')}
    for r in report['regressions']:
        r['reference']=refs[r['id']]
        r['matches_reference']=matches(r['answer'],refs[r['id']]) if 'answer' in r else False
    report['regression_summary']={'cases':len(report['regressions']),'matches':sum(r['matches_reference'] for r in report['regressions']),
                                  'errors':sum('error' in r for r in report['regressions']),
                                  'not_an_llm_benchmark':True}
    save()
    for r in report['regressions']:
        print(r['id'],r['business_id'],'MATCH' if r['matches_reference'] else 'REVIEW',r.get('answer',r.get('error')),flush=True)
    print(report['regression_summary'],flush=True)
    if args.tokenizers:
        from datamate.agent.prompts import initial_messages
        from datamate.agent.verification import question_contract,question_periods
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        from transformers import AutoTokenizer
        report['tokenizers']=[]
        for model_key in ('qwen35-9b','gemma4-12b'):
            directory=Path('.cache/tokenizers')/model_key
            tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
            lengths=[]
            for business in sorted(selected):
                q=max((q for q in questions.values() if q.business_id==business),key=lambda q:len(q.question))
                catalog=ingest.catalog(business);tool=BusinessTools(catalog,split.rulebook,q.as_of_date)
                periods=question_periods(q.question,q.as_of_date,tool.policy)
                spec={'kind':'range','start':periods[0]['start'],'end':periods[0]['end']} if periods else None
                preflight=tool.check(q.question,spec)
                messages=initial_messages(q,split.rulebook,split.readme(business),catalog.summary(),
                    {'contract':question_contract(q.question),'periods':periods,'answerability':preflight})
                encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                tokens=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
                tokens=tokens.tolist() if hasattr(tokens,'tolist') else tokens
                if tokens and isinstance(tokens[0],list):
                    tokens=tokens[0]
                assert len(tokens)>1000,'Tokenizer check must count input IDs, not BatchEncoding keys'
                lengths.append({'business_id':business,'initial_tokens':len(tokens),'fits_16k_with_768_output':len(tokens)+768<=16384})
            item={'model_key':model_key,'offline':True,'weights_loaded':False,'min_tokens':min(x['initial_tokens'] for x in lengths),
                  'max_tokens':max(x['initial_tokens'] for x in lengths),'businesses':lengths}
            report['tokenizers'].append(item);save()
            print('TOKENIZER',model_key,item['min_tokens'],item['max_tokens'],flush=True)
            if not all(x['fits_16k_with_768_output'] for x in lengths):
                raise RuntimeError('Initial prompt exceeds the existing 16K budget; fix prompt packaging before GPU inference')


if __name__=='__main__':
    main()
