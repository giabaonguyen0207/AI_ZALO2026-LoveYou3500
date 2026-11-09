"""Build TRAIN-only SQL supervision; proposals are frozen before gold scoring."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from datamate.agent.runner import Agent
from datamate.agent.tools import BusinessTools
from datamate.config import Settings
from datamate.data.split import DatasetSplit,read_jsonl
from datamate.domain import json_text,DataMateError
from datamate.evaluation.scoring import matches
from datamate.training.dataset import business_holdout,require_approved_source
from datamate.training.seed import training_messages
from datamate.training.sql_recipes import annotate,compile_proposal,VERSION


class Replay:
    counts_as_model=True
    supports_token_budget=False
    def generate(self,messages,remaining_seconds,**kwargs):
        return json_text(next(self.actions,{'action':'final','type':'abstain','reason':'No further curated action'}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,default=ROOT/'outputs/runs/review/gemini-sft-20261109/catalog_cache')
    p.add_argument('--limit',type=int)
    p.add_argument('--business')
    p.add_argument('--retry-from',type=Path,help='Review queue from an earlier pass; ID is for selection only')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    source_hashes={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'src/datamate').rglob('*.py'))}
    source_hashes['tools/prepare_sql_training.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'source_at_generation.json').write_text(json_text(source_hashes,indent=2),encoding='utf-8')
    split=DatasetSplit(ROOT/'train/train')
    source=require_approved_source(split.root,ROOT/'configs/training_sources.json')
    settings=Settings(split_root=str(split.root),cache_dir=str(args.cache_dir.resolve()),backend='transformers',schema_mode='code_only',
                      adaptive_preparation=False,max_steps=8,max_result_rows=1000,question_timeout_seconds=120,sql_timeout_seconds=15)
    backend=Replay();agent=Agent(settings,backend,split)
    questions=split.questions();train_b,val_b=business_holdout(questions,.2,42)
    selected=[q for q in questions if not args.business or q.business_id==args.business]
    if args.retry_from:
        retry_ids={r['id'] for r in read_jsonl(args.retry_from)}
        selected=[q for q in selected if q.id in retry_ids]
    selected=sorted(selected,key=lambda q:(q.business_id,q.id))[:args.limit]
    proposals=[];candidates={}
    # No answer reference is opened until ALL proposals and executions are fixed.
    with (out/'execution.jsonl').open('w',encoding='utf-8') as f:
        for i,q in enumerate(selected,1):
            started=time.monotonic()
            record={'id':q.id,'business_id':q.business_id,**q.prompt_dict(),'partition':'validation' if q.business_id in val_b else 'train'}
            try:
                catalog=agent.ingestor.catalog(q.business_id)
                tools=BusinessTools(catalog,split.rulebook,q.as_of_date,15,1000)
                recipe=annotate(q,tools);record['recipe']=recipe
                actions,checks=compile_proposal(recipe,q,tools);record.update(actions=actions,checks=checks)
                if recipe.get('source_discovery_actions'):
                    actions=[*recipe['source_discovery_actions'],*actions];record['actions']=actions
                if 'tet' in recipe['question_key']:
                    import re
                    year_match=re.search(r'tet\s+(?:am lich\s+)?(?:nam\s+)?(20\d\d)',recipe['question_key'])
                    year=int(year_match[1]) if year_match else int(q.as_of_date[:4])
                    calendar_lookup={'action':'sql','sql':'SELECT MIN(date) AS start_date,MAX(date) AS end_date FROM calendar WHERE (CAST(lunar_year AS INTEGER)=? AND CAST(lunar_month AS INTEGER)=12 AND CAST(lunar_day AS INTEGER)>=23) OR (CAST(lunar_year AS INTEGER)=? AND CAST(lunar_month AS INTEGER)=1 AND CAST(lunar_day AS INTEGER)<=10)','parameters':[year-1,year]}
                    actions=[calendar_lookup,*actions];record['actions']=actions
                    if 'nam truoc' in recipe['question_key']:
                        prior_lookup={**calendar_lookup,'parameters':[year-2,year-1]}
                        actions=[calendar_lookup,prior_lookup,*actions[1:]];record['actions']=actions
                backend.actions=iter(actions)
                result=agent.run(q)
                record.update(submission=result.answer.to_dict(),trace=result.trace,scripted_backend_calls=result.model_calls,
                              runtime_elapsed_seconds=result.elapsed_seconds,actual_model_inference_calls=0)
                messages=training_messages(result)
                if result.answer.type=='abstain':raise ValueError('Runtime abstained')
                candidates[q.id]={'question_id':q.id,'business_id':q.business_id,'messages':messages,'kind':'verified_sql_sft',
                                  'metadata':{'recipe':recipe,'generation':'deterministic_question_schema_curation','references_passed_to_compiler':False,
                                              'gold_used_for_postexecution_review':True,'development_revisions_guided_by_training_errors':True,
                                              'catalog_fingerprint':catalog.fingerprint,'source_fingerprint':source['fingerprint']}}
                record['execution_status']='executed'
            except Exception as exc:
                record.update(execution_status='error',error=f'{type(exc).__name__}: {exc}')
            record['preparation_and_replay_seconds']=round(time.monotonic()-started,4)
            proposals.append(record);f.write(json_text(record)+'\n');f.flush()
            if i%100==0 or i==len(selected):print(f'{i}/{len(selected)}',dict(Counter(r['execution_status'] for r in proposals)),flush=True)
    refs={r['id']:r for r in read_jsonl(split.root/'answers.jsonl')}
    accepted=[];queue=[]
    for r in proposals:
        ok=r['id'] in candidates and matches(r['submission'],refs[r['id']])
        if ok:accepted.append(candidates[r['id']])
        else:queue.append({**r,'reference':refs[r['id']]})
    for name,rows in [('train',[r for r in accepted if r['business_id'] in train_b]),('validation',[r for r in accepted if r['business_id'] in val_b]),('review_queue',queue)]:
        (out/(name+'.jsonl')).write_text(''.join(json_text(r)+'\n' for r in rows),encoding='utf-8')
    summary={'version':VERSION,'selected':len(selected),'accepted':len(accepted),'rejected':len(queue),
             'tasks':dict(Counter(r.get('recipe',{}).get('task') for r in proposals)),
             'rejected_tasks':dict(Counter(r.get('recipe',{}).get('task') for r in queue)),
             'errors':dict(Counter(r.get('error','reference_mismatch') for r in queue)),
             'train_examples':sum(r['business_id'] in train_b for r in accepted),'validation_examples':sum(r['business_id'] in val_b for r in accepted),
             'source_role':source['role'],'source_fingerprint':source['fingerprint'],'train_businesses':sorted(train_b),'validation_businesses':sorted(val_b)}
    (out/'summary.json').write_text(json_text(summary,indent=2),encoding='utf-8')
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in source_hashes.items()):
        raise DataMateError('Source changed during generation; this pass cannot be sealed. Run a new pass with frozen source.')
    print(json_text(summary,indent=2),flush=True)


if __name__=='__main__':main()
