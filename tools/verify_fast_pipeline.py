"""Offline CPU/replay verification. Does not run an LLM or measure GPU speed."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from datamate.agent.prompts import initial_messages
from datamate.agent.runner import Agent
from datamate.config import Settings
from datamate.data.split import DatasetSplit
from datamate.domain import Question, json_text
from datamate.training.dataset import require_approved_source
from tools.diagnose_train import diagnostic_plan


class ReplayBackend:
    # Exercise the model control-flow path without loading or invoking weights.
    counts_as_model=True
    def __init__(self,action):
        self.actions=iter([action])
    def generate(self,messages,remaining_seconds):
        return json_text(next(self.actions))


def token_count(tokenizer,messages):
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    ids=ids['input_ids'] if hasattr(ids,'keys') else ids
    ids=ids.tolist() if hasattr(ids,'tolist') else ids
    return len(ids[0] if ids and isinstance(ids[0],list) else ids)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root',default='train/train')
    parser.add_argument('--output',default='outputs/verification-fast-v5-2026-11-05')
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    split=DatasetSplit(args.split_root)
    source=require_approved_source(split.root,'configs/training_sources.json')
    evidence_dir=Path('outputs/diagnosis-2026-11-05/round4-notebook-analysis')
    received=json.loads((evidence_dir/'interactive-biz101.json').read_text(encoding='utf-8'))
    original_plan=next(e['llm_action'] for e in received['trace'] if e.get('llm_action',{}).get('action')=='plan')
    settings=Settings(split_root=str(split.root),model_key=received['model_key'],
                      model_dir='models/'+received['model_key'],cache_dir='.cache/datamate-v3-probes',
                      max_steps=6,max_new_tokens=768)
    agent=Agent(settings,ReplayBackend(original_plan),split)
    question=Question('diagnostic-plan-replay',received['business_id'],received['question'],received['as_of_date'])
    result=agent.run(question)
    # Prior diagnostic results are consulted after this calculation, never
    # passed to the backend or used to choose the plan.
    prior=json.loads((evidence_dir/'analysis.json').read_text(encoding='utf-8'))
    replay={'scope':'CPU replay of one actual round-4 model plan; no new model inference or GPU timing',
            'source_trace_sha256':hashlib.sha256((evidence_dir/'interactive-biz101.json').read_bytes()).hexdigest(),
            'plan':original_plan,'answer':result.answer.to_dict(),'expected_prior_cpu_probe':prior['cpu_probe']['revenue'],
            'matches_prior_cpu_probe':result.answer.answer==prior['cpu_probe']['revenue'],
            'backend_generate_invocations':result.model_calls,'cpu_elapsed_seconds':result.elapsed_seconds,
            'trace':result.trace,'initial_messages':result.initial_messages}
    (output/'actual-plan-replay.json').write_text(json_text(replay,indent=2),encoding='utf-8')
    if not replay['matches_prior_cpu_probe'] or result.model_calls!=1:
        raise SystemExit('Replay did not recover the known calculation in one backend call; inspect trace')

    sample,*_=diagnostic_plan(split.questions(),{},seed=20261103,sample_size=100,sampling_mode='random')
    first_seven=[q.id for q in sample[:7]]
    selection={'sampling':'uniform random over all training questions, same 100-question sample',
               'seed':20261103,'sample_size':len(sample),'default_run_limit':7,'first_seven_ids':first_seven,
               'same_first_seven_as_round4':first_seven==[q['id'] for q in prior['paired_questions']],
               'all_sample_ids':[q.id for q in sample]}
    (output/'sample-check.json').write_text(json_text(selection,indent=2),encoding='utf-8')
    if not selection['same_first_seven_as_round4']:
        raise SystemExit('Default sample no longer matches round 4')

    from transformers import AutoTokenizer
    prompts=[]
    for q in sample[:7]:
        catalog=agent.ingestor.catalog(q.business_id)
        prompts.append((q,initial_messages(q,split.rulebook,split.readme(q.business_id),catalog.summary(),
                       {'plan_required':True,'auto_execute_plan':True,'schema_mode':'code_first'})))
    token_report=[]
    for model in ('qwen35-9b','gemma4-12b'):
        tokenizer=AutoTokenizer.from_pretrained(Path('.cache/tokenizers')/model,local_files_only=True,trust_remote_code=False)
        items=[]
        for q,messages in prompts:
            previous=deepcopy(messages);previous[0]['content']=received['initial_messages'][0]['content']
            current_size=token_count(tokenizer,messages)
            items.append({'id':q.id,'business_id':q.business_id,'input_tokens':current_size,
                          'same_user_context_with_prior_system_tokens':token_count(tokenizer,previous),
                          'fits_16k_with_768_output':current_size+768<=16384})
        token_report.append({'model':model,'scope':'first seven training questions only; tokenizer offline, no generation',
                             'questions':items})
    (output/'fast-tokenizers.json').write_text(json_text(token_report,indent=2),encoding='utf-8')
    report={'pipeline_version':agent.pipeline_version,'source_fingerprint':source['fingerprint'],
            'scope':'software/CPU/tokenizer verification only; no new SLM benchmark, PA, or GPU speed measurement',
            'source_hashes':{p.relative_to(Path('src/datamate')).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(Path('src/datamate').rglob('*.py'))},
            'same_first_seven_as_round4':selection['same_first_seven_as_round4'],
            'actual_plan_replay_answer':result.answer.to_dict(),'replay_backend_calls':result.model_calls,
            'all_seven_prompts_fit_for_both_tokenizers':all(q['fits_16k_with_768_output'] for m in token_report for q in m['questions'])}
    (output/'verification.json').write_text(json_text(report,indent=2),encoding='utf-8')
    print(json_text({k:v for k,v in report.items() if k!='source_hashes'},indent=2),flush=True)
    if not report['all_seven_prompts_fit_for_both_tokenizers']:
        raise SystemExit('Prompt budget exceeded')


if __name__=='__main__':
    main()
