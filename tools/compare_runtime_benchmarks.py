"""Compare paired runtime evidence; quality requires gold labels loaded after inference."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def timeout_call(row):
    error = str(row.get('error') or '').lower()
    return row.get('stop_reason')=='timeout_aborted' or any(t in error for t in ('timeout','hết thời gian'))


def question_evidence(folder, identity, predictions, calls):
    """Read old and new result formats without rerunning a model or agent."""
    timing_path=folder/'question_timing.jsonl'
    timing={r['id']:r for r in lines(timing_path)} if timing_path.is_file() else {}
    traces={}
    for path in sorted((folder/'logs').glob('*.json')):
        item=read(path)
        if isinstance(item,dict) and 'id' in item:
            if item['id'] in traces:raise ValueError('Duplicate question log: '+str(item['id']))
            traces[item['id']]=item
    result={}
    for prediction in predictions:
        qid=prediction['id']
        rows=sorted((r for r in calls if r.get('id')==qid),key=lambda r:r['start_seconds'])
        clock=timing.get(qid,{})
        reason=prediction.get('reason','') or clock.get('answer_reason','') or ''
        budget=clock.get('question_timeout_seconds',identity['settings'].get('question_timeout_seconds'))
        explicit_timeout=any(timeout_call(r) for r in rows)
        possible_stop=any(r.get('stop_reason')=='time_or_other' for r in rows)
        # The agent's legacy reason merges step/time exhaustion. Resolve it
        # only when elapsed time also reaches the configured wall deadline.
        deadline_reason=('hết ngân sách bước hoặc thời gian' in reason.lower()
                         and isinstance(budget,(int,float)) and clock.get('duration_seconds',0)>=budget)
        limited=explicit_timeout or possible_stop or deadline_reason
        infrastructure=any(r.get('error') and not timeout_call(r) for r in rows) or bool(clock.get('error'))
        outcome=('infrastructure_error' if infrastructure else 'deadline_limited' if limited
                 else 'abstain' if prediction['type']=='abstain' else 'answered')
        first_delay=None
        if rows and 'start_seconds' in clock:
            first_delay=max(0,rows[0]['start_seconds']-clock['start_seconds'])
        duration=clock.get('duration_seconds')
        generation=sum(r.get('duration_seconds',0) for r in rows) if rows else None
        result[qid]={'outcome':outcome,'answer_type':prediction['type'],'duration_seconds':duration,
                     'first_model_delay_seconds':first_delay,'generation_wall_seconds':generation,
                     'other_wall_seconds':max(0,duration-generation) if duration is not None and generation is not None else None,
                     'model_calls':len(rows),'timeout_calls':sum(timeout_call(r) for r in rows),
                     'possible_time_stop_calls':sum(r.get('stop_reason')=='time_or_other' for r in rows),
                     'trace':traces.get(qid,{}).get('trace',[]),'calls':rows}
    return result


def first_action_difference(left, right):
    events=[]
    for evidence in (left,right):
        events.append([e for e in evidence['trace'] if 'generation' in e or 'raw' in e])
    for index in range(max(len(events[0]),len(events[1]))):
        pair=[events[side][index] if index<len(events[side]) else None for side in (0,1)]
        actions=[None if e is None else e.get('llm_action',e.get('action')) for e in pair]
        if actions[0]!=actions[1]:
            hashes=[evidence['calls'][index].get('messages_sha256') if index<len(evidence['calls']) else None
                    for evidence in (left,right)]
            return {'generation_index':index+1,'same_request_messages':bool(hashes[0]) and hashes[0]==hashes[1],
                    'baseline_action':actions[0],'candidate_action':actions[1]}
    return None


def difference_paths(left, right, path=''):
    if type(left)!=type(right):return [path or '/']
    if isinstance(left,dict):
        return [item for key in sorted(set(left)|set(right))
                for item in difference_paths(left.get(key),right.get(key),path+'/'+key)]
    if isinstance(left,list):
        if len(left)!=len(right):return [path+'/length']
        return [item for index,(a,b) in enumerate(zip(left,right))
                for item in difference_paths(a,b,path+'/'+str(index))]
    return [] if left==right else [path or '/']


def first_feedback_difference(left,right):
    events=[[e for e in side['trace'] if 'generation' in e or 'raw' in e] for side in (left,right)]
    for index,(a,b) in enumerate(zip(*events)):
        paths=difference_paths(a.get('model_observation'),b.get('model_observation'))
        if paths:
            return {'after_generation_index':index+1,'paths':paths[:20],
                    'only_elapsed_seconds':all(p.endswith('/elapsed_seconds') for p in paths)}
    return None


def positive_ratio(numerator, denominator):
    if (type(numerator) not in (int,float) or type(denominator) not in (int,float)
            or not math.isfinite(numerator) or not math.isfinite(denominator) or numerator<=0 or denominator<=0):
        return None
    return numerator/denominator


def compare(baseline, candidate, *, references=None):
    baseline, candidate = Path(baseline), Path(candidate)
    identities = [read(p/'benchmark_identity.json') for p in (baseline,candidate)]
    metrics = [read(p/'benchmark_metrics.json') for p in (baseline,candidate)]
    predictions = [lines(p/'submission.jsonl') for p in (baseline,candidate)]
    calls = [lines(p/'model_calls.jsonl') for p in (baseline,candidate)]
    inputs = [read(p/'run_identity.json') for p in (baseline,candidate)]
    left,right = identities
    common = ('model_key','mode','quantization','max_steps','max_repeated_actions','max_context_tokens','max_new_tokens','seed',
              'question_timeout_seconds','sql_timeout_seconds','max_result_rows','schema_mode','adaptive_preparation',
              'require_plan','auto_execute_plan','schema_repair_timeout_seconds','schema_repair_max_new_tokens',
              'preparation_timeout_seconds','preparation_max_attempts','preparation_max_new_tokens')
    config_differences = {key:[left['settings'].get(key),right['settings'].get(key)] for key in common
                          if left['settings'].get(key) != right['settings'].get(key)}
    # Nonquantized FP16 is the primary experiment. NF4-vs-vLLM BNB needs its
    # own explicit deployment comparison and cannot pass this strict gate.
    expected_ids = set(left['ids'])
    first = []
    for rows in calls:
        mapping = {}
        for row in sorted(rows,key=lambda r:r['start_seconds']):
            mapping.setdefault(row['id'],row['messages_sha256'])
        first.append(mapping)
    mismatched_prompts = [qid for qid in sorted(expected_ids) if first[0].get(qid)!=first[1].get(qid)
                          or first[0].get(qid) is None]
    runtime_packages = ('torch','transformers','tokenizers','accelerate','bitsandbytes')
    package_differences = {key:[left['packages'].get(key),right['packages'].get(key)] for key in runtime_packages
                           if left['packages'].get(key)!=right['packages'].get(key)}
    hardware = [{(v['name'],v['total_mib']) for v in m['gpu_hardware'].values()} for m in metrics]
    hardware_equal = bool(hardware[0]) and hardware[0]==hardware[1] and len(metrics[0]['gpu_hardware'])==len(metrics[1]['gpu_hardware'])
    ids_equal = (bool(expected_ids) and len(left['ids'])==len(expected_ids) and left['ids']==right['ids']
                 and all({r['id'] for r in rows}==expected_ids and len(rows)==len(expected_ids) for rows in predictions))
    model_equal = left['model_manifest_sha256']==right['model_manifest_sha256'] and left['model_manifest_sha256'] is not None
    precisions=[]
    for m in metrics:
        info=m.get('backend_info',{})
        dtype=info.get('model_dtype') or info.get('runtime',{}).get('engine_args',{}).get('dtype')
        precisions.append(str(dtype).removeprefix('torch.').replace('half','float16'))
    gates = {'same_ids_complete':ids_equal,'same_model_manifest':model_equal,
             'expected_backend_pair':left['settings'].get('backend')=='transformers' and right['settings'].get('backend')=='vllm'
                                     and metrics[0].get('backend_info',{}).get('class')=='TransformersBackend'
                                     and metrics[1].get('backend_info',{}).get('class')=='VLLMBackend',
             'token_metadata_complete':all(m.get('token_metadata_complete',False) for m in metrics),
             'same_explicit_float16_precision':precisions==['float16','float16'],
             'same_dataset_hashes':all(inputs[0].get(k) is not None and inputs[0].get(k)==inputs[1].get(k)
                                      for k in ('rulebook_sha256','questions_sha256','input_hashes')),
             'same_visible_devices':left.get('cuda_visible_devices')==right.get('cuda_visible_devices'),
             'same_pipeline_settings':not config_differences,'same_initial_requests':not mismatched_prompts,
             'same_runtime_libraries':not package_differences,'same_gpu_type_count':hardware_equal,
             'both_runs_valid':all(m['valid_complete_run'] for m in metrics),
             'nonquantized_primary':left['settings']['quantization']=='none' and right['settings']['quantization']=='none'
                                    and right['settings'].get('vllm',{}).get('quantization','none')=='none',
             'not_all_abstain':all(any(r['type']!='abstain' for r in rows) for rows in predictions)}
    scores = None
    if references is not None:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        from datamate.evaluation.scoring import score
        gold = lines(references)
        scores = [{k:v for k,v in score(rows,gold,expected_ids).items() if k!='details'} for rows in predictions]
    maps = [{r['id']:{k:v for k,v in r.items() if k in {'type','answer'}} for r in rows} for rows in predictions]
    disagreements = [qid for qid in sorted(expected_ids) if maps[0].get(qid)!=maps[1].get(qid)]
    evidence=[question_evidence(folder,identity,prediction,rows)
              for folder,identity,prediction,rows in zip((baseline,candidate),identities,predictions,calls)]
    identity_compatible=all(value for key,value in gates.items() if key not in {'both_runs_valid','not_all_abstain'})
    complete=ids_equal and all(m.get('error') is None and m.get('written',len(predictions[i]))==len(expected_ids)
                              and m.get('execution_complete',True) for i,m in enumerate(metrics))
    no_infrastructure=all(r['outcome']!='infrastructure_error' for side in evidence for r in side.values())
    no_limits=all(r['outcome']!='deadline_limited' for side in evidence for r in side.values())
    gates['no_deadline_limited_questions']=no_limits
    gates['no_infrastructure_errors']=no_infrastructure
    gates['finite_positive_times']=all(positive_ratio(m.get('inference_seconds'),m.get('cold_total_seconds')) is not None for m in metrics)
    valid = all(gates.values())
    observed_ratio=positive_ratio(metrics[0].get('inference_seconds'),metrics[1].get('inference_seconds')) if identity_compatible and complete and no_infrastructure else None
    observed_cold_ratio=positive_ratio(metrics[0].get('cold_total_seconds'),metrics[1].get('cold_total_seconds')) if identity_compatible and complete and no_infrastructure else None
    speedup = observed_ratio if valid else None
    cold_speedup = observed_cold_ratio if valid else None
    per_question=[]
    for qid in sorted(expected_ids):
        pair=[side.get(qid,{}) for side in evidence]
        different=qid in disagreements
        missing=not pair[0] or not pair[1]
        if missing:kind='missing_result'
        elif not different:kind='same_answer'
        elif any(p['outcome']=='infrastructure_error' for p in pair):kind='infrastructure_affected_difference'
        elif any(p['outcome']=='deadline_limited' for p in pair):kind='deadline_affected_difference'
        elif all(p['answer_type']!='abstain' for p in pair):kind='different_non_abstain_answers'
        else:kind='answer_vs_abstain'
        per_question.append({'id':qid,'difference_kind':kind,'same_answer':not different and not missing,
            'baseline':{k:v for k,v in pair[0].items() if k not in {'trace','calls'}},
            'candidate':{k:v for k,v in pair[1].items() if k not in {'trace','calls'}},
            'first_action_difference':first_action_difference(*pair) if different and not missing else None,
            'first_feedback_difference':first_feedback_difference(*pair) if different and not missing else None})
    outcomes=[dict(Counter(r['outcome'] for r in side.values())) for side in evidence]
    succeeded=[{qid for qid,r in side.items() if r['outcome']=='answered'} for side in evidence]
    quality = None if scores is None else scores[1]['penalized_accuracy']>=scores[0]['penalized_accuracy']
    return {'format_version':2,'comparison_kind':'pipeline_deployment',
            'execution_complete_both':complete,'identity_compatible':identity_compatible,
            'paired_comparison_valid':valid,'gates':gates,'inference_speedup':speedup,'cold_speedup':cold_speedup,
            'observed_inference_time_ratio':observed_ratio,'observed_cold_time_ratio':observed_cold_ratio,
            'observed_ratio_note':'Descriptive wall-time ratios for all selected IDs, including aborted calls; not engine-only speedup or a quality guarantee.',
            'engine_only_speedup':None,'outcome_counts_baseline_candidate':outcomes,
            'answered_only_baseline_ids':sorted(succeeded[0]-succeeded[1]),
            'answered_only_candidate_ids':sorted(succeeded[1]-succeeded[0]),
            'per_question':per_question,
            'pipeline_config_differences':config_differences,'runtime_package_differences':package_differences,
            'initial_request_mismatches':mismatched_prompts,'answer_disagreement_ids':disagreements,
            'answer_agreement_fraction':1-len(disagreements)/len(expected_ids) if expected_ids else None,
            'scores_baseline_candidate':scores,'quality_not_worse_on_this_subset':quality,
            'evidence':('paired timing only; quality unverified without references' if scores is None else 'local subset timing and local evaluator; not official/private-test accuracy'),
            'baseline':metrics[0],'candidate':metrics[1],
            'warning':'24 diagnostic questions are not a statistical proof for all tasks. Repeat/extend a fixed sample when differences are small; never drop slow/failed questions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--references',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compare(args.baseline,args.candidate,references=args.references)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    display={k:v for k,v in report.items() if k not in {'baseline','candidate','per_question'}}
    print(json.dumps(display,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
