"""Frozen validation sampling, resumable inference and paired SQL-SFT review."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import statistics
import time
import uuid

from datamate.application import create_agent
from datamate.config import Settings
from datamate.data.split import DatasetSplit, read_jsonl, ensure_outside_dataset
from datamate.debug_log import write_debug_log
from datamate.domain import Answer, DataMateError
from datamate.evaluation.scoring import score
from datamate.models.artifacts import sha256 as hash_file, verify_artifact
from datamate.training.checkpoints import atomic_json as write_json, training_lock


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':')).encode('utf-8')


def select_questions(questions, validation_ids, size=100, seed=42):
    by_id={q['id']:q for q in questions}
    if len(by_id)!=len(questions) or len(set(validation_ids))!=len(validation_ids):
        raise DataMateError('Trùng ID nguồn/validation')
    if not set(validation_ids)<=set(by_id) or not 1<=size<=len(validation_ids):
        raise DataMateError('Không đủ ID validation hợp lệ')
    groups=defaultdict(list)
    for qid in validation_ids:groups[by_id[qid]['business_id']].append(by_id[qid])
    total=len(validation_ids)
    quota={b:len(rows)*size//total for b,rows in groups.items()}
    order=sorted(groups,key=lambda b:(-(len(groups[b])*size%total),b))
    for b in order[:size-sum(quota.values())]:quota[b]+=1
    def rank(q):return hashlib.sha256(f'{seed}:{q["id"]}'.encode()).hexdigest()
    selected=[q for b in sorted(groups) for q in sorted(groups[b],key=rank)[:quota[b]]]
    return sorted(selected,key=rank)


def verify_benchmark(root, *, include_references=True):
    root=Path(root).resolve()
    manifest=json.loads((root/'benchmark_manifest.json').read_text(encoding='utf-8'))
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(manifest['files'])|{'benchmark_manifest.json'}:
        raise DataMateError('Benchmark có file thừa/thiếu')
    for name,digest in manifest['files'].items():
        if not include_references and name.startswith('evaluation/'):continue
        path=(root/name).resolve()
        if not path.is_relative_to(root) or hash_file(path)!=digest:
            raise DataMateError('Benchmark sai checksum: '+name)
    if any(p.name=='answers.jsonl' for p in (root/'input').rglob('*')):
        raise DataMateError('Inference split chứa gold answers')
    ids=json.loads((root/'question_ids.json').read_text(encoding='utf-8'))
    questions=read_jsonl(root/'input/questions.jsonl')
    if len(ids)!=manifest['sample_size'] or len(set(ids))!=len(ids) or [q['id'] for q in questions]!=ids:
        raise DataMateError('Benchmark khác danh sách/thứ tự ID')
    return manifest


def runtime_identity():
    import torch
    return {'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'packages':{name:importlib.metadata.version(name) for name in
                        ('torch','transformers','accelerate','peft','bitsandbytes','flash-linear-attention','fla-core')}}


def read_records(folder, questions):
    allowed={q.id for q in questions}; records={}
    for path in sorted(Path(folder).glob('*.json')):
        sealed=json.loads(path.read_text(encoding='utf-8'))
        payload=sealed['payload'];qid=payload['id']
        if (sealed['sha256']!=hashlib.sha256(canonical(payload)).hexdigest()
                or qid not in allowed or qid in records
                or path.stem!=hashlib.sha256(qid.encode()).hexdigest()
                or payload['submission'].get('id')!=qid):
            raise DataMateError('Log resume sai checksum/ID: '+str(path))
        Answer.from_dict(payload['submission'])
        records[qid]=payload
    return records


def run_variant(config_path, benchmark):
    config=json.loads(Path(config_path).read_text(encoding='utf-8'))
    settings=Settings(**config).validate()
    with training_lock(settings.result_dir):
        return _run_variant(settings,benchmark)


def _run_variant(settings, benchmark):
    benchmark=Path(benchmark).resolve();manifest=verify_benchmark(benchmark,include_references=False)
    split=DatasetSplit(settings.split_root)
    if split.root!=(benchmark/'input').resolve():raise DataMateError('Sai inference split')
    output=ensure_outside_dataset(settings.result_dir,split.root)
    ensure_outside_dataset(settings.cache_dir,split.root)
    output.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).resolve().parents[1]/'src/datamate'
    identity={'version':1,'settings':settings.to_dict(),'benchmark_sha256':hash_file(benchmark/'benchmark_manifest.json'),
              'model_manifest_sha256':hash_file(Path(settings.model_dir)/'artifact_manifest.json'),
              'runner_sha256':hash_file(Path(__file__)),'runtime':runtime_identity(),
              'runtime_source':{p.relative_to(source).as_posix():hash_file(p) for p in sorted(source.rglob('*.py'))}}
    ip=output/'evaluation_identity.json'
    if ip.exists():
        if json.loads(ip.read_text(encoding='utf-8'))!=identity:raise DataMateError('Resume identity khác; dùng RUN_NAME mới')
    elif any(output.iterdir()):raise DataMateError('Output thiếu identity; không ghi đè')
    else:write_json(ip,identity)
    logs=output/'records';logs.mkdir(exist_ok=True)
    questions=split.questions();records=read_records(logs,questions)
    ledger_path=output/'sessions.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else []
    if len(records)<len(questions):
        started=time.monotonic()
        session={'id':uuid.uuid4().hex,'closed':False,'load_seconds':None,'elapsed_seconds':0}
        ledger.append(session);write_json(ledger_path,ledger)
        from transformers import set_seed
        set_seed(settings.seed)
        agent=create_agent(settings)
        session['load_seconds']=time.monotonic()-started
        session['elapsed_seconds']=session['load_seconds'];write_json(ledger_path,ledger)
        # No gold/reference files are read by this process.
        for index,q in enumerate(questions,1):
            if q.id in records:continue
            remaining=settings.wall_time_seconds-(time.monotonic()-started)
            if remaining<=0:break
            print(f'[{index}/{len(questions)}] {q.id} | {q.question}',flush=True)
            run=agent.run(q,remaining)
            payload={'id':q.id,'business_id':q.business_id,**q.prompt_dict(),**run.to_dict(q.id)}
            write_json(logs/(hashlib.sha256(q.id.encode()).hexdigest()+'.json'),
                       {'payload':payload,'sha256':hashlib.sha256(canonical(payload)).hexdigest()})
            records[q.id]=payload
            session['elapsed_seconds']=time.monotonic()-started;write_json(ledger_path,ledger)
            print(' ->',payload['submission'],'seconds:',payload['elapsed_seconds'],flush=True)
        session['closed']=True;session['elapsed_seconds']=time.monotonic()-started;write_json(ledger_path,ledger)
    ordered=[records[q.id] for q in questions if q.id in records]
    temporary=output/('submission.jsonl.tmp-'+uuid.uuid4().hex)
    temporary.write_text(''.join(json.dumps(r['submission'],ensure_ascii=False)+'\n' for r in ordered),encoding='utf-8')
    temporary.replace(output/'submission.jsonl')
    summary={'completed':len(ordered),'planned':manifest['sample_size'],
             'status':'complete' if len(ordered)==len(questions) else 'partial',
             'known_process_seconds':sum(s['elapsed_seconds'] for s in ledger),
             'wall_seconds':sum(s['elapsed_seconds'] for s in ledger) if all(s['closed'] for s in ledger) else None,
             'timing_incomplete_after_runtime_loss':any(not s['closed'] for s in ledger),
             'load_seconds':sum(s['load_seconds'] or 0 for s in ledger),
             'model_calls':sum(r['model_calls'] for r in ordered),
             'mean_question_seconds':statistics.mean(r['elapsed_seconds'] for r in ordered) if ordered else None,
             'p95_question_seconds':sorted(r['elapsed_seconds'] for r in ordered)[max(0,math.ceil(.95*len(ordered))-1)] if ordered else None}
    write_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def paired_report(benchmark, baseline, finetuned, output):
    benchmark=Path(benchmark);verify_benchmark(benchmark)
    ids=json.loads((benchmark/'question_ids.json').read_text(encoding='utf-8'))
    questions=DatasetSplit(benchmark/'input').questions()
    paths=[Path(baseline),Path(finetuned)]; identities=[];records=[];summaries=[]
    for folder in paths:
        identities.append(json.loads((folder/'evaluation_identity.json').read_text(encoding='utf-8')))
        records.append(read_records(folder/'records',questions))
        if set(records[-1])!=set(ids):raise DataMateError('Chưa đủ câu ở cả hai model; chạy lại ô inference để resume')
        summaries.append(json.loads((folder/'summary.json').read_text(encoding='utf-8')))
    a,b=identities
    current_benchmark_sha=hash_file(benchmark/'benchmark_manifest.json')
    if any(identity['benchmark_sha256']!=current_benchmark_sha for identity in identities):
        raise DataMateError('Run không thuộc benchmark hiện tại')
    for key in ('benchmark_sha256','runner_sha256','runtime_source','runtime'):
        if a[key]!=b[key]:raise DataMateError('Đối chứng khác '+key)
    def shared(config):return {k:v for k,v in config.items() if k not in {'model_dir','result_dir','cache_dir'}}
    if shared(a['settings'])!=shared(b['settings']):raise DataMateError('Hai model khác cấu hình inference')
    if a['model_manifest_sha256']==b['model_manifest_sha256']:
        raise DataMateError('Hai lượt đang dùng cùng model artifact')
    refs=read_jsonl(benchmark/'evaluation/references.jsonl')
    reports=[score([rows[i]['submission'] for i in ids],refs,set(ids)) for rows in records]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics={label:{**{k:v for k,v in report.items() if k!='details'},**summary}
             for label,report,summary in zip(('baseline','finetuned'),reports,summaries)}
    metrics['delta_pa']=reports[1]['penalized_accuracy']-reports[0]['penalized_accuracy']
    details=[{r['id']:r for r in report['details']} for report in reports]
    paired=[]
    for q in questions:
        row={'id':q.id,'business_id':q.business_id,**q.prompt_dict(),
             'baseline':records[0][q.id]['submission'],'finetuned':records[1][q.id]['submission'],
             'baseline_result':details[0][q.id]['result'],'finetuned_result':details[1][q.id]['result'],
             'delta_points':details[1][q.id]['points']-details[0][q.id]['points']}
        paired.append(row)
        for label,record in zip(('baseline','finetuned'),records):
            dest=output/label;dest.mkdir(exist_ok=True)
            write_debug_log(dest/(q.id+'.md'),record[q.id])
    write_json(output/'comparison.json',metrics)
    write_json(output/'paired_review.json',paired)
    write_json(output/'regressions.json',[r for r in paired if r['baseline_result']=='correct' and r['finetuned_result']!='correct'])
    print(json.dumps(metrics,ensure_ascii=False,indent=2));return metrics


def stage_local(source, target):
    source,target=Path(source).resolve(),Path(target).resolve()
    if (target/'artifact_manifest.json').exists():
        verify_artifact(target,'qwen35-9b')
        if hash_file(target/'artifact_manifest.json')!=hash_file(source/'artifact_manifest.json'):
            raise DataMateError('Local model cache khác nguồn')
        return target
    verify_artifact(source,'qwen35-9b')
    manifest=json.loads((source/'artifact_manifest.json').read_text(encoding='utf-8'))
    target.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(target.parent).free<sum(v['bytes'] for v in manifest['files'].values())+1024**3:
        raise DataMateError('Thiếu local SSD để stage model')
    for name in [*manifest['files'],'artifact_manifest.json']:
        src=(source/name).resolve();dst=(target/name).resolve()
        if not src.is_relative_to(source) or not dst.is_relative_to(target):raise DataMateError('Model path vượt thư mục')
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    verify_artifact(target,'qwen35-9b');return target


def main():
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run');run.add_argument('--config',required=True);run.add_argument('--benchmark',required=True)
    report=commands.add_parser('report')
    for name in ('benchmark','baseline','finetuned','output'):report.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if args.command=='run':run_variant(args.config,args.benchmark)
    else:paired_report(args.benchmark,args.baseline,args.finetuned,args.output)


if __name__=='__main__':main()
