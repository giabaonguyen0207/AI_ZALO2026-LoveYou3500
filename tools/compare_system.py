"""Run baseline, vLLM and Torch + RAG on one frozen set using current source."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT))
from datamate.config import Settings
from datamate.data.split import ensure_outside_dataset
from datamate.execution import run_checked_process
from tools.compare_runtime_benchmarks import compare


def save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def run(settings, ids, output, *, project_root=PROJECT, references=None,
        dtype='float16', tensor_parallel_size=1, vllm_workers=1, rag_mode='dense_rerank'):
    output = ensure_outside_dataset(Path(output).resolve(),settings.split_root)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a fresh comparison output directory')
    if not ids or any(not isinstance(q,str) or not q for q in ids) or len(set(ids))!=len(ids):
        raise ValueError('Provide a nonempty JSON array of unique question IDs')
    output.mkdir(parents=True,exist_ok=True)
    project_root = Path(project_root).resolve()
    selected = output/'selected_ids.json'
    save(selected,ids)
    timings = {}
    for arm,backend,retrieval in (('baseline','transformers','none'),('vllm','vllm','none'),('rag','transformers',rag_mode)):
        data = settings.to_dict()
        data.update(backend=backend,dtype=dtype,question_workers=vllm_workers if arm=='vllm' else 1,
                    result_dir=str(output/arm),cache_dir=str(output/'cache'/arm))
        data['rag'].update(mode=retrieval,index_dir=str(output/'cache'/arm/'retrieval'))
        data['vllm'].update(dtype=dtype,tensor_parallel_size=tensor_parallel_size,
            init_timeout_seconds=max(1200,data['vllm']['init_timeout_seconds']),
            max_num_seqs=vllm_workers,enable_prefix_caching=False,cache_prompt_prefix=False,group_by_business=False)
        resolved = Settings(**data).validate()
        config = output/f'{arm}.json'
        save(config,resolved.to_dict())
        command = [sys.executable,'-X','utf8',str(project_root/'tools/benchmark_runtime.py'),
                   '--project-root',str(project_root),'--config',str(config),'--ids',str(selected),'--output',str(output/arm)]
        env = {**os.environ,'PYTHONPATH':str(project_root/'src')+os.pathsep+str(project_root)+os.pathsep+os.environ.get('PYTHONPATH','')}
        started = time.monotonic()
        failure = None
        try:
            print(f'Running {arm}; log: {output/(arm+".log")}',flush=True)
            with (output/f'{arm}.log').open('w',encoding='utf-8') as log:
                run_checked_process(command,timeout=min(resolved.wall_time_seconds,21570),
                    startup_progress=output/arm/'startup_progress.json',cwd=project_root,env=env,
                    stdout=log,stderr=log)
        except BaseException as exc:
            failure = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            timings[arm] = {'process_wall_seconds':time.monotonic()-started,'error':failure}
            save(output/'process_timing.json',timings)
    # Labels are read only after every arm has finished inference.
    reports = {arm:compare(output/'baseline',output/arm,references=references,comparison_mode=mode)
               for arm,mode in (('vllm','backend'),('rag','rag'))}
    save(output/'comparisons.json',reports)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=PROJECT/'configs/comparison/baseline.json')
    p.add_argument('--ids',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--references',type=Path,help='Local gold file; never passed into inference')
    for name in ('split-root','model-dir','embedding-dir','reranker-dir','rag-device'):
        p.add_argument('--'+name)
    p.add_argument('--dtype',choices=['float16','bfloat16','float32'],default='float16')
    p.add_argument('--vllm-tensor-parallel-size',type=int,default=1)
    p.add_argument('--vllm-workers',type=int,default=1,help='Use 1 for the primary controlled comparison')
    p.add_argument('--rag-mode',choices=['dense','dense_rerank'],default='dense_rerank')
    args = p.parse_args()
    settings = Settings.load(args.config,use_env=False)
    for name in ('split_root','model_dir'):
        if getattr(args,name) is not None:
            setattr(settings,name,getattr(args,name))
    for argument,field in (('embedding_dir','embedding_dir'),('reranker_dir','reranker_dir'),('rag_device','device')):
        if getattr(args,argument) is not None:
            setattr(settings.rag,field,getattr(args,argument))
    ids = json.loads(args.ids.read_text(encoding='utf-8-sig'))
    if not isinstance(ids,list):
        p.error('--ids must contain a JSON array')
    reports = run(settings,ids,args.output,references=args.references,dtype=args.dtype,
                  tensor_parallel_size=args.vllm_tensor_parallel_size,vllm_workers=args.vllm_workers,rag_mode=args.rag_mode)
    print(json.dumps({arm:{key:r[key] for key in ('paired_comparison_valid','observed_inference_time_ratio',
        'observed_cold_time_ratio','scores_baseline_candidate','quality_not_worse_on_this_subset')} for arm,r in reports.items()},
        ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
