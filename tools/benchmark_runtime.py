"""Observe the selected source's real run_batch/agent calls; never replace the agent loop."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import wraps
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered)-1) * fraction
    low = int(index)
    return ordered[low] + (ordered[min(low+1,len(ordered)-1)]-ordered[low])*(index-low)


def package_versions():
    result = {}
    for name in ('torch','transformers','vllm','accelerate','bitsandbytes','tokenizers','psutil'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def override_loader_dtype(backends, dtype):
    """Configure only from_pretrained dtype, including on an immutable old source.

    Old loaders call is_bf16_supported() including emulation. Do not globally
    falsify that CUDA API, or alter the agent/backend generation algorithm.
    """
    original = backends.load_target
    @wraps(original)
    def load(loader, model_dir, **kwargs):
        import torch
        kwargs['dtype'] = getattr(torch,dtype)
        return original(loader,model_dir,**kwargs)
    backends.load_target = load
    return original


class Observer:
    def __init__(self, output, *, interval=1.0):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.interval = interval
        self.lock = threading.Lock()
        self.local = threading.local()
        self.active = self.peak_active = 0
        self.calls, self.questions = [], []
        self.backend_info = {}
        self.backend = None
        self.load_finished = None
        self.gpu_peak, self.gpu_util = {}, {}
        self.ram_peak = None
        self.hardware = {}
        self.stop_event = threading.Event()
        self.monitor = threading.Thread(target=self._monitor, daemon=True)

    def append(self, filename, row):
        with self.lock:
            with (self.output / filename).open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')

    def _sample(self):
        query = 'index,uuid,name,memory.used,memory.total,utilization.gpu'
        try:
            response = subprocess.run(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, check=True, timeout=5)
            for fields in csv.reader(response.stdout.splitlines(), skipinitialspace=True):
                index, uuid, name, used, total, utilization = fields
                row = {'seconds': time.monotonic()-self.started, 'index': index, 'uuid': uuid, 'name': name,
                       'used_mib': float(used), 'total_mib': float(total), 'utilization_percent': float(utilization)}
                self.hardware[index] = {'uuid': uuid, 'name': name, 'total_mib': float(total)}
                self.gpu_peak[index] = max(self.gpu_peak.get(index,0), row['used_mib'])
                self.gpu_util.setdefault(index,[]).append(row['utilization_percent'])
                self.append('gpu_samples.jsonl', row)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        try:
            import psutil
            process = psutil.Process()
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.Error:
                    pass
            self.ram_peak = max(self.ram_peak or 0, total)
        except Exception:
            pass

    def _monitor(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def start(self):
        self._sample()
        self.monitor.start()

    def close(self):
        self.stop_event.set()
        self.monitor.join(timeout=7)
        if not self.monitor.is_alive():
            self._sample()
        if self.backend is not None and callable(getattr(self.backend,'runtime_info',None)):
            self.backend_info['runtime'] = self.backend.runtime_info()

    def wrap_backend(self, backend):
        self.backend = backend
        generate = backend.generate

        @wraps(generate)
        def observed(messages, *args, **kwargs):
            started = time.monotonic()
            qid = getattr(self.local,'question_id',None)
            error, raw = None, None
            with self.lock:
                self.active += 1
                self.peak_active = max(self.peak_active,self.active)
            try:
                raw = generate(messages,*args,**kwargs)
                return raw
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                meta = dict(getattr(backend,'last_generation',None) or {})
                row = {'id': qid, 'start_seconds': started-self.started,
                       'duration_seconds': time.monotonic()-started,
                       'messages_sha256': hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                       'input_tokens': meta.get('input_tokens'), 'output_tokens': meta.get('output_tokens'),
                       'cached_tokens': meta.get('cached_tokens'), 'stop_reason': meta.get('stop_reason'), 'error':error,
                       'remaining_seconds_at_call': args[0] if args else kwargs.get('remaining_seconds'),
                       'output_sha256': hashlib.sha256(raw.encode()).hexdigest() if isinstance(raw,str) else None}
                with self.lock:
                    self.active -= 1
                    self.calls.append(row)
                self.append('model_calls.jsonl',row)
        backend.generate = observed
        model = getattr(backend,'model',None)
        self.backend_info['model_dtype'] = str(model.dtype) if getattr(model,'dtype',None) is not None else None
        self.backend_info['device_map'] = {k:str(v) for k,v in (getattr(model,'hf_device_map',{}) or {}).items()}
        runtime_info = getattr(backend,'runtime_info',None)
        if callable(runtime_info):
            self.backend_info['runtime'] = runtime_info()
        self.backend_info['class'] = type(backend).__name__
        return backend

    def wrap_agent_run(self, original):
        @wraps(original)
        def observed(agent, question, *args, **kwargs):
            self.local.question_id = question.id
            started, result, error = time.monotonic(), None, None
            try:
                result = original(agent, question, *args, **kwargs)
                return result
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                duration = time.monotonic()-started
                row = {'id':question.id,'business_id':question.business_id,
                       'start_seconds':started-self.started,'duration_seconds':duration,
                       'model_calls':getattr(result,'model_calls',None),
                       'answer_type':getattr(getattr(result,'answer',None),'type',None),'error':error,
                       'answer_reason':getattr(getattr(result,'answer',None),'reason',None),
                       'question_timeout_seconds':getattr(getattr(agent,'settings',None),'question_timeout_seconds',None)}
                with self.lock:
                    calls = [r for r in self.calls if r['id']==question.id]
                    row['first_model_delay_seconds'] = min((r['start_seconds'] for r in calls),default=None)
                    if row['first_model_delay_seconds'] is not None:
                        row['first_model_delay_seconds'] = max(0,row['first_model_delay_seconds']-row['start_seconds'])
                    row['generation_wall_seconds'] = sum(r['duration_seconds'] for r in calls)
                    row['other_wall_seconds'] = max(0,duration-row['generation_wall_seconds'])
                    self.questions.append(row)
                self.append('question_timing.jsonl',row)
                self.local.question_id = None
        return observed

    def summarize(self, duration, expected_ids, app_summary, error):
        latencies = [r['duration_seconds'] for r in self.questions if r['error'] is None]
        errors = [r['error'] for r in self.calls if r['error']]
        overflow = sum('context' in e.lower() or 'vượt giới hạn' in e.lower() for e in errors)
        oom = sum('out of memory' in e.lower() or 'cuda oom' in e.lower() for e in errors)
        timed_out = sum(('timeout' in (r['error'] or '').lower() or 'hết thời gian' in (r['error'] or '').lower()
                         or r['stop_reason']=='timeout_aborted') for r in self.calls)
        possible_time_stops = sum(r['stop_reason']=='time_or_other' for r in self.calls)
        known_input = [r['input_tokens'] for r in self.calls if type(r['input_tokens']) is int]
        known_output = [r['output_tokens'] for r in self.calls if type(r['output_tokens']) is int]
        cache_rows = [r for r in self.calls if type(r['cached_tokens']) is int and type(r['input_tokens']) is int]
        inference = duration-(self.load_finished-self.started) if self.load_finished is not None else None
        written = app_summary.get('written',0)
        device_map = self.backend_info.get('device_map',{})
        cpu_offload = any(value in {'cpu','disk'} for value in device_map.values())
        valid = (not error and written == len(expected_ids) and not errors and not possible_time_stops and not cpu_offload
                 and not app_summary.get('time_budget_exhausted'))
        execution_complete = (not error and not app_summary.get('error') and written==len(expected_ids)
                              and {r['id'] for r in self.questions}==set(expected_ids)
                              and len(self.questions)==len(expected_ids) and not app_summary.get('time_budget_exhausted'))
        warnings = [name for name,present in (
            ('generation_timeouts',timed_out),('possible_generation_time_stops',possible_time_stops),
            ('generation_errors',max(0,len(errors)-timed_out)),('cpu_or_disk_offload',cpu_offload),
            ('all_abstain',bool(self.questions) and all(r['answer_type']=='abstain' for r in self.questions))) if present]
        return {'format_version':2,'execution_complete':execution_complete,'measurement_warnings':warnings,
                'valid_complete_run':valid,'error':error,'expected_questions':len(expected_ids),'written':written,
                'cold_total_seconds':duration,'load_seconds':None if self.load_finished is None else self.load_finished-self.started,
                'inference_seconds':inference,'questions_per_minute':None if not inference else written*60/inference,
                'latency_p50_seconds':percentile(latencies,.5),'latency_p95_seconds':percentile(latencies,.95),
                'model_calls':len(self.calls),'input_tokens':sum(known_input),'output_tokens':sum(known_output),
                'output_tokens_per_second':None if not inference else sum(known_output)/inference,
                'token_metadata_complete':len(known_input)==len(self.calls)==len(known_output),
                'cache_observed_requests':len(cache_rows), 'cached_tokens':sum(r['cached_tokens'] for r in cache_rows) if cache_rows else None,
                'cache_hit_ratio_observed_requests':sum(r['cached_tokens'] for r in cache_rows)/sum(r['input_tokens'] for r in cache_rows) if cache_rows and sum(r['input_tokens'] for r in cache_rows) else None,
                'context_overflow_calls':overflow,'oom_calls':oom,'timeout_calls':timed_out,
                'possible_time_stop_calls':possible_time_stops,
                'generation_error_calls':len(errors),'length_stop_calls':sum(r['stop_reason'] in {'length','max_new_tokens'} for r in self.calls),
                'generation_requests_peak':self.peak_active,'answer_types':dict(Counter(r['answer_type'] for r in self.questions)),
                'gpu_peak_used_mib':self.gpu_peak,
                'gpu_mean_utilization_percent':{k:statistics.mean(v) for k,v in self.gpu_util.items()},
                'process_tree_peak_rss_bytes_sampled':self.ram_peak,'gpu_hardware':self.hardware,
                'cpu_or_disk_offload_detected':cpu_offload,'backend_info':self.backend_info,
                'timing_policy':'new process/engine; filesystem cache state is not controlled; model download/install excluded; no artificial warmup responses',
                'latency_note':'per-question wall latency; concurrent latencies overlap. Throughput divides by batch wall span, never by their sum.'}


def run(project, config, ids_path, output, *, interval=1.0, transformers_dtype=None):
    project, output = Path(project).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory; benchmark runs never resume/overwrite')
    sys.path.insert(0,str(project/'src'))
    import datamate
    from datamate import application
    from datamate.config import Settings
    from datamate.models import backends
    from datamate.data.split import ensure_outside_dataset
    if not Path(datamate.__file__).resolve().is_relative_to(project):
        raise ValueError('Wrong source imported; start a fresh process')
    options = {'use_env':False} if 'use_env' in inspect.signature(Settings.load).parameters else {}
    settings = Settings.load(config,**options)
    settings.result_dir = str(ensure_outside_dataset(output,settings.split_root))
    ids = json.loads(Path(ids_path).read_text(encoding='utf-8-sig'))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Nonempty unique benchmark IDs are required')
    observer = Observer(output,interval=interval)
    identity = {'source_root':str(project),'settings':settings.to_dict(),'ids':ids,'ids_sha256':sha256(ids_path),
                'transformers_loader_dtype_override':transformers_dtype,
                'model_manifest_sha256':sha256(Path(settings.model_dir)/'artifact_manifest.json') if settings.backend!='demo' else None,
                'packages':package_versions(),'python':sys.version,
                'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'source_hashes':{p.relative_to(project/'src').as_posix():sha256(p) for p in sorted((project/'src/datamate').rglob('*.py'))}}
    (output/'benchmark_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    original_build, original_create, original_run = application.build_backend, application.create_agent, application.Agent.run
    original_load = override_loader_dtype(backends,transformers_dtype) if transformers_dtype and settings.backend=='transformers' else None
    def build(*args,**kwargs):
        return observer.wrap_backend(original_build(*args,**kwargs))
    def create(*args,**kwargs):
        agent = original_create(*args,**kwargs)
        observer.load_finished = time.monotonic()
        return agent
    application.build_backend = build
    application.create_agent = create
    application.Agent.run = observer.wrap_agent_run(original_run)
    app_summary, error = {}, None
    observer.start()
    # Identity creation and initial telemetry are excluded for both arms;
    # run_batch itself starts before model validation/loading/ingestion.
    observer.started = time.monotonic()
    try:
        app_summary = application.run_batch(settings,set(ids))
        return app_summary
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        duration = time.monotonic()-observer.started
        observer.close()
        application.build_backend, application.create_agent, application.Agent.run = original_build,original_create,original_run
        if original_load is not None:
            backends.load_target = original_load
        report = observer.summarize(duration,ids,app_summary,error)
        (output/'benchmark_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2,default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--ids',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-interval',type=float,default=1.0)
    parser.add_argument('--transformers-dtype',choices=['float16','bfloat16','float32'],
                        help='Explicit benchmark loader precision override; recorded without editing the frozen source')
    args = parser.parse_args()
    if args.sample_interval < .2:
        parser.error('Telemetry interval must be >=0.2 seconds')
    run(args.project_root,args.config,args.ids,args.output,interval=args.sample_interval,transformers_dtype=args.transformers_dtype)


if __name__ == '__main__':
    main()
