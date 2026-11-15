"""Retry vLLM/RAG using an existing completed baseline and its original source.

Can be uploaded as a standalone script to Colab. It changes engine configuration,
not the original source/model/baseline. All new files go in a fresh subdirectory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_baseline(project, comparison):
    baseline = comparison/'baseline'
    identity = read(baseline/'benchmark_identity.json')
    metrics = read(baseline/'benchmark_metrics.json')
    ids = identity['ids']
    if (not metrics.get('execution_complete') or metrics.get('error')
            or not ids or len(set(ids)) != len(ids)):
        raise ValueError('Baseline is incomplete; do not reuse it as a completed run')
    predictions = [json.loads(line) for line in (baseline/'submission.jsonl').read_text(
        encoding='utf-8-sig').splitlines() if line.strip()]
    if len(predictions) != len(ids) or {p['id'] for p in predictions} != set(ids):
        raise ValueError('Baseline predictions do not match its frozen IDs')
    selected = comparison/'selected_ids.json'
    if read(selected) != ids or digest(selected) != identity['ids_sha256']:
        raise ValueError('The original selected_ids.json has changed')
    source = {p.relative_to(project/'src').as_posix(): digest(p)
              for p in sorted((project/'src/datamate').rglob('*.py'))}
    if not source or source != identity.get('source_hashes'):
        raise ValueError('Use the exact original WORK source that ran baseline; do not patch it')
    settings = identity['settings']
    if settings['backend'] != 'transformers' or settings['rag']['mode'] != 'none':
        raise ValueError('Expected a Torch baseline with RAG disabled')
    if digest(Path(settings['model_dir'])/'artifact_manifest.json') != identity['model_manifest_sha256']:
        raise ValueError('Generator manifest differs from baseline')
    if not metrics.get('valid_complete_run'):
        print('WARNING: baseline completed with measurement warnings; see benchmark_metrics.json', flush=True)
    return identity, selected


def run_logged(command, path, *, runner, **options):
    """Keep the existing process watchdog and mirror its file log to Colab."""
    stop = threading.Event()

    def follow():
        with path.open(encoding='utf-8', errors='replace') as reader:
            while True:
                line = reader.readline()
                if line:
                    print(line, end='', flush=True)
                elif stop.is_set():
                    break
                else:
                    stop.wait(0.1)

    with path.open('w', encoding='utf-8') as log:
        thread = threading.Thread(target=follow, daemon=True)
        thread.start()
        try:
            return runner(command, stdout=log, stderr=log, **options)
        finally:
            log.flush()
            stop.set()
            thread.join(timeout=3)


def worker_environment(project):
    # Running /venv/bin/python does not activate the venv's executable PATH.
    # FlashInfer invokes ninja by name; preserve the venv path even when its
    # Python executable is a symlink to the system interpreter.
    executable_dir = Path(sys.executable).parent
    env = {**os.environ, 'PYTHONUNBUFFERED': '1',
           'PYTHONPATH': str(project/'src')+os.pathsep+str(project),
           'PATH': str(executable_dir)+os.pathsep+os.environ.get('PATH', '')}
    if (executable_dir.parent/'pyvenv.cfg').is_file():
        env['VIRTUAL_ENV'] = str(executable_dir.parent)
    return env


def run(project, comparison, *, arms=('vllm', 'rag'), references=None):
    project, comparison = Path(project).resolve(), Path(comparison).resolve()
    if not arms or len(set(arms)) != len(arms) or set(arms)-{'vllm', 'rag'}:
        raise ValueError('Choose vllm and/or rag, each at most once')
    identity, selected = validate_baseline(project, comparison)
    # Import the original selected source, not a newly downloaded project.
    sys.path[:0] = [str(project/'src'), str(project)]
    from datamate.config import Settings
    from datamate.execution import run_checked_process
    from tools.compare_runtime_benchmarks import compare
    from tools.benchmark_runtime import package_versions

    if package_versions() != identity['packages']:
        raise ValueError('Runtime package versions differ from baseline; use its RUNTIME_PYTHON')
    output = comparison/('recovery-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(exist_ok=False)
    (output/'selected_ids.json').write_bytes(selected.read_bytes())
    selected = output/'selected_ids.json'
    print('Recovery output:', output, flush=True)
    timings, reports = {}, {}
    for arm in arms:
        data = deepcopy(identity['settings'])
        data.update(backend='vllm' if arm == 'vllm' else 'transformers',
                    question_workers=1, result_dir=str(output/arm), cache_dir=str(output/'cache'/arm))
        data['rag'].update(mode='none' if arm == 'vllm' else 'dense_rerank',
                           index_dir=str(output/'cache'/arm/'retrieval'))
        data['vllm'].update(max_num_seqs=1, enable_prefix_caching=False,
                            cache_prompt_prefix=False, group_by_business=False)
        data['vllm']['init_timeout_seconds'] = max(1200, data['vllm']['init_timeout_seconds'])
        if arm == 'vllm':
            data['vllm']['extra_engine_args']['language_model_only'] = True
        settings = Settings(**data).validate()
        if settings.vllm.extra_sampling_params:
            raise ValueError('Controlled comparison requires the original greedy defaults')
        config = output/f'{arm}.json'
        save(config, settings.to_dict())
        command = [sys.executable, '-u', '-X', 'utf8', str(project/'tools/benchmark_runtime.py'),
                   '--project-root', str(project), '--config', str(config),
                   '--ids', str(selected), '--output', str(output/arm)]
        env = worker_environment(project)
        print(f'Running {arm}; baseline is reused; log: {output/(arm+".log")}', flush=True)
        started, failure = time.monotonic(), None
        try:
            run_logged(command, output/f'{arm}.log', runner=run_checked_process,
                       timeout=min(settings.wall_time_seconds, 21570),
                       startup_progress=output/arm/'startup_progress.json', cwd=project, env=env)
        except BaseException as exc:
            failure = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            timings[arm] = {'process_wall_seconds': time.monotonic()-started, 'error': failure}
            save(output/'process_timing.json', timings)
    # Gold is accessed only after all requested inference jobs have finished.
    for arm in arms:
        reports[arm] = compare(comparison/'baseline', output/arm, references=references,
                               comparison_mode='backend' if arm == 'vllm' else 'rag')
    save(output/'comparisons.json', reports)
    print('Results:', output/'comparisons.json', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--comparison-dir', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=['vllm', 'rag'], default=['vllm', 'rag'])
    parser.add_argument('--references', type=Path)
    args = parser.parse_args()
    run(args.project_root, args.comparison_dir, arms=args.arms, references=args.references)


if __name__ == '__main__':
    main()
