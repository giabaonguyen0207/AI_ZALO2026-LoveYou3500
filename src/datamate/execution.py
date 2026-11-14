"""Process guard for GPU batches; the worker still calls application.run_batch."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from .data.split import DatasetSplit, ensure_outside_dataset
from .domain import DataMateError, json_text


def stop_process_group(process, grace_seconds=2.0):
    """Reap our child and, on Linux, its group even if the parent exited."""
    if os.name == 'posix':
        def alive():
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False
        if alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + grace_seconds
                while alive() and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(0.02)
                if alive():
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    elif process.poll() is None:
        # Windows is used for CPU fixtures. GPU deployment uses Linux groups.
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if process.poll() is None:
            process.kill()
    try:
        process.wait(timeout=max(1, grace_seconds))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def run_checked_process(command, *, timeout, startup_progress=None, grace_seconds=2.0, **options):
    if not math.isfinite(timeout) or timeout <= 0:
        raise DataMateError('Inference wall budget exhausted before process start')
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=os.name == 'posix', stdin=subprocess.DEVNULL, **options)
    previous = None
    if os.name == 'posix' and threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)
        def interrupted(signum, frame):
            raise DataMateError('Inference supervisor received SIGTERM')
        signal.signal(signal.SIGTERM, interrupted)
    try:
        while process.poll() is None:
            now = time.monotonic()
            if now-started >= timeout:
                raise DataMateError('Inference process timeout: wall budget exhausted')
            if startup_progress is not None:
                try:
                    state = json.loads(Path(startup_progress).read_text(encoding='utf-8'))
                    if (state.get('phase') == 'generator_init'
                            and now > float(state['monotonic_started']) + float(state['timeout_seconds'])):
                        raise DataMateError('Inference startup timeout: generator initialization budget exhausted')
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            time.sleep(min(0.05, max(0, timeout - (now-started))))
        if process.returncode != 0:
            raise DataMateError(f'Inference worker failed (exit {process.returncode}); xem logs/run_summary')
        return subprocess.CompletedProcess(command, process.returncode)
    finally:
        try:
            stop_process_group(process, grace_seconds)
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)


def run_isolated(settings, ids=None, limit=None, resume=False):
    settings.validate()
    started = time.monotonic()
    split = DatasetSplit(settings.split_root)
    output = ensure_outside_dataset(settings.result_dir, split.root)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'submission.jsonl').exists() and not resume:
        raise DataMateError('submission.jsonl đã có; chọn result_dir mới hoặc --resume')
    config = output / 'settings.resolved.json'
    config.write_text(json_text(settings.to_dict(), indent=2), encoding='utf-8')
    progress = output / 'startup_progress.json'
    progress.write_text(json_text({'phase': 'starting', 'monotonic_started': time.monotonic()}), encoding='utf-8')
    command = [sys.executable, '-X', 'utf8', '-m', 'datamate', 'run', '--config', str(config)]
    if ids is not None:
        selected = output / 'selected_ids.json'
        selected.write_text(json_text(sorted(ids)), encoding='utf-8')
        command += ['--ids', str(selected)]
    if limit is not None:
        command += ['--limit', str(limit)]
    if resume:
        command.append('--resume')
    env = {**os.environ, 'DATAMATE_INFERENCE_WORKER': '1',
           'PYTHONPATH': str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    failure = None
    try:
        run_checked_process(command, timeout=min(settings.wall_time_seconds, 21570),
                            startup_progress=progress, env=env)
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        elapsed = time.monotonic()-started
        (output / 'supervisor.json').write_text(json_text({'process_wall_seconds': elapsed,
            'error': failure, 'startup_budget_seconds': settings.vllm.init_timeout_seconds
            if settings.backend == 'vllm' else settings.wall_time_seconds}, indent=2), encoding='utf-8')
    summary_file = output / 'run_summary.json'
    if not summary_file.is_file():
        raise DataMateError('Inference worker returned without run_summary.json')
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    summary['process_wall_seconds'] = elapsed
    summary_file.write_text(json_text(summary, indent=2), encoding='utf-8')
    return summary
