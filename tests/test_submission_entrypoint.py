"""Exercise runtime-only shell configuration with stub processes, without Docker."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_entrypoint_creates_writable_scratch_and_forwards_runtime_arguments(tmp_path):
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    bash = str(git_bash) if git_bash.is_file() else shutil.which('bash')
    if not bash:
        pytest.skip('Bash is needed to exercise the POSIX entrypoint')
    root = Path(__file__).resolve().parents[1]
    (tmp_path / 'predict.sh').write_bytes((root / 'predict.sh').read_bytes())
    timeout = tmp_path / 'timeout'
    timeout.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > watchdog-args.txt\nshift 3\nexec "$@"\n', encoding='utf-8', newline='\n')
    runtime = tmp_path / 'runtime-python'
    runtime.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > python-args.txt\n', encoding='utf-8', newline='\n')
    timeout.chmod(0o755)
    runtime.chmod(0o755)
    env = {**os.environ, 'DATAMATE_SCRATCH_DIR': 'scratch', 'DATAMATE_RESULT_DIR': 'results',
           'TMPDIR': 'scratch/temp', 'VLLM_CACHE_ROOT': 'scratch/compile',
           'VLLM_CONFIG_ROOT': 'scratch/config', 'DATAMATE_CONFIG': 'configs/local.json',
           'DATAMATE_WATCHDOG_SECONDS': '120', 'DATAMATE_KILL_AFTER_SECONDS': '5',
           'DATAMATE_RUNTIME_PYTHON': './runtime-python'}
    result = subprocess.run([bash, '-c', 'export PATH="$PWD:$PATH"; sh ./predict.sh --concurrency 2'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'watchdog-args.txt').read_text().splitlines()[:3] == ['--signal=TERM', '--kill-after=5s', '120s']
    assert (tmp_path / 'python-args.txt').read_text().splitlines() == [
        '-m', 'datamate', 'run', '--config', 'configs/local.json', '--concurrency', '2']
    assert all((tmp_path / name).is_dir() for name in ['results/logs', 'scratch/temp', 'scratch/compile', 'scratch/config'])
