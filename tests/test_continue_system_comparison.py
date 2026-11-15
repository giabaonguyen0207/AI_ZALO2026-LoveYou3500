import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from datamate.config import Settings
from tools import continue_system_comparison as tool
from tools.benchmark_runtime import package_versions


PROJECT = Path(__file__).resolve().parents[1]


def test_worker_can_find_venv_build_executable_without_activation(tmp_path, monkeypatch):
    executable_dir = tmp_path/'private-env'/('Scripts' if os.name=='nt' else 'bin')
    executable_dir.mkdir(parents=True)
    (executable_dir.parent/'pyvenv.cfg').touch()
    ninja = executable_dir/('ninja.exe' if os.name=='nt' else 'ninja')
    ninja.touch()
    ninja.chmod(0o755)
    monkeypatch.setattr(sys, 'executable', str(executable_dir/'python'))
    monkeypatch.setenv('PATH', str(tmp_path/'unrelated-bin'))
    env = tool.worker_environment(PROJECT)
    assert Path(shutil.which('ninja', path=env['PATH'])) == ninja
    assert env['VIRTUAL_ENV'] == str(executable_dir.parent)
    assert os.environ['PATH'] == str(tmp_path/'unrelated-bin')


@pytest.fixture
def completed_baseline(tmp_path):
    comparison = tmp_path/'comparison'
    baseline = comparison/'baseline'
    baseline.mkdir(parents=True)
    model = tmp_path/'model'
    model.mkdir()
    tool.save(model/'artifact_manifest.json', {'fixture': True})
    settings = Settings(model_dir=str(model), dtype='float16', question_workers=1)
    tool.save(comparison/'selected_ids.json', ['q1', 'q2'])
    identity = {'ids': ['q1', 'q2'], 'settings': settings.to_dict(),
                'packages': package_versions(),
                'model_manifest_sha256': tool.digest(model/'artifact_manifest.json'),
                'ids_sha256': tool.digest(comparison/'selected_ids.json'),
                'source_hashes': {p.relative_to(PROJECT/'src').as_posix(): tool.digest(p)
                                  for p in (PROJECT/'src/datamate').rglob('*.py')}}
    tool.save(baseline/'benchmark_identity.json', identity)
    tool.save(baseline/'benchmark_metrics.json', {
        'execution_complete': True, 'valid_complete_run': True, 'error': None})
    (baseline/'submission.jsonl').write_text('\n'.join(
        json.dumps({'id': q, 'type': 'number', 'answer': 1}) for q in identity['ids']))
    (comparison/'vllm.log').write_text('original failed vLLM log')
    return comparison


def test_recovery_preserves_baseline_and_retries_only_requested_arms(completed_baseline, monkeypatch):
    from datamate import execution
    from tools import compare_runtime_benchmarks
    original = {p: p.read_bytes() for p in completed_baseline.rglob('*') if p.is_file()}
    events, configs = [], []

    def process(command, **options):
        config = tool.read(command[command.index('--config')+1])
        configs.append(config)
        events.append(config['backend'])
        assert '--references' not in command
        assert tool.read(command[command.index('--ids')+1]) == ['q1', 'q2']
        assert options['startup_progress'] == Path(config['result_dir'])/'startup_progress.json'

    def compare(baseline, candidate, **options):
        assert events == ['vllm', 'transformers']  # no scoring between inference arms
        assert baseline == completed_baseline/'baseline'
        return {'mode': options['comparison_mode'], 'fixture': True}

    monkeypatch.setattr(execution, 'run_checked_process', process)
    monkeypatch.setattr(compare_runtime_benchmarks, 'compare', compare)
    output = tool.run(PROJECT, completed_baseline)
    assert output.parent == completed_baseline
    assert all(path.read_bytes() == value for path, value in original.items())
    assert [(c['backend'], c['rag']['mode']) for c in configs] == [
        ('vllm', 'none'), ('transformers', 'dense_rerank')]
    assert configs[0]['vllm']['extra_engine_args']['language_model_only'] is True
    baseline_settings = tool.read(completed_baseline/'baseline/benchmark_identity.json')['settings']
    for config in configs:
        for key in ('model_dir', 'dtype', 'seed', 'max_context_tokens', 'max_new_tokens',
                    'question_timeout_seconds', 'max_steps'):
            assert config[key] == baseline_settings[key]
        assert Path(config['result_dir']).is_relative_to(output)
        assert Path(config['cache_dir']).is_relative_to(output)
    assert tool.read(output/'comparisons.json')['rag']['mode'] == 'rag'


@pytest.mark.parametrize('change', ['incomplete', 'ids', 'source', 'model', 'packages'])
def test_recovery_rejects_changed_or_incomplete_baseline_before_starting(completed_baseline, change):
    baseline = completed_baseline/'baseline'
    if change == 'incomplete':
        tool.save(baseline/'benchmark_metrics.json', {'execution_complete': False})
    elif change == 'ids':
        tool.save(completed_baseline/'selected_ids.json', ['q1'])
    else:
        identity = tool.read(baseline/'benchmark_identity.json')
        if change == 'source':
            identity['source_hashes']['datamate/models/vllm_backend.py'] = 'changed'
        elif change == 'model':
            identity['model_manifest_sha256'] = 'changed'
        else:
            identity['packages']['torch'] = 'changed'
        tool.save(baseline/'benchmark_identity.json', identity)
    with pytest.raises(ValueError):
        tool.run(PROJECT, completed_baseline)
    assert not list(completed_baseline.glob('recovery-*'))


def test_log_is_visible_before_worker_finishes_and_retains_stderr(tmp_path, monkeypatch):
    from datamate.execution import run_checked_process
    ack = tmp_path/'ack'
    seen = []

    def display(*args, **kwargs):
        text = str(args[0])
        seen.append(text)
        if 'ready' in text:
            ack.touch()

    monkeypatch.setattr(tool, 'print', display, raising=False)
    script = (
        'import sys,time; from pathlib import Path; '
        f'p=Path({str(ack)!r}); '
        'print("ready",flush=True); print("stderr message",file=sys.stderr,flush=True); '
        'deadline=time.monotonic()+3\n'
        'while not p.exists() and time.monotonic()<deadline: time.sleep(.02)\n'
        'sys.exit(0 if p.exists() else 9)'
    )
    log = tmp_path/'worker.log'
    tool.run_logged([sys.executable, '-u', '-c', script], log,
                    runner=run_checked_process, timeout=5, grace_seconds=.1)
    assert ack.exists() and any('ready' in line for line in seen)
    assert 'stderr message' in log.read_text()


def test_failed_recovery_preserves_error_and_does_not_start_rag(completed_baseline, monkeypatch):
    from datamate import execution
    calls = []

    def fail(command, **options):
        calls.append(command)
        raise RuntimeError('engine boot failed')

    monkeypatch.setattr(execution, 'run_checked_process', fail)
    with pytest.raises(RuntimeError, match='engine boot failed'):
        tool.run(PROJECT, completed_baseline)
    assert len(calls) == 1
    output, = completed_baseline.glob('recovery-*')
    assert 'engine boot failed' in tool.read(output/'process_timing.json')['vllm']['error']
    assert not (output/'rag.json').exists()
