import ast
import json
from pathlib import Path
import sys

import pytest

from tools.build_sql_colab import build


def notebook_logger(tmp_path):
    notebook = tmp_path / 'training.ipynb'
    build(notebook, 'a' * 64, 'source.zip', 'dataset.zip')
    cells = json.loads(notebook.read_text(encoding='utf-8'))['cells']
    functions = [node for cell in cells if cell['cell_type'] == 'code'
                 for node in ast.parse(''.join(cell['source'])).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'run_logged']
    assert len(functions) == 1, 'Notebook must forward child-process output to VS Code and a log'
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<notebook logger>', 'exec'), namespace)
    return namespace['run_logged']


def test_training_child_error_is_visible_and_saved(tmp_path, capsys):
    run = notebook_logger(tmp_path)
    log = tmp_path / 'failure.log'
    with pytest.raises(RuntimeError, match='original GPU failure'):
        run([sys.executable, '-u', '-c',
             'import sys; print("before failure", flush=True); '
             'print("original GPU failure", file=sys.stderr, flush=True); sys.exit(7)'],
            cwd=tmp_path, log_path=log)
    visible = capsys.readouterr().out
    assert 'before failure' in visible and 'original GPU failure' in visible
    assert 'original GPU failure' in log.read_text(encoding='utf-8')


def test_training_success_streams_output_and_does_not_overwrite_log(tmp_path, capsys):
    run = notebook_logger(tmp_path)
    log = tmp_path / 'success.log'
    command = [sys.executable, '-u', '-c', 'print("optimizer step completed")']
    assert run(command, cwd=tmp_path, log_path=log) == 0
    assert 'optimizer step completed' in capsys.readouterr().out
    original = log.read_bytes()
    with pytest.raises(FileExistsError):
        run(command, cwd=tmp_path, log_path=log)
    assert log.read_bytes() == original


def test_smoke_cell_preserves_failed_attempt_and_reuses_completed_retry(tmp_path):
    run = notebook_logger(tmp_path)
    cells = json.loads((tmp_path/'training.ipynb').read_text(encoding='utf-8'))['cells']
    smoke_cell = next(''.join(c['source']) for c in cells if c['cell_type']=='code'
                      and ''.join(c['source']).startswith('profile=json.loads'))
    work = tmp_path/'work'
    (work/'configs').mkdir(parents=True)
    (work/'configs/sql-qwen35-9b-qlora.json').write_text('{"method":"qlora"}', encoding='utf-8')
    output = tmp_path/'run'
    canonical = output/'smoke_adapter'
    canonical.mkdir(parents=True)
    marker = canonical/'run_identity.json'
    marker.write_text('failed attempt, preserve me', encoding='utf-8')
    # Stand-in for the GPU-only CLI: real process, real files and nonempty-output guard.
    script = ('import json,sys; from pathlib import Path; '
              'c=json.loads(Path(sys.argv[1]).read_text()); p=Path(c["output_dir"]); '
              'assert not(p.exists() and any(p.iterdir())), "partial output exists"; '
              'p.mkdir(parents=True,exist_ok=True); '
              '(p/"training_report.json").write_text(json.dumps({"training_config":c}))')
    def training(command, *, cwd, log_path):
        return run([sys.executable, '-u', '-c', script, command[-1]], cwd=cwd, log_path=log_path)
    namespace = {'json':json, 'sys':sys, 'WORK':work, 'METHOD':'qlora', 'BASE_DIR':tmp_path/'base',
                 'DATASET':tmp_path/'data', 'SEED':42, 'OUT':output, 'manifest':{'max_length':16384},
                 'STAGE':'pilot', 'run_logged':training}
    exec(compile(smoke_cell, '<actual smoke cell>', 'exec'), namespace)
    assert marker.read_text(encoding='utf-8') == 'failed attempt, preserve me'
    reports = list(output.glob('smoke_adapter*/training_report.json'))
    assert len(reports) == 1 and reports[0].parent != canonical
    exec(compile(smoke_cell, '<actual smoke cell rerun>', 'exec'), namespace)
    assert list(output.glob('smoke_adapter*/training_report.json')) == reports
    assert len(list((output/'logs').glob('*.log'))) == 1
