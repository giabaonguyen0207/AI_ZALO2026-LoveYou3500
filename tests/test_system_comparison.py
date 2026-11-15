import json
from pathlib import Path
import pytest

from datamate.config import Settings


def test_three_arms_freeze_ids_source_precision_and_generator_budgets(tmp_path, split_root, monkeypatch):
    from tools import compare_system as tool
    calls = []
    def process(command, **options):
        config = json.loads(Path(command[command.index('--config') + 1]).read_text())
        calls.append((command, options, config))
    monkeypatch.setattr(tool, 'run_checked_process', process)
    monkeypatch.setattr(tool, 'compare', lambda *args, **kwargs: {'comparison_kind': kwargs['comparison_mode']})
    settings = Settings(split_root=str(split_root), model_dir='sealed/generator')
    output = tmp_path / 'comparison'
    tool.run(settings, ['q1','q2'], output, project_root=Path(__file__).resolve().parents[1])
    assert len(calls) == 3
    assert [(c[2]['backend'],c[2]['rag']['mode']) for c in calls] == [
        ('transformers','none'),('vllm','none'),('transformers','dense_rerank')]
    assert all(c[2]['dtype']=='float16' and c[2]['question_workers']==1 for c in calls)
    common = ['model_key','model_dir','max_steps','max_new_tokens','max_context_tokens','question_timeout_seconds','mode','seed']
    assert all(len({json.dumps(c[2][key]) for c in calls})==1 for key in common)
    assert len({c[0][c[0].index('--project-root')+1] for c in calls})==1
    assert len({c[0][c[0].index('--ids')+1] for c in calls})==1
    assert len({c[2]['cache_dir'] for c in calls})==3
    assert all(c[1]['startup_progress'] == Path(c[2]['result_dir'])/'startup_progress.json' for c in calls)
    assert (output/'comparisons.json').is_file()


def test_comparison_presets_have_equal_generator_settings():
    root = Path(__file__).resolve().parents[1]
    settings = [Settings.load(root/'configs/comparison'/f'{arm}.json',use_env=False) for arm in ('baseline','vllm','rag')]
    assert all(s.dtype=='float16' and s.question_workers==1 for s in settings)
    assert settings[1].vllm.init_timeout_seconds >= 1200
    for key in ('model_key','model_dir','mode','max_steps','max_new_tokens','max_context_tokens','question_timeout_seconds','seed'):
        assert len({getattr(s,key) for s in settings})==1


def test_three_arm_runner_rejects_non_greedy_configuration_before_running(tmp_path):
    from tools.compare_system import run
    settings = Settings(vllm={'extra_sampling_params':{'temperature':1}}).validate()
    with pytest.raises(ValueError,match='greedy'):
        run(settings,['q1'],tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_notebook_install_resolves_vllm_and_rag_from_config(tmp_path, monkeypatch):
    import runpy
    import sys
    from types import SimpleNamespace
    from tools.create_inference_notebook import INSTALL
    root = Path(__file__).resolve().parents[1]
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'backend':'vllm','rag':{'mode':'dense'}}))
    calls = []
    namespace = {'WORK':root,'CONFIG_FILE':config,'sys':sys,
                 'subprocess':SimpleNamespace(run=lambda command,**kwargs:calls.append(command))}
    # Only test dependency selection/restart checks; no GPU or pip runs.
    namespace['CHECK_CUDA'] = False
    monkeypatch.delenv('DATAMATE_BACKEND',raising=False)
    monkeypatch.delenv('DATAMATE_RAG_MODE',raising=False)
    exec(compile(INSTALL,'notebook-install','exec'),namespace)
    assert any('.[vllm,rag]' in command for command in calls)


def test_notebook_explicit_rag_choice_overrides_config(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from tools.create_inference_notebook import INSTALL
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'backend':'vllm','rag':{'mode':'none'}}))
    calls = []
    namespace = {'WORK':Path(__file__).resolve().parents[1],'CONFIG_FILE':config,'sys':sys,
                 'subprocess':SimpleNamespace(run=lambda command,**kwargs:calls.append(command)),
                 'RAG_MODE':'dense_rerank','BACKEND':'vllm','CHECK_CUDA':False}
    exec(compile(INSTALL,'notebook-install','exec'),namespace)
    assert namespace['selected'].rag.mode=='dense_rerank'
    assert any('.[vllm,rag]' in command for command in calls)
