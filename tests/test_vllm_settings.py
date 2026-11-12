import json

import pytest

from datamate.cli import load_settings, parser
from datamate.config import Settings, VLLMSettings
from datamate.domain import DataMateError, Question
from datamate.models.runtime import runtime_identity


def test_cli_env_json_precedence_and_optional_paths(tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'backend': 'demo', 'model_dir': 'file model',
        'vllm': {'max_num_seqs': 2, 'dtype': 'float16'}}))
    monkeypatch.setenv('DATAMATE_CONFIG', str(config))
    monkeypatch.setenv('DATAMATE_BACKEND', 'vllm')
    monkeypatch.setenv('DATAMATE_MODEL_DIR', 'env model with spaces')
    monkeypatch.setenv('DATAMATE_VLLM', '{"gpu_memory_utilization":0.65,"max_num_seqs":3}')
    monkeypatch.setenv('DATAMATE_VLLM_MAX_NUM_SEQS', '4')
    args = parser().parse_args(['run', '--model-dir', 'cli model', '--concurrency', '6',
        '--cache-dir', 'custom cache', '--max-context-tokens', '32768',
        '--vllm-max-num-seqs', '7', '--no-vllm-enable-prefix-caching',
        '--vllm-extra-engine-args', '{"disable_log_stats":true}'])
    settings = load_settings(args)
    assert settings.backend == 'vllm' and settings.model_dir == 'cli model'
    assert settings.effective_workers == 6 and settings.cache_dir == 'custom cache'
    assert settings.max_context_tokens == 32768
    assert settings.vllm.max_num_seqs == 7 and settings.vllm.dtype == 'float16'
    assert settings.vllm.gpu_memory_utilization == 0.65 and settings.vllm.enable_prefix_caching is False
    assert settings.vllm.extra_engine_args == {'disable_log_stats': True}
    assert Settings.load(config, use_env=False).backend == 'demo'


def test_old_presets_and_nested_roundtrip():
    settings = Settings.load('configs/qwen35-9b.json', use_env=False)
    assert settings.effective_workers == 1
    assert Settings(**settings.to_dict()).validate().to_dict() == settings.to_dict()
    assert Settings(backend='vllm').validate().effective_workers == 4
    assert Settings(backend='vllm', vllm={'max_num_seqs': 2}).validate().effective_workers == 2


@pytest.mark.parametrize('options', [
    {'question_workers': True}, {'question_workers': 0},
    {'vllm': {'unknown': 1}}, {'vllm': {'gpu_memory_utilization': 1.1}},
    {'vllm': {'max_num_seqs': 0}}, {'vllm': {'enable_prefix_caching': 'false'}},
    {'vllm': {'abort_timeout_seconds': float('nan')}},
    {'vllm': {'extra_engine_args': {'model': 'unchecked'}}},
    {'vllm': {'extra_engine_args': {'speculative_config': {'model': 'undeclared'}}}},
    {'vllm': {'extra_sampling_params': {'max_tokens': 99999}}},
    {'mode': 'speculative'}, {'quantization': '4bit'},
])
def test_bad_options_fail_before_importing_engine(options):
    with pytest.raises(DataMateError):
        Settings(backend='vllm', **options).validate()


def test_invalid_environment_is_actionable(monkeypatch):
    monkeypatch.setenv('DATAMATE_VLLM_ENABLE_PREFIX_CACHING', 'yes')
    with pytest.raises(DataMateError, match='DATAMATE_VLLM_ENABLE_PREFIX_CACHING'):
        Settings.load()


def test_runtime_identity_tracks_engine_and_prompt_and_scheduler_options():
    base = runtime_identity(Settings(backend='vllm'))
    assert base != runtime_identity(Settings(backend='vllm', question_workers=8))
    assert base != runtime_identity(Settings(backend='vllm', vllm=VLLMSettings(enable_prefix_caching=False)))


def test_prefix_layout_preserves_context_but_places_business_before_question():
    from datamate.agent.prompts import initial_messages
    catalog = {'tables': []}
    first = initial_messages(Question('a', 'shop', 'câu thứ nhất', '2026-01-01'), 'luật', 'readme', catalog,
                             {'periods': []}, cache_prefix=True)
    second = initial_messages(Question('b', 'shop', 'câu thứ hai', '2026-02-01'), 'luật', 'readme', catalog,
                              {'periods': []}, cache_prefix=True)
    assert first[0] == second[0]
    assert first[1]['content'].splitlines()[0] == second[1]['content'].splitlines()[0]
    common, request = [json.loads(line) for line in first[1]['content'].splitlines()]
    assert common['rulebook'] == 'luật' and common['business_readme'] == 'readme'
    assert request['request']['question'] == 'câu thứ nhất'
    assert request['request']['as_of_date'] == '2026-01-01'
    assert first != second
