import json

from tools.compare_runtime_benchmarks import compare
from test_kaggle_runtime_benchmark import comparison_fixture, save


CRITICAL = ['agent/planning.py', 'agent/protocol.py', 'agent/tools.py', 'agent/verification.py', 'data/rules.py', 'evaluation/scoring.py']


def paired(tmp_path, candidate='vllm'):
    left = comparison_fixture(tmp_path / 'baseline')
    right = comparison_fixture(tmp_path / 'candidate', candidate)
    for folder in (left, right):
        path = folder / 'benchmark_identity.json'
        identity = json.loads(path.read_text())
        identity['source_hashes'] = {'datamate/' + name: 'f' * 64 for name in CRITICAL}
        identity['inventory'] = {'models': [{'key': 'fixture', 'parameters': 3}], 'total_parameters': 3}
        identity['artifact_manifests'] = {'fixture': 'same'}
        save(path, identity)
        (folder / 'question_timing.jsonl').write_text(json.dumps({'id': 'q1', 'baseline_messages_sha256': 'base-prompt'}) + '\n')
    return left, right


def add_rag(folder):
    path = folder / 'benchmark_identity.json'
    identity = json.loads(path.read_text())
    identity['settings']['rag'] = {'mode': 'dense', 'embedding_key': 'vietnamese-embedding'}
    identity['artifact_manifests']['vietnamese-embedding'] = 'e' * 64
    identity['inventory']['models'].append({'key': 'vietnamese-embedding', 'parameters': 567754752})
    identity['inventory']['total_parameters'] += 567754752
    save(path, identity)


def test_backend_comparison_rejects_unexpected_rag_change(tmp_path):
    left, right = paired(tmp_path)
    add_rag(right)
    result = compare(left, right)
    assert not result['paired_comparison_valid'] and result['inference_speedup'] is None
    assert result['gates']['same_retrieval_configuration'] is False


def test_rag_comparison_allows_context_change_but_keeps_baseline_prompt(tmp_path):
    left, right = paired(tmp_path, 'transformers')
    add_rag(right)
    (right / 'model_calls.jsonl').write_text(json.dumps({'id': 'q1', 'start_seconds': 1, 'messages_sha256': 'with-context'}) + '\n')
    result = compare(left, right, comparison_mode='rag')
    assert result['paired_comparison_valid'] and result['comparison_kind'] == 'rag_augmentation'
    assert result['gates']['same_baseline_prompt']
    assert result['engine_only_speedup'] is None


def test_comparison_rejects_changed_business_logic_even_with_equal_first_prompt(tmp_path):
    left, right = paired(tmp_path)
    path = right / 'benchmark_identity.json'
    identity = json.loads(path.read_text())
    identity['source_hashes']['datamate/agent/planning.py'] = 'changed'
    save(path, identity)
    assert not compare(left, right)['paired_comparison_valid']


def test_rag_comparison_rejects_missing_embedding_artifact_evidence(tmp_path):
    left, right = paired(tmp_path, 'transformers')
    add_rag(right)
    path = right / 'benchmark_identity.json'
    identity = json.loads(path.read_text())
    del identity['artifact_manifests']['vietnamese-embedding']
    save(path, identity)
    assert not compare(left, right, comparison_mode='rag')['paired_comparison_valid']


def test_backend_comparison_rejects_runner_or_global_budget_changes(tmp_path):
    left,right = paired(tmp_path)
    path = right/'benchmark_identity.json'
    original = json.loads(path.read_text())
    for field in ('source','budget','sampling'):
        identity = json.loads(json.dumps(original))
        if field=='source': identity['source_hashes']['datamate/agent/runner.py']='changed'
        if field=='budget': identity['settings']['wall_time_seconds']=100
        if field=='sampling': identity['settings']['vllm']={'extra_sampling_params':{'temperature':1}}
        save(path,identity)
        result = compare(left,right)
        assert not result['paired_comparison_valid'] and result['inference_speedup'] is None,field
