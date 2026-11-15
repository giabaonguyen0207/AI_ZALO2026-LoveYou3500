import json

import pytest

from datamate.application import run_batch
from datamate.config import Settings
from tools.verify_submission import verify_result


@pytest.fixture
def result(split_root, tmp_path):
    output = tmp_path / 'result'
    run_batch(Settings(backend='demo', split_root=str(split_root), result_dir=str(output),
                       cache_dir=str(tmp_path/'cache')))
    return split_root, output


def test_verifies_complete_output_and_logs_without_claiming_quality(result):
    split, output = result
    report = verify_result(split, output)
    assert report['questions'] == report['answers'] == report['logs'] == 3
    assert report['format_verified'] is True and report['quality_evaluated'] is False


@pytest.mark.parametrize('problem', ['duplicate_id', 'wrong_type', 'missing_thinking', 'multiline_log', 'aborted_run'])
def test_rejects_broken_submission_or_required_logs(result, problem):
    split, output = result
    submission = output / 'submission.jsonl'
    lines = submission.read_text(encoding='utf-8').splitlines()
    if problem == 'duplicate_id':
        submission.write_text('\n'.join(lines + [lines[0]])+'\n', encoding='utf-8')
    elif problem == 'wrong_type':
        row = json.loads(lines[0]); row.update(type='number', answer='42')
        submission.write_text('\n'.join([json.dumps(row), *lines[1:]])+'\n', encoding='utf-8')
    elif problem == 'aborted_run':
        summary = json.loads((output/'run_summary.json').read_text(encoding='utf-8'))
        summary['error'] = 'CUDA out of memory'
        (output/'run_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    else:
        path = next((output/'logs').glob('*.json'))
        record = json.loads(path.read_text(encoding='utf-8'))
        if problem == 'missing_thinking':
            record.pop('thinking')
        path.write_text(json.dumps(record, indent=2 if problem == 'multiline_log' else None)+'\n', encoding='utf-8')
    with pytest.raises(ValueError):
        verify_result(split, output)
