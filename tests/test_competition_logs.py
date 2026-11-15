import json

from datamate.application import run_batch
from datamate.config import Settings


def test_batch_writes_one_line_logs_with_organizer_required_fields(split_root, tmp_path):
    output = tmp_path/'result'
    run_batch(Settings(backend='demo', split_root=str(split_root),
                       result_dir=str(output), cache_dir=str(tmp_path/'cache')))
    files = list((output/'logs').glob('*.json'))
    assert len(files) == 3
    assert len(list((output/'logs').iterdir())) == 3
    assert len(list((output/'debug').glob('*.md'))) == 3
    for path in files:
        text = path.read_text(encoding='utf-8')
        record = json.loads(text)
        assert {'id', 'answer', 'elapsed_ms', 'thinking'} <= record.keys()
        assert type(record['elapsed_ms']) is int and record['elapsed_ms'] >= 0
        assert record['thinking'] is None  # Demo does not generate reasoning.
        assert record['answer']['type'] == record['submission']['type']
        assert len(text.splitlines()) == 1


def test_thinking_and_executed_sql_are_preserved_without_fabrication():
    from datamate.competition_log import competition_fields
    from datamate.domain import Answer, RunResult
    thought = '  Dòng suy luận thật\n' + 'x' * 12000
    sql = 'SELECT SUM(amount) FROM payments WHERE date<=?'
    result = RunResult(Answer('number', 42), 1.2346, model_calls=3, trace=[
        {'step': 1, 'raw': '<think>'+thought+'</think>{"action":"sql"}'},
        {'step': 2, 'raw': '{"action":"final","reason":"not thinking"}',
         'execution': {'sql': sql, 'parameters': ['2026-11-14'],
                       'columns': ['total'], 'rows': [[42]], 'truncated': False}},
        {'step': 3, 'raw': '<think> unfinished reasoning'},
    ])
    record = competition_fields(result)
    assert record['thinking'] == [thought, '', ' unfinished reasoning']
    assert record['code'] == sql
    assert record['inputs'][0]['parameters'] == ['2026-11-14']
    assert record['output'][0]['rows'] == [[42]]
    assert record['elapsed_ms'] == 1235
    assert record['answer'] == {'type': 'number', 'answer': 42}


def test_action_reason_is_not_reported_as_model_thinking():
    from datamate.competition_log import competition_fields
    from datamate.domain import Answer, RunResult
    record = competition_fields(RunResult(Answer('abstain'), 0, trace=[
        {'raw': '{"action":"final","reason":"cannot answer"}'}]))
    assert record['thinking'] is record['code'] is record['output'] is None
