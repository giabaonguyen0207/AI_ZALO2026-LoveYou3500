from datamate.agent.tools import BusinessTools
from datamate.data.ingest import Ingestor
from datamate.data.split import DatasetSplit
from datamate.domain import Answer, RunResult
from datamate.competition_log import competition_fields
from datamate.execution_audit import record_executions


def test_lookup_logs_actual_function_inputs_and_internal_sql_outputs(split_root,tmp_path):
    split=DatasetSplit(split_root)
    @record_executions
    def actual_run():
        tools=BusinessTools(Ingestor(split,tmp_path/'cache').catalog('shop_a'),split.rulebook,'2026-06-02')
        tools.lookup('product','Không có tên hàng',3)
        return RunResult(Answer('abstain'),0)
    result=actual_run()
    lookup=next(e for e in result.executions if e['source'].endswith('BusinessTools.lookup'))
    assert lookup['inputs']['text']=='Không có tên hàng' and lookup['inputs']['limit']==3
    assert lookup['output']['query']=='Không có tên hàng'
    sql=[e for e in result.executions if e['source'].endswith('SQLExecutor.execute')]
    assert len(sql)==2 and all('rows' in e['output'] for e in sql)
    record=competition_fields(result)
    assert 'BusinessTools.lookup' in record['code'] and 'SELECT' in record['code']
    assert record['inputs'] and record['output']


def test_execution_receipts_do_not_leak_between_concurrent_questions(tmp_path):
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from datamate.data.sql import SQLExecutor
    path=tmp_path/'empty.sqlite'
    sqlite3.connect(path).close()
    barrier=Barrier(2)
    @record_executions
    def run(value):
        barrier.wait(timeout=5)
        SQLExecutor(path).execute('SELECT ?',[value])
        return RunResult(Answer('number',value),0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,[11,22]))
    assert [r.executions[0]['inputs']['parameters'] for r in results]==[[11],[22]]
    assert [r.executions[0]['output']['rows'] for r in results]==[[[11]],[[22]]]
    assert all(len(r.executions)==1 for r in results)
