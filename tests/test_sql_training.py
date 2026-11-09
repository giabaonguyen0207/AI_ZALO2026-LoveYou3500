import csv
import sqlite3
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from datamate.agent.tools import BusinessTools
from datamate.data.ingest import Ingestor
from datamate.data.split import DatasetSplit
from datamate.domain import Question
from datamate.domain import DataMateError
from datamate.data.sql import SQLExecutor
from datamate.models.registry import get_model
from datamate.training.sql_recipes import annotate, compile_proposal, trusted_inline
from datamate.training.sql_templates import compile_custom, customer_display
from datamate.training.sql_dataset import verify_sql_dataset
from test_pipeline_v3 import RULES


@pytest.fixture
def sql_tools(split_root, tmp_path):
    (split_root/'rulebook.md').write_text(RULES,encoding='utf-8')
    folder=split_root/'data/shop_a'
    (folder/'README.md').write_text('Dữ liệu đến hết 31/05/2026. Giá và tiền là số nguyên VND.',encoding='utf-8')
    with (folder/'sales.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['Mã đơn hàng','Ngày bán','Trạng thái','Tên hàng','Đơn vị tính','Số lượng','Số lượng quy đổi','Thành tiền','Đơn giá'])
        w.writerows([['o1','2026-05-01','Xong','Coca Cola 330ml','lon',2,2,200,100],
                     ['o2','2026-05-01','Đang giao','Trà','ly',3,3,300,100],
                     ['o3','2026-05-01','Hủy','Coca Cola 330ml','lon',1,1,100,100],
                     ['o4','2026-05-01','Trả hàng','Coca Cola 330ml','lon',1,1,100,100]])
    split=DatasetSplit(split_root)
    catalog=Ingestor(split,tmp_path/'catalog').catalog('shop_a')
    return BusinessTools(catalog,split.rulebook,'2026-05-31',max_rows=1000)


@pytest.mark.parametrize('text,expected',[
    ('Doanh thu Coca Cola 330ml trong tháng 5/2026 là bao nhiêu?',200),
    ('Tháng 5/2026 bán được bao nhiêu lon Coca Cola 330ml?',2),
    ('Tháng 5/2026 cửa hàng bán được bao nhiêu đơn?',2),
])
def test_sql_targets_keep_measure_product_and_sales_statuses(sql_tools,text,expected):
    q=Question('arbitrary_id','shop_a',text,'2026-05-31')
    recipe=annotate(q,sql_tools)
    actions,_=compile_proposal(recipe,q,sql_tools)
    query=actions[-1]['query']
    result=sql_tools.executor.execute(query['sql'],query['parameters'])
    assert result['rows']==[[expected]]
    renamed=Question('different_id','shop_a',text,'2026-05-31')
    assert annotate(renamed,sql_tools)==recipe


def test_product_percentage_is_not_a_percentage_question(sql_tools):
    q=Question('q','shop_a','Ngày 01/05/2026 bán Coca Cola 330ml giá bao nhiêu một lon?','2026-05-31')
    assert annotate(q,sql_tools)['task']=='historical_price'


def test_cte_rewrite_keeps_cross_join_and_parameter_order():
    sql='WITH current AS (SELECT ? AS value), previous AS (SELECT ? AS value) SELECT (current.value-previous.value)*100.0/previous.value+? FROM current,previous'
    rewritten=trusted_inline(sql)
    db=sqlite3.connect(':memory:')
    assert db.execute(sql,[120,100,3]).fetchall()==db.execute(rewritten,[120,100,3]).fetchall()==[(23.0,)]
    db.close()


@pytest.mark.parametrize('question,column',[('Doanh thu bán qua Zalo OA?', 'Bán qua'),
                                            ('Khách ở đường Trần Não có mấy người?', 'Địa chỉ')])
def test_nonempty_unmapped_information_is_unsupported_not_missing(question,column):
    sem={'policy':{},'capabilities':{'sales':True,'customer_identity':True,'inventory_snapshot':False,
        'balance_snapshot':False,'payments':False,'order_total':False,'channel':False},
        'bindings':[{'table':'raw','kind':'orders' if column=='Bán qua' else 'customers',
                     'roles':{'order_id':'id','customer_id':'customer_id'},'unmapped':[column],'ambiguous':{}}],
        'profiles':{'raw':{column:{'nonempty':4}}},'coverage':{'start':None,'end':None,'declared_end':None}}
    # check(None period) needs no SQL; use the actual check logic with inspected profiles.
    tools=object.__new__(BusinessTools);tools.semantic=sem;tools.as_of_date='2026-05-31'
    result=tools.check(question)
    assert result['availability']=='unsupported' and not result['missing']


def test_absent_customer_feedback_has_schema_evidence():
    sem={'capabilities':{},'bindings':[],'profiles':{},'coverage':{'start':None,'end':None,'declared_end':None}}
    tools=object.__new__(BusinessTools);tools.semantic=sem
    result=tools.check('Nhân viên nào được khách hay khen nhất?')
    assert result['availability']=='missing' and 'employee_feedback' in result['missing']


def test_stock_value_uses_last_observation_not_sum_of_balances(tmp_path):
    path=tmp_path/'stock.sqlite'
    db=sqlite3.connect(path)
    db.executescript("CREATE TABLE dm_stock_ledger (product_id TEXT, quantity REAL,occurred_at TEXT,_dm_row INTEGER); CREATE TABLE dm_products (product_id TEXT,cost REAL);")
    db.executemany('INSERT INTO dm_stock_ledger VALUES (?,?,?,?)',[('p1',10,'2026-05-01T01:00',1),('p1',7,'2026-05-02T01:00',2),('p1',99,'2026-06-01T01:00',3),('p2',3,'2026-05-01T01:00',4)])
    db.executemany('INSERT INTO dm_products VALUES (?,?)',[('p1',100),('p2',200)]);db.commit();db.close()
    tools=SimpleNamespace(semantic={'capabilities':{'inventory_snapshot':False,'stock_ledger':True}})
    recipe={'task':'inventory_value','question_key':'gia tri kho','periods':[]}
    q=Question('x','b','Giá trị kho hết hôm nay?','2026-05-31')
    actions,_=compile_custom(recipe,q,tools)
    query=actions[0]['query']
    assert SQLExecutor(path).execute(query['sql'],query['parameters'])['rows']==[[1300]]


def test_customer_phone_display_does_not_strip_identity_numbers():
    db=sqlite3.connect(':memory:')
    expr=customer_display('name')
    for original,expected in [('Đoàn Bích Khoa 0914717486','Đoàn Bích Khoa'),('Anh Nam - 0973 189 069','Anh Nam'),('Nguyễn Văn A 01','Nguyễn Văn A 01'),('Nguyễn Văn A - 2','Nguyễn Văn A - 2')]:
        assert db.execute('SELECT '+expr+' FROM (SELECT ? AS name)',[original]).fetchone()[0]==expected
    db.close()


def make_sealed_fixture(directory, *, no_sql=False, literal=False):
    directory.mkdir()
    target={'action':'plan','measure':'orders','operation':'count','type':'number','population':'sales','evidence_id':'Q','periods':[{'kind':'today'}],'filters':[]}
    if not no_sql:target['query']={'sql':'SELECT COUNT(*) FROM dm_orders','parameters':[]}
    if literal:target['answer']=5
    row={'question_id':'q','business_id':'train_b','messages':[{'role':'system','content':'SQL agent'},{'role':'user','content':json.dumps({'request':{'question':'Số đơn hôm nay?','as_of_date':'2026-05-31'}})},{'role':'assistant','content':json.dumps(target)}]}
    (directory/'train.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    (directory/'validation.jsonl').write_text('',encoding='utf-8')
    manifest={'kind':'verified_sql_sft','source_role':'competition_training','target_model':get_model('qwen35-9b').to_dict(),
        'train_businesses':['train_b'],'validation_businesses':['val_b'],'train_examples':1,'validation_examples':0,
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()},'runtime_source_hashes':{}}
    (directory/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    return manifest


def test_sql_release_rejects_tampering_and_partition_leak(tmp_path):
    folder=tmp_path/'release';manifest=make_sealed_fixture(folder)
    verify_sql_dataset(folder,verify_source=False)
    manifest['validation_businesses']=['train_b'];(folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(DataMateError,match='Overlapping'):verify_sql_dataset(folder,verify_source=False)
    manifest['validation_businesses']=['val_b'];(folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (folder/'train.jsonl').write_text('changed',encoding='utf-8')
    with pytest.raises(DataMateError,match='hash mismatch'):verify_sql_dataset(folder,verify_source=False)


@pytest.mark.parametrize('option,reason',[('no_sql','lacks SQL'),('literal','Literal final answer')])
def test_sql_release_rejects_missing_sql_or_copied_answers(tmp_path,option,reason):
    folder=tmp_path/'release';make_sealed_fixture(folder,**{option:True})
    with pytest.raises(DataMateError,match=reason):verify_sql_dataset(folder,verify_source=False)
