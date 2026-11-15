"""CPU fixtures for context/schema fixes; no model imports or reference test data."""
import csv
import json
import time
from types import SimpleNamespace

import pytest

from datamate.agent.context import fit_conversation
from datamate.agent.prompts import compact_sources, initial_messages
from datamate.agent.schema_repair import SchemaRepair
from datamate.config import Settings
from datamate.data.binding import bind_table, readme_descriptions, documented_file_matches
from datamate.data.ingest import Ingestor
from datamate.data.normalization import _note_status, _adjustment_patterns
from datamate.data.split import DatasetSplit
from datamate.data.sql import SQLExecutor
from datamate.domain import DataMateError, Question
from test_pipeline_v3 import RULES, README, business, plan


def test_repeated_schema_keeps_all_queryable_names_and_exact_fields():
    bindings = [{'table':f'raw_{i}', 'clean_table':f'clean_{i}', 'kind':'sales',
                 'roles':{'quantity':'Qty'}, 'ambiguous':{}, 'unmapped':['Note']} for i in range(17)]
    packed = compact_sources(bindings)
    assert len(packed)==17 and len(json.dumps(packed)) < len(json.dumps(bindings))
    assert packed[0]['roles']==bindings[0]['roles']
    assert all(item['same_schema_as']=='raw_0' for item in packed[1:])
    assert [item['table'] for item in packed]==[item['table'] for item in bindings]
    bindings[-1]['roles']={'quantity':'Different Qty'}
    assert compact_sources(bindings)[-1]['roles']=={'quantity':'Different Qty'}


def test_prompt_keeps_question_rulebook_and_readme_verbatim():
    question=Question('q','shop','SP-X 2/3/2025 giá 123?', '2025-03-03')
    messages=initial_messages(question, 'RULE ALL', 'README ALL', {'tables':[]})
    payload=json.loads(messages[1]['content'])
    assert payload['request']==question.prompt_dict()
    assert payload['rulebook']=='RULE ALL' and payload['business_readme']=='README ALL'


class Counter:
    def count_tokens(self, messages):
        return len(json.dumps(messages,ensure_ascii=False))


def test_context_checkpoint_preserves_values_ids_plan_and_latest_exchange():
    base=[{'role':'system','content':'ALL RULES'}, {'role':'user','content':'ORIGINAL QUESTION'}]
    messages=base + [{'role':role,'content':'x'*800} for _ in range(5)
                     for role in ('assistant','user')]
    result={'columns':['value'],'rows':[[42]],'verified':True}
    packed,event=fit_conversation(messages,Counter(),SimpleNamespace(max_context_tokens=3500,max_new_tokens=768),
        plan={'type':'number'},results={'r1':result},evidence={'e1':{'availability':'missing','missing':['x']}},
        compact_result=lambda value:value)
    assert packed[:2]==base and packed[-2:]==messages[-2:]
    state=json.loads(packed[3]['content'])
    assert state['query_results']['r1']==result and state['current_plan']=={'type':'number'}
    assert state['missing_data_evidence']['e1']['missing']==['x']
    assert event['preserved_query_ids']==['r1']


def test_oversized_baseline_is_rejected_without_truncating_or_generating():
    messages=[{'role':'system','content':'rule'*1000},{'role':'user','content':'question'}]
    with pytest.raises(DataMateError,match='Context'):
        fit_conversation(messages,Counter(),SimpleNamespace(max_context_tokens=2048,max_new_tokens=768),
                         plan=None,results={},evidence={},compact_result=lambda value:value)
    assert messages[0]['content']=='rule'*1000


def test_history_checkpoint_retains_a_repaired_catalog_from_earlier_turns():
    messages=[{'role':'system','content':'rules'},{'role':'user','content':'question'},
              {'role':'assistant','content':'{}'},
              {'role':'user','content':json.dumps({'tool_result':{'catalog_updates':{'new_role':'actual_column'}}})}]
    messages += [{'role':role,'content':'x'*800} for _ in range(5) for role in ('assistant','user')]
    packed,event=fit_conversation(messages,Counter(),SimpleNamespace(max_context_tokens=3500,max_new_tokens=768),
                                 plan=None,results={},evidence={},compact_result=lambda value:value)
    assert json.loads(packed[3]['content'])['catalog_updates']=={'new_role':'actual_column'}
    assert event['phase']=='context_compaction'


def test_file_template_scope_and_documented_order_counter():
    table={'name':'raw_book','file':'Ledger_04_2025.xlsx','sheet':'Sales','columns':['STT','Qty','Money','Item']}
    readme='''## Cột trong `Ledger_<month>_<year>.xlsx`, sheet `Sales`
- **STT**: số thứ tự đơn (hoá đơn) trong sổ
- **Qty**: số lượng
- **Money**: thành tiền của dòng
- **Item**: tên hàng
## Cột trong `Unrelated.xlsx`
- **STT**: số thứ tự dòng
'''
    bound=bind_table(table,readme)
    assert bound['roles']['order_id']=='STT' and 'line_no' not in bound['roles']
    assert not readme_descriptions(readme,{**table,'file':'UnrelatedLedger_04_2025.xlsx'})
    assert bind_table(table,'')['roles']['line_no']=='STT'  # Bare STT stays a row counter.


def test_documented_filename_only_expands_angle_placeholders():
    assert documented_file_matches('Sales [final]_<year>.xlsx', 'Sales [final]_2025.xlsx')
    assert not documented_file_matches('Sales [final]_<year>.xlsx', 'Sales f_2025.xlsx')
    assert not documented_file_matches('Sales*.csv', 'Sales123.csv')
    assert not documented_file_matches('Sales_<year>.csv', 'Sales_.csv')


def test_non_item_evidence_does_not_classify_an_unrelated_product():
    readme = ('Các dòng không phải hàng: "Voucher".\n'
              'Sản phẩm "Thẻ điểm" được bán như hàng thông thường.')
    assert _adjustment_patterns(readme) == ['voucher']


def test_successful_repair_is_visible_even_when_plan_retry_fails(business,tmp_path,scripted_backend,monkeypatch):
    from copy import deepcopy
    from datamate.agent.runner import Agent
    import datamate.agent.runner as runner
    split,tools=business
    backend=scripted_backend([{'action':'plan','measure':'revenue','population':'sales'},
                              {'action':'final','type':'abstain'}])
    backend.counts_as_model=True
    settings=Settings(backend='transformers',cache_dir=str(tmp_path/'catalog-error'),max_steps=2)
    agent=Agent(settings,backend,split)
    original=agent.ingestor.catalog('shop_a')
    changed=deepcopy(original)
    changed.fingerprint='repaired-catalog'
    changed.semantic['quality']['review_marker']='repaired'
    monkeypatch.setattr(runner,'repair_needs',lambda *args:{'roles':['status']})
    def invalid(*args,**kwargs):
        raise DataMateError('Plan still invalid after repair')
    monkeypatch.setattr(runner,'validate_plan',invalid)
    monkeypatch.setattr(agent.schema_repair,'repair',lambda *args:(changed,[]))
    agent.run(Question('q','shop_a','Doanh thu tháng này?', '2025-01-31'))
    feedback=json.loads(backend.messages[1][-1]['content'])['tool_result']
    assert feedback['error']=='Plan still invalid after repair'
    assert feedback['catalog_updates']['quality']['review_marker']=='repaired'


@pytest.mark.parametrize('value,expected',[(None,'completed'),('Gọi trước khi giao','completed'),
    ('Giao cho anh Huy','completed'),
    ('Khách trả lại','returned'),('Đơn huỷ','cancelled'),('Đang giao','shipping'),
    ('Không hủy đơn','unknown'),('Đã hủy, đang giao','unknown')])
def test_documented_note_status_and_negative_phrases(value,expected):
    assert _note_status(value,'Ghi chú không nhắc trạng thái là đơn hoàn thành.')==expected
    assert _note_status(None,'') is None


def test_notes_order_counter_and_non_item_adjustments_end_to_end(split_root,tmp_path):
    folder=split_root/'data/shop_a'
    (folder/'sales.csv').unlink()
    with (folder/'Ledger_2025.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['STT','Ngày','Hàng','SL','TT','Ghi chú'])
        writer.writerows([[1,'01/01','Hàng A',1,100,None],
                         [1,None,'Voucher',None,-20,None],
                         [1,None,'Dùng 3 điểm',None,-3,None],
                         [1,None,'Phí ship',None,10,None],
                         [2,'02/01','Hàng B',1,900,'Khách trả lại']])
    readme='''## Cột trong `Ledger_<year>.csv`
- **STT**: số thứ tự đơn trong sổ
- **Ghi chú**: ghi chú của đơn; đơn khách trả lại ghi ở đây
Các dòng không phải hàng: "Voucher", "Dùng <n> điểm", "Phí ship".
'''
    (folder/'README.md').write_text(readme,encoding='utf-8')
    (split_root/'rulebook.md').write_text(RULES+'\nGhi chú không nhắc trạng thái là đơn hoàn thành.',encoding='utf-8')
    catalog=Ingestor(DatasetSplit(split_root),tmp_path/'new-cache').catalog('shop_a')
    sql=SQLExecutor(catalog.database)
    assert sql.execute('SELECT SUM(line_amount),COUNT(*) FROM dm_sales')['rows']==[[100,1]]
    assert sql.execute('SELECT COUNT(*) FROM dm_orders')['rows']==[[1]]
    assert sql.execute('SELECT COUNT(*) FROM dm_sales_all')['rows']==[[2]]
    assert sql.execute('SELECT order_date FROM dm_sales')['rows']==[['2025-01-01']]
    assert sql.execute('SELECT COUNT(*) FROM clean_0_ledger_2025')['rows']==[[5]]


def test_note_mapping_requires_documentation_and_real_status_column_wins():
    table={'name':'r','file':'a.csv','columns':['Ngày','Hàng','SL','TT','Ghi chú']}
    assert 'status' not in bind_table(table,'Ghi chú là thông tin giao hàng')['roles']
    documented='Ghi chú của đơn ghi khách trả lại; đơn khách trả lại ghi ở đây.'
    assert bind_table(table,documented)['status_from_notes']['column']=='Ghi chú'
    explicit=bind_table({**table,'columns':table['columns']+['Trạng thái']},documented)
    assert explicit['roles']['status']=='Trạng thái' and 'status_from_notes' not in explicit
    other_file='## Cột trong `different.csv`\n- **Ghi chú**: trạng thái đơn huỷ hoặc trả hàng'
    assert 'status_from_notes' not in bind_table(table,other_file)


def test_note_default_never_fills_an_undocumented_or_missing_status(split_root,tmp_path):
    folder=split_root/'data/shop_a'
    (folder/'sales.csv').write_text('Đơn,Ngày,Hàng,SL,TT\nx,01/01/2025,A,1,100\n',encoding='utf-8')
    (folder/'README.md').write_text('Không có giải thích trạng thái.',encoding='utf-8')
    (split_root/'rulebook.md').write_text(RULES+'\nGhi chú không nhắc trạng thái là đơn hoàn thành.',encoding='utf-8')
    catalog=Ingestor(DatasetSplit(split_root),tmp_path/'missing-status').catalog('shop_a')
    sql=SQLExecutor(catalog.database)
    assert sql.execute('SELECT COUNT(*) FROM dm_sales')['rows']==[[0]]
    assert sql.execute('SELECT DISTINCT status FROM dm_sales_all')['rows']==[['unknown']]


def test_validated_status_patch_takes_precedence_over_note_default(split_root,tmp_path):
    from copy import deepcopy
    from datamate.data.normalization import prepare_catalog
    folder=split_root/'data/shop_a'
    (folder/'sales.csv').write_text('Đơn,Ngày,Hàng,SL,TT,Ghi chú\nx,01/01/2025,A,1,100,Đợi\n',encoding='utf-8')
    (folder/'README.md').write_text('Ghi chú thể hiện trạng thái đơn. Đợi nghĩa là đang giao.',encoding='utf-8')
    (split_root/'rulebook.md').write_text(RULES+'\nGhi chú không nhắc trạng thái là đơn hoàn thành.',encoding='utf-8')
    split=DatasetSplit(split_root)
    ingestor=Ingestor(split,tmp_path/'note-patch')
    catalog=ingestor.catalog('shop_a')
    patched=deepcopy(catalog.semantic['bindings'])
    sales=next(b for b in patched if b.get('status_from_notes'))
    sales['status_map']={'Đợi':'shipping'}  # Represents a patch already validated by SchemaRepair.
    ready=prepare_catalog(ingestor.catalog('shop_a',normalize=False),split.rulebook,split.readme('shop_a'),
                          tmp_path/'note-patch',{'mode':'code_first','policy':catalog.semantic['policy'],
                          'bindings':patched,'cache_key':'fixture-validated-patch'})
    assert SQLExecutor(ready.database).execute('SELECT status FROM dm_sales_all')['rows']==[['shipping']]
    assert ready.semantic['status_profiles'][sales['table']]['values'][0]['canonical']=='shipping'


def test_parse_failure_can_retry_and_does_not_poison_disk_cache(business,tmp_path,scripted_backend):
    split,tools=business
    catalog=tools.catalog
    # A policy-only need is enough to exercise parser recovery without a GPU.
    import datamate.agent.schema_repair as module
    from unittest.mock import patch
    settings=Settings(cache_dir=str(tmp_path/'repair'),backend='transformers')
    backend=scripted_backend([{'action':'sql','sql':'SELECT 1'}, {'action':'prepare'}])
    repair=SchemaRepair(settings,split)
    with patch.object(module,'repair_needs',return_value={'policy':['week_start']}):
        ready,trace=repair.repair(catalog,catalog,README,{},backend,time.monotonic()+20)
    assert len(backend.messages)==2 and sum('raw' in item for item in trace)==2
    assert ready.fingerprint==catalog.fingerprint
    assert trace[0]['observation']['retry'] is True
    assert all('error' not in json.loads(path.read_text(encoding='utf-8'))['payload']
               for path in repair.cache.directory.glob('*.json'))


def test_exhausted_parse_retry_is_only_cached_for_the_worker(business,tmp_path):
    split,tools=business
    import datamate.agent.schema_repair as module
    from unittest.mock import patch
    class RawBackend:
        def __init__(self,responses): self.responses=iter(responses); self.calls=0
        def generate(self,*args,**kwargs): self.calls+=1; return next(self.responses)
    settings=Settings(cache_dir=str(tmp_path/'transient-repair'),backend='transformers')
    repair=SchemaRepair(settings,split)
    broken=RawBackend(['{"action":"prepare"','not json'])
    with patch.object(module,'repair_needs',return_value={'policy':['week_start']}):
        repair.repair(tools.catalog,tools.catalog,README,{},broken,time.monotonic()+20)
        repair.repair(tools.catalog,tools.catalog,README,{},broken,time.monotonic()+20)
        assert broken.calls==2
        assert not list(repair.cache.directory.glob('*.json'))
        recovered=RawBackend(['{"action":"prepare"}'])
        SchemaRepair(settings,split).repair(tools.catalog,tools.catalog,README,{},recovered,time.monotonic()+20)
        assert recovered.calls==1
