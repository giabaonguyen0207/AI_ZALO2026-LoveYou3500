import csv
import json
import shutil
from pathlib import Path

import pytest
from openpyxl import Workbook
from openpyxl.utils.datetime import CALENDAR_MAC_1904

from datamate.agent.protocol import parse_action, answer_from_query, verify_legacy_answer
from datamate.agent.runner import Agent
from datamate.agent.tools import BusinessTools
from datamate.agent.verification import question_periods, verify_metric_request
from datamate.config import Settings
from datamate.data.ingest import Ingestor
from datamate.data.rules import read_policy, resolve_period
from datamate.data.split import DatasetSplit
from datamate.data.transforms import parse_number
from datamate.domain import Answer, DataMateError, Question

RULES = '''## 1. Đơn hàng và doanh thu
Đơn không huỷ và không trả hàng. Đơn có trả hàng kể cả một phần: bỏ cả đơn. Đơn đang giao vẫn tính.
Doanh thu là tổng thành tiền các dòng hàng, không trừ giảm giá cả đơn.
## 2. Thời gian
Tuần bắt đầu thứ Hai. Ca sáng 06:00–13:59, ca chiều 14:00–21:59.
## 3. Xếp hạng
Bán chạy theo số lượng quy về đơn vị cơ bản. Phí phục vụ không phải mặt hàng.
## 4. Công nợ
Đơn không huỷ. Đơn trả hàng vẫn tính. COD không phải khách nợ.
## 8. Cách ghi
Ô để trống ở các dòng sau giống ô đó ở dòng đầu nhóm.
Hai dòng của cùng một đơn giống nhau ở mọi ô: tính một lần.
## 9. Đáp án
Phần trăm làm tròn 2 chữ số thập phân.
'''

README = '''## Cột trong `ledger.csv`
- **Ref**: mã đơn
- **Placed**: ngày bán
- **State**: trạng thái đơn
- **Item**: tên hàng
- **Qty**: số lượng
- **Cash**: thành tiền sau giảm
- **Unit**: đơn vị tính
- **Client**: tên khách
## Quy ước
Dữ liệu có đến hết 31/01/2025.
- Bánh: 1 hộp = 6 cái
'''
HEADERS = ['Ref','Placed','State','Item','Qty','Cash','Unit','Client']


def setup_ledger(split_root, rows, readme=README):
    folder=split_root/'data/shop_a'
    (folder/'sales.csv').unlink()
    with (folder/'ledger.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(HEADERS);writer.writerows(rows)
    (folder/'README.md').write_text(readme,encoding='utf-8')
    (split_root/'rulebook.md').write_text(RULES,encoding='utf-8')
    return folder


def tools_for(root,tmp_path):
    split=DatasetSplit(root);catalog=Ingestor(split,tmp_path/'cache').catalog('shop_a')
    return BusinessTools(catalog,split.rulebook,'2025-01-31')


MONTH={'kind':'month','year':2025,'month':1}


def test_full_column_scan_includes_late_cancel_return_and_counts(split_root,tmp_path):
    rows=[[str(i),'01/01/2025','Hoàn thành','A',1,'1.000đ','cái','Lê An'] for i in range(200)]
    rows += [['cancel','02/01/2025','ĐÃ HỦY','A',1,'1,45tr','cái','Lê An'],['return','03/01/2025','Đã trả hàng','A',1,'990.000đ','cái','Lê An']]
    setup_ledger(split_root,rows)
    t=tools_for(split_root,tmp_path)
    profile=t.catalog.semantic['status_profiles']['raw_0_ledger']
    assert profile['rows_scanned']==202 and profile['values_complete']
    assert {v['canonical'] for v in profile['values']}=={'completed','cancelled','returned'}
    assert t.metric({'name':'revenue','period':MONTH})['rows']==[[200000]]
    assert t.metric({'name':'orders','period':MONTH})['rows']==[[200]]
    assert t.profile('raw_0_ledger','State')['values'][0]['count']==1


def test_unseen_status_is_not_silently_counted_as_completed(split_root,tmp_path):
    setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A'],['b','01/01/2025','Chờ chủ quán xác nhận','A',1,999,'cái','A']])
    t=tools_for(split_root,tmp_path)
    assert any(v['canonical']=='unknown' for v in t.catalog.semantic['status_profiles']['raw_0_ledger']['values'])
    with pytest.raises(DataMateError,match='trạng thái'):
        t.metric({'name':'revenue','period':MONTH})


def test_mixed_money_and_quantity_with_declared_units(split_root,tmp_path):
    setup_ledger(split_root,[['a','1 thg 1 2025','Xong','A','1 cái','50.000đ','cái','A'],
                             ['b','02/01/2025','Xong','A','2 cái','1,45tr','cái','A'],
                             ['c','2025-01-03','Xong','A',3,'19,000.0000','cái','A']])
    t=tools_for(split_root,tmp_path)
    assert t.metric({'name':'revenue','period':MONTH})['rows']==[[1519000]]
    assert t.metric({'name':'quantity','period':MONTH})['rows']==[[6]]
    assert t.catalog.semantic['profiles']['raw_0_ledger']['Qty']['parse_errors']==0
    with pytest.raises(ValueError):
        parse_number('1.234')
    assert parse_number('1tr440')==1440000
    assert parse_number('1tr4')==1400000


def test_summary_rows_do_not_inherit_previous_order_or_double_count(split_root,tmp_path):
    setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A'],
                             ['','','','','',10,'','Tổng ngày 01/01'],
                             ['b','02/01/2025','Xong','A',1,20,'cái','B']])
    t=tools_for(split_root,tmp_path)
    assert t.catalog.semantic['excluded_summary_rows']==1
    assert t.metric({'name':'revenue','period':MONTH})['rows']==[[30]]
    assert t.metric({'name':'orders','period':MONTH})['rows']==[[2]]
    assert t.executor.execute('SELECT COUNT(*) FROM raw_0_ledger')['rows']==[[3]]


def test_notes_headers_and_iso_timestamp_do_not_create_false_missing_data(split_root,tmp_path):
    folder=setup_ledger(split_root,[])
    with (folder/'ledger.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(HEADERS+['Created'])
        writer.writerows([['a','01/01/2025','Xong','A',1,10,'cái','A','2025-01-01T07:29:30'],
                          HEADERS+['Created'],['','','','','','','','Ghi chú: số liệu ghi tay','']])
    (folder/'README.md').write_text(README.replace('## Quy ước','- **Created**: thời điểm đặt đơn\n## Quy ước'),encoding='utf-8')
    t=tools_for(split_root,tmp_path)
    assert t.executor.execute('SELECT order_timestamp FROM dm_orders')['rows']==[['2025-01-01T07:29:30']]
    assert t.metric({'name':'revenue','period':MONTH})['rows']==[[10]]


def test_blank_group_fields_and_duplicate_scope_preserve_other_orders(split_root,tmp_path):
    setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A'],
                             ['','','','B',1,20,'cái',''],['','','','B',1,20,'cái',''],
                             ['b','01/01/2025','Xong','B',1,20,'cái','B']])
    t=tools_for(split_root,tmp_path)
    assert t.catalog.semantic['deduplicated_rows']==1
    assert t.metric({'name':'revenue','period':MONTH})['rows']==[[50]]
    assert t.metric({'name':'orders','period':MONTH})['rows']==[[2]]


def test_partial_return_excludes_whole_order_and_unit_conversion_only_large_unit(split_root,tmp_path):
    folder=setup_ledger(split_root,[['a','01/01/2025','Xong','Bánh',87,87,'cái','A'],
                                    ['b','01/01/2025','Xong','Bánh',1,6,'hộp','A'],
                                    ['c','01/01/2025','Xong','Nước',100,100,'cái','A'],
                                    ['d','01/01/2025','Xong','Nước',1000,1000,'cái','A']])
    with (folder/'return.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['Ticket','Returned']);writer.writerow(['d',1])
    with (folder/'README.md').open('a',encoding='utf-8') as handle:
        handle.write('\n## Cột trong `return.csv`\n- **Ticket**: mã đơn\n- **Returned**: số lượng trả\n')
    t=tools_for(split_root,tmp_path)
    assert t.metric({'name':'quantity','period':MONTH,'product':'Bánh'})['rows']==[[93]]
    assert t.metric({'name':'rank_products','period':MONTH})['rows']==[['Nước']]
    assert t.metric({'name':'revenue','period':MONTH})['rows']==[[193]]


@pytest.mark.parametrize('epoch,serial,expected',[(None,45658,'2025-01-01'),(CALENDAR_MAC_1904,44197,'2025-01-02')])
def test_excel_serial_dates_use_workbook_epoch(split_root,tmp_path,epoch,serial,expected):
    folder=split_root/'data/shop_a';(folder/'sales.csv').unlink()
    wb=Workbook();ws=wb.active
    if epoch:
        wb.epoch=epoch
    ws.append(HEADERS);ws.append(['a',serial,'Xong','A',1,10,'cái','A']);wb.save(folder/'ledger.xlsx');wb.close()
    (folder/'README.md').write_text(README.replace('ledger.csv','ledger.xlsx'),encoding='utf-8')
    (split_root/'rulebook.md').write_text(RULES,encoding='utf-8')
    t=tools_for(split_root,tmp_path)
    assert t.executor.execute('SELECT order_date FROM dm_orders')['rows']==[[expected]]


def test_rulebook_change_and_unknown_wording_do_not_use_fixed_train_rules(split_root,tmp_path):
    setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A'],['b','01/01/2025','Đã hủy','A',1,20,'cái','A']])
    (split_root/'rulebook.md').write_text(RULES.replace('Đơn không huỷ và','Tính cả đơn huỷ và'),encoding='utf-8')
    t=tools_for(split_root,tmp_path)
    assert t.metric({'name':'revenue','period':MONTH})['rows']==[[30]]
    (split_root/'rulebook.md').write_text('Một bộ quy tắc hoàn toàn mới chưa diễn giải được.',encoding='utf-8')
    with pytest.raises(DataMateError,match='Luật split'):
        tools_for(split_root,tmp_path).metric({'name':'revenue','period':MONTH})


def test_business_id_does_not_select_normalization_or_answer_logic(split_root,tmp_path):
    folder=setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A']])
    shutil.copytree(folder,folder.parent/'completely_new_store')
    split=DatasetSplit(split_root);ingest=Ingestor(split,tmp_path/'cache')
    results=[BusinessTools(ingest.catalog(b),split.rulebook,'2025-01-31').metric({'name':'revenue','period':MONTH})['rows'] for b in ('shop_a','completely_new_store')]
    assert results==[[[10]],[[10]]]


def test_lookup_keeps_original_name_and_does_not_merge_numeric_variants(split_root,tmp_path):
    setup_ledger(split_root,[['a','01/01/2025','Xong','Cà phê (M)',1,10,'ly','An 1'],['b','01/01/2025','Xong','Cà phê (L)',1,20,'ly','An 2']])
    t=tools_for(split_root,tmp_path)
    assert t.lookup('product','ca phe m')['candidates'][0]['name']=='Cà phê (M)'
    assert t.lookup('customer','An')['ambiguous']
    assert t.lookup('customer','An 1')['candidates'][0]['name']=='An 1'


def test_empty_sql_cannot_prove_unanswerable_and_failed_query_has_no_stale_default(split_root,tmp_path,scripted_backend):
    setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A']])
    config=Settings(split_root=str(split_root),cache_dir=str(tmp_path/'cache'),backend='demo')
    backend=scripted_backend([{'action':'sql','sql':'SELECT COUNT(*) FROM dm_sales WHERE order_date=?','parameters':['1900-01-01']},
                              {'action':'final','type':'unanswerable'}, {'action':'final','type':'abstain'}])
    result=Agent(config,backend).run(Question('random_id','shop_a','Một câu hỏi chưa xác định','2025-01-31'))
    assert result.answer.type=='abstain' and 'evidence_id' in result.trace[1]['observation']['error']
    backend=scripted_backend([{'action':'sql','sql':'SELECT 12'}, {'action':'sql','sql':'SELECT "no_such_column" FROM calendar'},
                              {'action':'final','type':'number'},{'action':'final','type':'abstain'}])
    result=Agent(config,backend).run(Question('different_id','shop_a','q','2025-01-31'))
    assert result.answer.type=='abstain' and 'query_id' in result.trace[2]['observation']['error']


def test_known_metric_does_not_finalize_a_wrong_raw_status_filter(split_root,tmp_path,scripted_backend):
    setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A'],['b','02/01/2025','Đã hủy','A',1,999,'cái','A']])
    backend=scripted_backend([{'action':'sql','sql':'''SELECT SUM(Cash) FROM raw_0_ledger WHERE State NOT IN ('huỷ','trả hàng')'''},
        {'action':'final','type':'number'}, {'action':'metric','name':'revenue','period':MONTH},
        {'action':'final','type':'number','query_id':'r2'}])
    agent=Agent(Settings(split_root=str(split_root),cache_dir=str(tmp_path/'cache'),backend='demo'),backend)
    result=agent.run(Question('not_used','shop_a','Doanh thu tháng 1/2025?','2025-01-31'))
    assert result.answer.answer==10 and result.trace[1]['observation']['error']
    assert 'Đã hủy' in backend.messages[0][1]['content']


def test_extra_metric_filters_are_not_silently_ignored_and_novel_sql_uses_typed_facts(split_root,tmp_path,scripted_backend):
    folder=setup_ledger(split_root,[])
    with (folder/'ledger.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(HEADERS+['Pay'])
        writer.writerows([['a','01/01/2025','Xong','A',1,10,'cái','A','Tiền mặt'],
                          ['b','02/01/2025','Xong','A',1,20,'cái','B','Chuyển khoản']])
    (folder/'README.md').write_text(README.replace('## Quy ước','- **Pay**: hình thức thanh toán\n## Quy ước'),encoding='utf-8')
    t=tools_for(split_root,tmp_path)
    with pytest.raises(DataMateError,match='không tự bỏ qua'):
        t.metric({'name':'revenue','period':MONTH,'unknown_filter':'cash'})
    assert t.metric({'name':'orders','period':MONTH,'product':'A'})['rows']==[[2]]
    backend=scripted_backend([{'action':'sql','sql':"SELECT SUM(line_amount) FROM dm_sales WHERE payment_method='cash' AND order_date BETWEEN '2025-01-01' AND '2025-01-31'"},
                              {'action':'final','type':'number'}])
    agent=Agent(Settings(split_root=str(split_root),cache_dir=str(tmp_path/'cache'),backend='demo'),backend)
    result=agent.run(Question('arbitrary','shop_a','Doanh thu tiền mặt tháng 1/2025?','2025-01-31'))
    assert result.answer.answer==10,result.trace


def test_readme_order_key_outranks_ambiguous_receipt_header_alias(split_root,tmp_path):
    folder=setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A']])
    (folder/'receipts.csv').write_text('Mã phiếu,Mã hóa đơn,Thời gian,Giá trị\nR1,a,02/01/2025,10\n',encoding='utf-8')
    with (folder/'README.md').open('a',encoding='utf-8') as handle:
        handle.write('\n## Cột trong `receipts.csv`\n- **Mã phiếu**: mã phiếu\n- **Mã hóa đơn**: mã đơn\n- **Thời gian**: thời điểm thu\n- **Giá trị**: số tiền\n')
    t=tools_for(split_root,tmp_path)
    assert t.executor.execute('SELECT order_id,amount FROM dm_payments')['rows']==[['a',10]]


def test_undocumented_ledger_foreign_key_is_verified_from_all_values(split_root,tmp_path):
    folder=setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A'],['b','01/01/2025','Xong','A',1,20,'cái','A']])
    (folder/'receipts.csv').write_text('Mã phiếu,Mã hóa đơn,Thời gian,Giá trị\nR1,a,02/01/2025,10\nR2,b,03/01/2025,20\n',encoding='utf-8')
    t=tools_for(split_root,tmp_path)
    assert t.executor.execute('SELECT order_id,amount FROM dm_payments ORDER BY order_id')['rows']==[['a',10],['b',20]]
    binding=next(b for b in t.catalog.semantic['bindings'] if b['kind']=='payments')
    assert binding['relationship_evidence']['scope']=='all_distinct_values'


def test_historical_debt_uses_payments_to_date_includes_returns_excludes_cod(split_root,tmp_path):
    folder=setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,100,'cái','Lê An']])
    (folder/'orders.csv').write_text('ID,Date,State,CID,Client,Total,Paid,Balance,Method\n'
        'a,01/01/2025,Xong,C1,Lê An,100,60,40,debt\n'
        'b,02/01/2025,Đã trả hàng,C1,Lê An,50,0,50,debt\n'
        'c,02/01/2025,Xong,C2,Bình,500,0,500,cod\n',encoding='utf-8')
    (folder/'payments.csv').write_text('Ref,Date,Amount\na,01/01/2025,20\na,20/01/2025,40\n',encoding='utf-8')
    with (folder/'README.md').open('a',encoding='utf-8') as handle:
        handle.write('''
## Cột trong `orders.csv`
- **ID**: mã đơn
- **Date**: ngày bán
- **State**: trạng thái đơn
- **CID**: mã khách
- **Client**: tên khách
- **Total**: tổng tiền đơn
- **Paid**: đã thanh toán
- **Balance**: còn nợ
- **Method**: hình thức thanh toán
## Cột trong `payments.csv`
- **Ref**: mã đơn
- **Date**: thời điểm thu
- **Amount**: số tiền
''')
    t=tools_for(split_root,tmp_path)
    assert t.metric({'name':'debt_rank','date':'2025-01-10','limit':3})['rows']==[['Lê An',130]]
    assert t.metric({'name':'debt_rank','date':'2025-01-31','limit':3})['rows']==[['Lê An',90]]


@pytest.mark.parametrize('text',[
    '{"action":"schema","table":"a"}',
    'thought\n```json\n{"action":"schema","table":"a"}\n```',
    'call:schema{"table":"a"}',
    'call:schema:{"table":"a"}',
    '<|channel>thought\nprivate channel<channel|>{"action":"schema","table":"a"}<turn|>',
    '<|tool_call>call:schema{table:<|"|>a<|"|>}<tool_call|>',
])
def test_protocol_only_accepts_single_known_envelope(text):
    assert parse_action(text)=={'action':'schema','table':'a'}


@pytest.mark.parametrize('text',[
    '{"action":"sql","action":"final"}',
    '{"action":"schema","table":"a"}{"action":"final","type":"abstain"}',
    'some prose {"action":"final","type":"abstain"}',
    '<think>unclosed {"action":"final","type":"abstain"}',
    '{"action":"sql","sql":"SELECT 1","parameters":[NaN]}',
    '{"action":"sql","sql":"SELECT 1","parameters":[1e999]}',
    '<|tool_call>call:sql{sql:<|"|>SELECT 1<|"|>}<tool_call|><|tool_call>call:final{type:<|"|>abstain<|"|>}<tool_call|>',
    '<|tool_call>call:sql{sql:<|"|>SELECT 1<|"|>}',
])
def test_protocol_rejects_ambiguity_incomplete_and_nonfinite_json(text):
    with pytest.raises(DataMateError):
        parse_action(text)


def test_alias_names_are_not_value_mismatches_but_changed_values_are_rejected():
    result={'columns':['KH','amount'],'rows':[['Lê An',12]]}
    answer=answer_from_query('table',result)
    verify_legacy_answer({'answer':{'columns':['khach_hang','so_tien_no'],'rows':[['Lê An',12]]}},answer)
    with pytest.raises(DataMateError):
        verify_legacy_answer({'answer':{'columns':['a','b'],'rows':[['Lê An',999]]}},answer)
    assert answer_from_query('date',{'columns':['day','money'],'rows':[['03/01/2025',123]]},['day']).answer=='2025-01-03'
    assert answer_from_query('boolean',{'columns':['v'],'rows':[['Không']]}).answer is False


def test_native_sql_arguments_preserve_quotes_newlines_and_parameters():
    action=parse_action('<|tool_call>call:sql{sql:<|"|>SELECT "Trạng thái"\nFROM tab WHERE name=?<|"|>,parameters:[<|"|>Lê An<|"|>]}<tool_call|>')
    assert action=={'action':'sql','sql':'SELECT "Trạng thái"\nFROM tab WHERE name=?','parameters':['Lê An']}


def test_periods_use_as_of_rulebook_and_reject_wrong_comparison():
    policy=read_policy(RULES)
    assert resolve_period({'kind':'last_week'},'2025-07-11',policy)['start']=='2025-06-30'
    assert resolve_period({'kind':'previous_month_to_date'},'2025-03-31',policy)['end']=='2025-02-28'
    periods=question_periods('dthu tháng 5/2025 so với tháng trước','2025-07-11',policy)
    assert [p['start'] for p in periods]==['2025-05-01','2025-04-01']
    with pytest.raises(DataMateError,match='Kỳ hỏi'):
        verify_metric_request({'name':'best_day','period':{'kind':'month','year':2025,'month':2}},'Ngày nào tháng 1/2025 doanh thu cao nhất','2025-07-11',policy)
    with pytest.raises(DataMateError,match='Ngày chốt'):
        verify_metric_request({'name':'inventory','date':'2025-08-31'},'Tồn kho hôm nay là bao nhiêu','2025-07-11',policy)


def test_customer_codes_are_distinct_even_when_names_match(split_root,tmp_path):
    folder=setup_ledger(split_root,[])
    with (folder/'ledger.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(HEADERS+['ClientCode'])
        writer.writerows([['a','01/01/2025','Xong','A',1,10,'cái','Trần An','C1'],
                          ['b','01/01/2025','Xong','A',1,20,'cái','Trần An','C2'],
                          ['c','01/01/2025','Xong','A',1,30,'cái','Khách lẻ','']])
    (folder/'README.md').write_text(README.replace('## Quy ước','- **ClientCode**: mã khách\n## Quy ước'),encoding='utf-8')
    t=tools_for(split_root,tmp_path)
    assert t.metric({'name':'customers_with_orders','period':MONTH,'threshold':1})['rows']==[[2]]


def test_unknown_schema_is_unsupported_not_false_missing_proof(split_root,tmp_path):
    folder=setup_ledger(split_root,[])
    (folder/'ledger.csv').write_text('x1,x2,x3\nA,B,C\n',encoding='utf-8')
    (folder/'README.md').write_text('Một dạng dữ liệu chưa có mô tả.',encoding='utf-8')
    assert tools_for(split_root,tmp_path).check('Tồn kho hôm nay là bao nhiêu')['availability']=='unsupported'


def test_repeated_failure_stops_before_spending_all_steps(split_root,tmp_path,scripted_backend):
    setup_ledger(split_root,[['a','01/01/2025','Xong','A',1,10,'cái','A']])
    backend=scripted_backend([{'action':'sql','sql':'SELECT broken FROM calendar'}]*3)
    result=Agent(Settings(split_root=str(split_root),cache_dir=str(tmp_path/'cache'),backend='demo'),backend).run(Question('unused','shop_a','q','2025-01-31'))
    assert result.answer.type=='abstain' and result.model_calls==2
    assert result.trace[-1]['loop_stopped']


def test_full_value_distribution_pagination_never_claims_preview_complete(split_root,tmp_path):
    setup_ledger(split_root,[[str(i),'01/01/2025','Xong','A',1,i,'cái',f'Khách {i}'] for i in range(80)])
    t=tools_for(split_root,tmp_path)
    first=t.profile('raw_0_ledger','Client',limit=50)
    second=t.profile('raw_0_ledger','Client',offset=first['next_offset'],limit=50)
    assert first['distinct_count']==80 and not first['values_complete']
    assert second['next_offset'] is None and len(first['values'])+len(second['values'])==80
