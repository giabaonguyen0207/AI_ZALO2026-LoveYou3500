"""Shared question-plan/data regressions for every selectable model."""
import csv
import json
from pathlib import Path
import sqlite3

import pytest

from datamate.agent.planning import validate_plan,calculate
from datamate.agent.protocol import answer_from_query
from datamate.agent.runner import Agent
from datamate.agent.tools import BusinessTools
from datamate.config import Settings
from datamate.data.binding import bind_table
from datamate.data.dates import date_epoch
from datamate.data.ingest import Ingestor
from datamate.data.sql import SQLExecutor
from datamate.data.transforms import typed_date
from datamate.domain import DataMateError,Question
from test_pipeline_v3 import business,plan,README


def execute(tools,proposal,text='Doanh thu từ đầu năm'):
    accepted=validate_plan(proposal,Question('any','shop_a',text,'2025-01-31'),tools,README)
    result=calculate(accepted,tools)
    return answer_from_query(accepted['type'],result).answer


@pytest.mark.parametrize('mutation',[
    {'group_by':'product_key'}, {'limit':5},
    {'numerator_filters':[{'column':'status','op':'eq','value':'completed','evidence_id':'Q'}]},
    {'periods':[{'kind':'today'},{'kind':'yesterday'}]},
    {'threshold':100},
    {'date':'2025-01-01'},
])
def test_compiler_never_silently_ignores_operation_fields(business,mutation):
    _,tools=business
    with pytest.raises(DataMateError):execute(tools,plan('Doanh thu từ đầu năm',**mutation))


def test_contradictory_equal_filters_rejected_but_series_branches_allowed(business):
    _,tools=business;text='Doanh thu hai nhóm'
    conditions=[{'column':'product_name','op':'eq','value':name,'evidence_id':'Q'} for name in ('Trà (L)','Cà phê (L)')]
    with pytest.raises(DataMateError,match='loại trừ'):execute(tools,plan(text,filters=conditions),text)
    assert execute(tools,plan(text,operation='series',answer_type='list',series_filters=[[conditions[0]],[conditions[1]]]),text)==[100,300]


def test_ratio_must_not_ignore_second_period(business):
    _,tools=business;text='Thay đổi so kỳ trước'
    with pytest.raises(DataMateError,match='một kỳ'):
        execute(tools,plan(text,operation='ratio',answer_type='percent',periods=[{'kind':'today'},{'kind':'yesterday'}],numerator_filters=[]),text)


def test_growth_named_month_cannot_use_today_anchored_previous_month(business):
    _,tools=business;text='Tháng 12 so tháng trước'
    with pytest.raises(DataMateError,match='neo theo as_of_date'):
        execute(tools,plan(text,operation='growth',answer_type='percent',periods=[{'kind':'month','month':12,'year':2024},{'kind':'previous_month_to_date'}]),text)
    from datamate.data.rules import resolve_period
    resolved=resolve_period({'kind':'month_offset','anchor':{'kind':'month','month':1,'year':2025},'months':-1},'2025-09-03',tools.policy)
    assert (resolved['start'],resolved['end'])==('2024-12-01','2024-12-31')
    with pytest.raises(DataMateError,match='không được dùng'):
        resolve_period({'kind':'this_month','month':12},'2025-09-03',tools.policy)


def test_repeated_period_assertions_must_match_the_question_anchor(business):
    _,tools=business;text='Doanh thu hôm nay'
    assert execute(tools,plan(text,periods=[{'kind':'today','date':'2025-01-31','as_of_date':'2025-01-31'}]),text)==0
    for extra in ({'date':'2025-01-30'},{'as_of_date':'2025-01-30'}):
        with pytest.raises(DataMateError,match='mâu thuẫn|khác mốc'):
            execute(tools,plan(text,periods=[{'kind':'today',**extra}]),text)


def test_best_day_redundant_fields_are_checked_not_ignored(business):
    _,tools=business;text='Ngày doanh thu cao nhất'
    assert execute(tools,plan(text,operation='best_day',answer_type='date',group_by='order_date',display='order_date',limit=1),text)=='2025-01-03'
    for changed in ({'group_by':'customer_key'},{'display':'customer_name'},{'limit':2}):
        with pytest.raises(DataMateError):
            execute(tools,plan(text,operation='best_day',answer_type='date',**changed),text)


def test_threshold_applies_after_day_or_order_aggregation(business):
    _,tools=business;text='Số ngày hoặc đơn đạt 200'
    assert execute(tools,plan(text,operation='day_count',threshold=200),text)==1
    assert execute(tools,plan(text,operation='order_count',threshold=200),text)==1
    with pytest.raises(DataMateError,match='sau tổng hợp'):
        execute(tools,plan(text,operation='day_count',threshold=200,filters=[{'column':'line_amount','op':'ge','value':200,'evidence_id':'Q'}]),text)
    tools.policy.money_places=0
    assert execute(tools,plan(text,operation='average',group_by='order_id'),text)==200


def test_no_match_is_not_zero_but_known_product_with_no_sales_in_period_is_zero(business):
    _,tools=business;text='Doanh thu hàng A'
    with pytest.raises(DataMateError,match='không khớp'):
        execute(tools,plan(text,filters=[{'column':'product_name','op':'eq','value':'imaginary','evidence_id':'Q'}]),text)
    assert execute(tools,plan(text,periods=[{'kind':'range','start':'2025-01-01','end':'2025-01-01'}],entities={'product':'Cà phê (L)'}),text)==0


def test_text_serial_dates_require_epoch_evidence():
    assert typed_date('45658',None,'1899-12-30T00:00:00')=='2025-01-01'
    assert typed_date('45658',None,'1904-01-01T00:00:00')=='2029-01-02'
    with pytest.raises(ValueError):typed_date('45658')
    binding={'roles':{'order_date':'Day','timestamp':'Created'}}
    epoch,evidence=date_epoch({},[{'Day':'45658','Created':'2025-01-01 08:10:00'}],binding,None)
    assert epoch=='1899-12-30T00:00:00' and evidence['paired_rows']==1
    assert date_epoch({},[{'Day':'45658'}],{'roles':{'order_date':'Day'}},None)[0] is None
    assert date_epoch({},[{'Day':'45658'}],{'roles':{'order_date':'Day'}},'2031-01-01')[0] is None


def test_pos_aliases_use_identity_context_and_readme_independent_roles():
    table={'name':'any','file':'any.csv','columns':['MaKhachHang','TenGoi','LoaiKhachHang','NamNu']}
    binding=bind_table(table,'')
    assert binding['kind']=='customers' and binding['roles']['customer_name']=='TenGoi'
    assert binding['roles']['customer_group']=='LoaiKhachHang' and binding['roles']['gender']=='NamNu'
    binding=bind_table({'name':'any','file':'any.csv','columns':['MaHoaDon','SoTienThu','NgayGioThu','MaPhieuThu']},'')
    assert binding['kind']=='payments' and binding['roles']['payment_date']=='NgayGioThu'


def test_cte_view_subquery_lineage_and_read_only_guard(business):
    _,tools=business
    for sql in ('WITH x AS (SELECT COUNT(*) n FROM dm_sales) SELECT n FROM x',
                'SELECT n FROM (SELECT COUNT(*) n FROM dm_sales)',
                'WITH x AS (SELECT COUNT(*) n FROM dm_orders_all) SELECT n FROM x'):
        result=tools.executor.execute(sql)
        assert result['read_tables'] and result['logical_tables']
        assert tools.verify_fact_sql(result,'',plan={'population':'all'})
    with pytest.raises(DataMateError):tools.executor.execute('DELETE FROM dm_sales_all')
    with pytest.raises(DataMateError):tools.verify_fact_sql(tools.executor.execute('SELECT 42'),'')


def test_missing_attribute_is_not_replaced_by_customer_name(business):
    _,tools=business
    proof=tools.check('Khách nữ đóng góp bao nhiêu phần trăm?')
    assert proof['availability']=='missing' and 'gender' in proof['missing']
    assert 'gender' in tools.check('Khách nữ đóng góp bao nhiêu?',plan=plan('Khách nữ'))['missing']


def test_reviewed_plan_is_computed_then_model_can_correct_its_scope(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu Trà từ đầu năm'
    broad=plan(text)
    specific=plan(text,entities={'product':'Trà (L)'})
    backend=scripted_backend([broad,specific,{'action':'final','type':'number','query_id':'r1'}])
    result=Agent(Settings(split_root=str(split.root),cache_dir=str(tmp_path/'run'),mode='self_review'),backend).run(Question('any','shop_a',text,'2025-01-31'))
    assert result.answer.answer==100 and result.model_calls==3
    assert result.trace[0]['review_requested'] and result.trace[1]['review_requested']
    assert json.loads(result.initial_messages[1]['content'])['request_contract']['result_review_required']


def write_csv(path,headers,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(headers);writer.writerows(rows)


def test_observed_catalog_groups_and_prescription_are_joined_by_unique_identity(business,tmp_path):
    split,_=business;folder=split.root/'data/shop_a'
    write_csv(folder/'products.csv',['Mã hàng','Tên hàng','Nhóm hàng','Thuốc kê đơn','Đơn vị cơ sở'],[
        ['P1','Trà (L)','Đồ uống','Có','ly'],['P2','Cà phê (L)','Đồ uống','Không','ly'],
        ['P3','Trà (M)','Đồ uống','','ly']])
    catalog=Ingestor(split,tmp_path/'catalog').catalog('shop_a');tools=BusinessTools(catalog,split.rulebook,'2025-01-31')
    text='Doanh thu nhóm đồ uống không kê đơn'
    assert catalog.semantic['domains']['product_group']==['Đồ uống']
    assert execute(tools,plan(text,filters=[{'column':'product_group','op':'eq','value':'do uong','evidence_id':'Q'},
                                         {'column':'prescription','op':'not_true','evidence_id':'Q'}]),text)==300
    assert tools.lookup('product','Cà phê (L)')['candidates'][0]['key']=='P2'
    # A malformed flag cannot silently become a blank non-prescription flag.
    write_csv(folder/'products.csv',['Mã hàng','Tên hàng','Nhóm hàng','Thuốc kê đơn','Đơn vị cơ sở'],[
        ['P1','Trà (L)','Đồ uống','maybe','ly'],['P2','Cà phê (L)','Đồ uống','Không','ly']])
    catalog=Ingestor(split,tmp_path/'catalog').catalog('shop_a');tools=BusinessTools(catalog,split.rulebook,'2025-01-31')
    with pytest.raises(DataMateError,match='lỗi parse'):
        execute(tools,plan(text,filters=[{'column':'prescription','op':'not_true','evidence_id':'Q'}]),text)


def test_ambiguous_dimension_names_do_not_choose_a_category():
    from datamate.data.dimensions import dimensions,resolve_dimension
    bindings=[{'table':'catalog','kind':'products','roles':{'product_id':'id','product_name':'name','product_group':'group'}}]
    index=dimensions(bindings,{'catalog':[{'id':'A','name':'Cùng tên','group':'A'},
                                          {'id':'B','name':'Cùng tên','group':'B'}]},'product')
    assert resolve_dimension(None,'Cung ten',index)==(None,True)
    assert resolve_dimension('B','Cùng tên',index)[0]['product_group']=='B'


def test_name_only_conversions_keep_working_when_catalog_adds_product_ids(business,tmp_path):
    split,_=business;folder=split.root/'data/shop_a'
    path=folder/'journal.csv'
    with path.open(encoding='utf-8',newline='') as handle:rows=list(csv.reader(handle))
    rows[1][6]='khay'
    write_csv(path,rows[0],rows[1:])
    write_csv(folder/'products.csv',['Mã hàng','Tên hàng','Đơn vị'],[['P1','Trà (L)','ly'],['P2','Cà phê (L)','ly']])
    write_csv(folder/'conversion.csv',['Tên sản phẩm','Đơn vị quy đổi','Đơn vị cơ bản','Tỉ lệ quy đổi'],[['Trà (L)','khay','ly',12]])
    catalog=Ingestor(split,tmp_path/'units').catalog('shop_a');tools=BusinessTools(catalog,split.rulebook,'2025-01-31')
    text='Số ly Trà (L) từ đầu năm'
    assert execute(tools,plan(text,measure='quantity',entities={'product':'Trà (L)'},unit='ly'),text)==24
    assert tools.executor.execute('SELECT product_id,factor FROM dm_conversions')['rows']==[['P1',12]]


def test_serial_date_is_normalized_during_ingestion_with_paired_timestamp(split_root,tmp_path):
    folder=split_root/'data/shop_a';(folder/'sales.csv').unlink()
    write_csv(folder/'pos.csv',['MaHoaDon','NgayBan','NgayGio','TrangThai','TenHang','SoLuong','ThanhTien','DonVi'],[
        ['A','45658','2025-01-01 08:30:00','Xong','Cà phê',2,50,'ly']])
    (folder/'README.md').write_text('NgayGio: thời điểm đặt đơn. Dữ liệu có đến hết 31/01/2025.',encoding='utf-8')
    from test_pipeline_v3 import RULES
    (split_root/'rulebook.md').write_text(RULES,encoding='utf-8')
    from datamate.data.split import DatasetSplit
    split=DatasetSplit(split_root);catalog=Ingestor(split,tmp_path/'serial').catalog('shop_a')
    tools=BusinessTools(catalog,split.rulebook,'2025-01-31')
    assert tools.executor.execute('SELECT order_date FROM dm_sales')['rows']==[['2025-01-01']]
    assert execute(tools,plan('Doanh thu từ đầu năm'))==50


def test_stock_ledger_uses_last_observation_not_sum_of_snapshots(split_root,tmp_path):
    folder=split_root/'data/shop_a'
    write_csv(folder/'stock.csv',['Mã hàng','Tên hàng','Tồn cuối','Ngày giờ phát sinh','Mã chi nhánh'],[
        ['P1','Vật liệu X',20,'2025-01-01 08:00:00','L1'],
        ['P1','Vật liệu X',16,'2025-01-01 12:00:00','L1'],
        ['P1','Vật liệu X',9,'2025-01-02 09:00:00','L1']])
    (folder/'README.md').write_text('Dữ liệu có đến hết 31/01/2025. Mỗi dòng thẻ kho ghi tồn sau một giao dịch.',encoding='utf-8')
    from datamate.data.split import DatasetSplit
    split=DatasetSplit(split_root);catalog=Ingestor(split,tmp_path/'cache').catalog('shop_a');tools=BusinessTools(catalog,split.rulebook,'2025-01-01')
    assert catalog.semantic['capabilities']['stock_ledger'] and not catalog.semantic['capabilities']['inventory_snapshot']
    result=tools.metric({'name':'inventory','product':'Vật liệu X','date':'2025-01-01'})
    assert result['rows']==[[16]]
    assert tools.metric({'name':'inventory','product':'Vật liệu X','date':'2025-02-01'})['availability']=='missing'


def test_debt_flow_uses_receipt_dates_and_purchase_day_not_balance_snapshot(split_root,tmp_path):
    folder=split_root/'data/shop_a';(folder/'sales.csv').unlink()
    write_csv(folder/'orders.csv',['Order ID','Ngày','Trạng thái','Tổng tiền','Khách đã trả','Hình thức thanh toán','Tên KH'],[
        ['a','2025-01-01','Xong',100,70,'debt','An'],['b','2024-12-20','Xong',80,30,'debt','An'],
        ['c','2025-01-02','Trả hàng',60,10,'debt','An'],['d','2025-01-03','Hủy',50,0,'debt','An'],
        ['e','2025-01-02','Xong',200,100,'cod','An'],['f','2025-01-02','Xong',40,40,'cash','An']])
    write_csv(folder/'receipts.csv',['Mã hóa đơn','Ngày thu','Số tiền thu'],[
        ['a','2025-01-01',50],['a','2025-01-03',20],['b','2025-01-04',30],['c','2025-01-02',10],
        ['e','2025-01-02',100],['f','2025-01-02',40]])
    from test_pipeline_v3 import RULES
    (split_root/'rulebook.md').write_text(RULES,encoding='utf-8')
    from datamate.data.split import DatasetSplit
    split=DatasetSplit(split_root);catalog=Ingestor(split,tmp_path/'cache').catalog('shop_a');tools=BusinessTools(catalog,split.rulebook,'2025-01-31')
    tools.policy.debt_flow_supported=True
    text='Nợ thêm và thu nợ trong tháng'
    p=plan(text,'debt','debt_flow','list',population='debt',flow='both',entities={'customer':'An'})
    assert execute(tools,p,text)==[100,50]
    tools.executor=SQLExecutor(catalog.database)
    tools.policy.debt_flow_supported=False
    with pytest.raises(DataMateError,match='chưa xác nhận'):execute(tools,p,text)
