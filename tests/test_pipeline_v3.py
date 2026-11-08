import csv
import json
import time

import pytest

from datamate.agent.planning import calculate, validate_plan
from datamate.agent.preparation import BusinessPreparer
from datamate.agent.protocol import answer_from_query
from datamate.agent.runner import Agent
from datamate.agent.tools import BusinessTools
from datamate.agent.verification import question_contract, question_periods, verify_metric_request
from datamate.config import Settings
from datamate.data.ingest import Ingestor
from datamate.data.interpretation import survey_catalog, validate_binding, validate_policy
from datamate.data.normalization import prepare_catalog
from datamate.data.rules import read_policy
from datamate.data.split import DatasetSplit
from datamate.domain import DataMateError, Question

RULES = '''## 1. Đơn hàng và doanh thu
Đơn không huỷ và không trả hàng. Đơn có trả hàng kể cả một phần: bỏ cả đơn. Đơn đang giao vẫn tính.
Doanh thu là tổng thành tiền các dòng hàng.
## 2. Thời gian
Tuần bắt đầu thứ Hai. Ca sáng 06:00–13:59, ca chiều 14:00–21:59.
## 3. Xếp hạng
Bán chạy theo số lượng quy về đơn vị cơ bản. Phí phục vụ không phải mặt hàng.
## 4. Công nợ
Đơn không huỷ. Đơn trả hàng vẫn tính. COD không phải khách nợ.
## 5. Giá vốn
Lãi gộp = doanh thu trừ số lượng nhân giá vốn ghi trên dòng bán.
## 8. Cách ghi
Ô để trống ở các dòng sau giống ô đó ở dòng đầu nhóm.
Hai dòng của cùng một đơn giống nhau ở mọi ô: tính một lần.
## 9. Đáp án
Phần trăm làm tròn 2 chữ số thập phân.
'''
HEADERS = ['Ref','Placed','State','Item','Qty','Cash','Unit','Client','Cost','Created']
ROLES = dict(order_id='Ref',order_date='Placed',status='State',product_name='Item',quantity='Qty',
             line_amount='Cash',unit='Unit',customer_name='Client',cost='Cost',timestamp='Created')
README = '''## Cột trong `journal.csv`
- **Ref**: mã đơn
- **Placed**: ngày bán
- **State**: trạng thái đơn
- **Item**: tên hàng
- **Qty**: số lượng
- **Cash**: thành tiền sau giảm
- **Unit**: đơn vị tính
- **Client**: tên khách
- **Cost**: giá vốn
- **Created**: thời điểm đặt đơn
Chốt nghĩa là hoàn thành. Bỏ nghĩa là huỷ.
Dữ liệu có đến hết 31/01/2025.
'''


@pytest.fixture
def business(split_root, tmp_path):
    folder = split_root/'data/shop_a'
    (folder/'sales.csv').unlink()
    with (folder/'journal.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(HEADERS)
        writer.writerows([
            ['a','01/01/2025','Xong','Trà (L)',2,100,'ly','An',30,'2025-01-01T07:00:00'],
            ['b','02/01/2025','Đã hủy','Trà (L)',7,900,'ly','An',10,'2025-01-02T15:00:00'],
            ['c','03/01/2025','Xong','Cà phê (L)',3,300,'ly','Bình',50,'2025-01-03T15:00:00'],
            ['d','20/12/2024','Xong','Trà (M)',4,400,'ly','Cúc',10,'2024-12-20T16:00:00'],
        ])
    (folder/'README.md').write_text(README,encoding='utf-8')
    (split_root/'rulebook.md').write_text(RULES,encoding='utf-8')
    split=DatasetSplit(split_root)
    catalog=Ingestor(split,tmp_path/'cache').catalog('shop_a')
    return split, BusinessTools(catalog,split.rulebook,'2025-01-31')


def plan(text, measure='revenue', operation='sum', answer_type='number', **extra):
    return {'action':'plan','measure':measure,'operation':operation,'type':answer_type,'population':'sales',
            'evidence':text,'periods':[{'kind':'year_to_date'}],'filters':[],**extra}


def execute(tools, text, proposal):
    accepted=validate_plan(proposal,Question('irrelevant','shop_a',text,'2025-01-31'),tools,README)
    result=calculate(accepted,tools)
    return answer_from_query(accepted['type'],result).answer


def test_year_to_date_quantities_profit_and_iso_shifts(business):
    _,tools=business
    text='Doanh thu từ đầu năm tới hôm nay'
    assert execute(tools,text,plan(text))==400
    text='Bán bao nhiêu ly size L từ đầu năm'
    conditions=[{'column':'base_unit','op':'eq','value':'ly','evidence':'bao nhiêu ly'},
                {'column':'product_name','op':'endswith','value':'(L)','evidence':'size L'}]
    assert execute(tools,text,plan(text,'quantity',filters=conditions))==5
    text='Lãi gộp từ đầu năm'
    assert execute(tools,text,plan(text,'gross_profit'))==190
    text='Doanh thu ca chiều từ đầu năm'
    conditions=[{'column':'order_timestamp','op':'shift','value':'afternoon','evidence':'ca chiều'}]
    assert execute(tools,text,plan(text,filters=conditions))==300
    text='Số đơn ca sáng từ đầu năm'
    conditions=[{'column':'order_timestamp','op':'shift','value':'morning','evidence':'ca sáng'}]
    assert execute(tools,text,plan(text,'orders','count',filters=conditions))==1


def test_cancel_count_and_ratio_share_a_full_population(business):
    _,tools=business
    text='Bao nhiêu đơn hủy từ đầu năm'
    cancelled=[{'column':'status','op':'eq','value':'cancelled','evidence':'đơn hủy'}]
    assert execute(tools,text,plan(text,'orders','count',population='all',filters=cancelled))==1
    text='Tỉ lệ đơn hủy từ đầu năm'
    assert execute(tools,text,plan(text,'orders','ratio','percent',population='all',numerator_filters=cancelled))==33.33
    text='Tỷ trọng doanh thu Trà từ đầu năm'
    assert execute(tools,text,plan(text,operation='ratio',answer_type='percent',
        numerator_filters=[{'column':'product_name','op':'contains','value':'Trà','evidence':'Trà'}]))==25


def test_lexical_hints_cannot_force_growth_or_one_customer(business):
    _,tools=business
    assert question_contract('Doanh thu nhóm A chiếm bao nhiêu %')['metrics']==[]
    text='3 khách mua nhiều tiền nhất từ đầu năm'
    assert execute(tools,text,plan(text,operation='rank',answer_type='list',group_by='customer_key',display='customer_name',limit=3))==['Bình','An']
    periods=question_periods('từ đầu năm tới hôm nay','2025-01-31',tools.policy)
    assert [(p['start'],p['end']) for p in periods]==[('2025-01-01','2025-01-31')]
    periods=question_periods('tháng này so tháng rồi','2025-03-31',tools.policy)
    assert [p['end'] for p in periods]==['2025-03-31','2025-02-28']
    periods=question_periods('từ 03/01/2025 đến 20/01/2025','2025-01-31',tools.policy)
    assert [(p['start'],p['end']) for p in periods]==[('2025-01-03','2025-01-20')]


def test_sql_gets_a_final_when_correct_and_repairs_wrong_population(business,tmp_path,scripted_backend):
    split,_=business
    text='Số đơn hủy từ đầu năm tới hôm nay'
    p=plan(text,'orders','count',population='all',filters=[{'column':'status','op':'eq','value':'cancelled','evidence':'đơn hủy'}])
    settings=Settings(split_root=str(split.root),cache_dir=str(tmp_path/'cache'),backend='demo')
    actions=[p,{'action':'sql','sql':"SELECT COUNT(*) FROM dm_orders WHERE status='cancelled'"},
             {'action':'final','type':'number'},{'action':'calculate'},{'action':'final','type':'number'}]
    result=Agent(settings,scripted_backend(actions)).run(Question('unused','shop_a',text,'2025-01-31'))
    assert result.answer.answer==1 and result.trace[2]['observation']['error']
    actions=[p,{'action':'sql','sql':"SELECT COUNT(*) FROM dm_orders_all WHERE status='cancelled' AND order_date BETWEEN ? AND ?",'parameters':['2025-01-01','2025-01-31']},
             {'action':'final','type':'number'}]
    result=Agent(settings,scripted_backend(actions)).run(Question('another','shop_a',text,'2025-01-31'))
    assert result.answer.answer==1 and result.trace[-1]['sql_validation']['scope']=='agreement_with_compiled_plan_and_source_quality'


def test_wrong_measure_or_output_type_cannot_be_saved_by_a_successful_metric(business,tmp_path,scripted_backend):
    split,_=business
    text='Bán bao nhiêu ly từ đầu năm'
    p=plan(text,'quantity')
    backend=scripted_backend([p,{'action':'metric','name':'revenue'},
                              {'action':'calculate'},{'action':'final','type':'number'}])
    result=Agent(Settings(split_root=str(split.root),cache_dir=str(tmp_path/'cache'),backend='demo'),backend).run(Question('unused','shop_a',text,'2025-01-31'))
    assert result.answer.answer==5
    assert 'đại lượng' in result.trace[1]['observation']['error']


def policy_proposal():
    policy=read_policy(RULES).to_dict()
    return {'action':'prepare','policy':{k:{'value':v,'evidence':RULES} for k,v in policy.items() if k!='unknown'}}


def binding_proposal(statuses=None):
    states=statuses or {'Xong':'completed','Đã hủy':'cancelled'}
    return {'action':'prepare','kind':'sales','roles':ROLES,'evidence':{k:v for k,v in ROLES.items()},
            'status_map':states,'status_evidence':{k:k for k in states},'attributes':{},'joins':[]}


def test_preparation_uses_same_backend_caches_and_keeps_mapping_evidence(business,tmp_path,scripted_backend):
    split,_=business
    settings=Settings(split_root=str(split.root),cache_dir=str(tmp_path/'adaptive'),backend='transformers')
    raw=Ingestor(split,tmp_path/'adaptive').catalog('shop_a',normalize=False)
    backend=scripted_backend([policy_proposal(),binding_proposal()])
    preparer=BusinessPreparer(settings,split)
    ready,trace=preparer.prepare_for_business(raw,'shop_a',backend,time.monotonic()+30)
    tools=BusinessTools(ready,split.rulebook,'2025-01-31')
    assert tools.metric({'name':'revenue','period':{'kind':'year_to_date'}})['rows']==[[400]]
    assert len(backend.messages)==2 and sum('raw' in e for e in trace)==2
    assert ready.semantic['bindings'][0]['origins']['Ref']=='LLM_HEADER'
    again,cached=BusinessPreparer(settings,split).prepare_for_business(raw,'shop_a',backend,time.monotonic()+30)
    assert again.fingerprint==ready.fingerprint and cached[0]['observation']['cache_hit']
    assert len(backend.messages)==2
    assert all('question_' not in json.dumps(m) and '"request"' not in json.dumps(m) for m in backend.messages)


def test_adaptive_run_logs_preparation_calls_and_requires_a_plan(business,tmp_path,scripted_backend):
    split,_=business
    settings=Settings(split_root=str(split.root),cache_dir=str(tmp_path/'adaptive'),backend='transformers',
                      schema_mode='llm',auto_execute_plan=False)
    text='Doanh thu từ đầu năm'
    backend=scripted_backend([policy_proposal(),binding_proposal(),{'action':'final','type':'number'},
                              plan(text),{'action':'calculate'},{'action':'final','type':'number'}])
    agent=Agent(settings,backend)
    result=agent.run(Question('unused','shop_a',text,'2025-01-31'))
    assert result.answer.answer==400 and result.model_calls==6
    assert sum(e.get('phase')=='preparation' and 'raw' in e for e in result.trace)==2
    assert any('Cần action plan' in e.get('observation',{}).get('error','') for e in result.trace)
    agent.backend=scripted_backend([plan(text),{'action':'calculate'},{'action':'final','type':'number'}])
    again=agent.run(Question('new_question','shop_a',text,'2025-01-31'))
    assert again.answer.answer==400 and again.model_calls==3
    assert again.trace[0]['observation']['cache_hit']


def test_ambiguous_status_is_preserved_and_renamed_documented_statuses_work(business,tmp_path,scripted_backend):
    split,_=business
    path=split.root/'data/shop_a/journal.csv'
    path.write_text(path.read_text(encoding='utf-8').replace('Xong','Chốt').replace('Đã hủy','Bỏ'),encoding='utf-8')
    raw=Ingestor(split,tmp_path/'raw').catalog('shop_a',normalize=False)
    survey=survey_catalog(raw)
    proposal=binding_proposal({'Chốt':'completed','Bỏ':'cancelled'})
    proposal['status_evidence']={'Chốt':'Chốt nghĩa là hoàn thành.','Bỏ':'Bỏ nghĩa là huỷ.'}
    b=validate_binding(proposal,raw.tables[0],survey['tables'][0],README,raw)
    interpretation={'policy':validate_policy(policy_proposal()['policy'],RULES),'bindings':[b],'unresolved':[]}
    catalog=prepare_catalog(raw,RULES,README,tmp_path/'raw',interpretation)
    assert BusinessTools(catalog,RULES,'2025-01-31').metric({'name':'revenue','period':{'kind':'year_to_date'}})['rows']==[[400]]
    path.write_text(path.read_text(encoding='utf-8').replace('Chốt','Hoàn'),encoding='utf-8')
    raw=Ingestor(split,tmp_path/'raw').catalog('shop_a',normalize=False)
    survey=survey_catalog(raw)
    proposal=binding_proposal({'Hoàn':'completed','Bỏ':'cancelled'})
    proposal['status_evidence']={'Hoàn':'Hoàn','Bỏ':'Bỏ nghĩa là huỷ.'}
    with pytest.raises(DataMateError,match='giải nghĩa'):
        validate_binding(proposal,raw.tables[0],survey['tables'][0],README,raw)
    proposal['status_map']['Hoàn']='unknown'
    b=validate_binding(proposal,raw.tables[0],survey['tables'][0],README,raw)
    interpretation['bindings']=[b]
    catalog=prepare_catalog(raw,RULES,README,tmp_path/'raw',interpretation)
    with pytest.raises(DataMateError,match='trạng thái'):
        BusinessTools(catalog,RULES,'2025-01-31').metric({'name':'revenue','period':{'kind':'year_to_date'}})


def test_invalid_citation_and_unsupported_policy_do_not_fall_back_to_train(business):
    assert validate_policy({},RULES)['sales_statuses'] is None
    with pytest.raises(DataMateError,match='nguyên văn'):
        validate_policy({'week_start':{'value':0,'evidence':'Câu tự bịa'}},RULES)
    with pytest.raises(DataMateError,match='chưa có phép'):
        validate_policy({'new_profit_formula':{'value':True,'evidence':RULES}},RULES)


def test_missing_mapping_is_not_a_zero_or_missing_data_proof(business,tmp_path):
    split,_=business
    raw=Ingestor(split,tmp_path/'raw').catalog('shop_a',normalize=False)
    catalog=prepare_catalog(raw,RULES,README,tmp_path/'raw',{'policy':read_policy(RULES).to_dict(),'bindings':[], 'unresolved':[raw.tables[0]['name']]})
    tools=BusinessTools(catalog,RULES,'2025-01-31')
    assert tools.check('Khách còn nợ')['availability']=='unsupported'
    result=tools.executor.execute('SELECT COUNT(*) FROM dm_orders_all')
    with pytest.raises(DataMateError,match='chưa ánh xạ'):
        tools.verify_fact_sql(result,'Tổng số đơn')


def test_duplicate_dimension_key_is_rejected_before_money_is_multiplied(business):
    _,tools=business
    text='Doanh thu theo danh mục từ đầu năm'
    p=plan(text,joins=[{'table':'dm_sales_all','left':'order_id','right':'dm_sales_all.product_name'}])
    with pytest.raises(DataMateError,match='không duy nhất'):
        validate_plan(p,Question('unused','shop_a',text,'2025-01-31'),tools)


def test_missing_joined_attribute_cannot_be_silently_dropped_from_a_total(business):
    _,tools=business
    text='Doanh thu mặt hàng có giá niêm yết lớn hơn 0 từ đầu năm'
    p=plan(text,joins=[{'table':'dm_products','left':'product_name','right':'dm_products.product_name'}],
           filters=[{'column':'dm_products.price','op':'gt','value':0,'evidence':'giá niêm yết lớn hơn 0'}])
    with pytest.raises(DataMateError,match='không tính tổng một phần'):
        execute(tools,text,p)


def test_rulebook_or_readme_change_invalidates_interpretation_cache(business,tmp_path,scripted_backend):
    split,_=business
    settings=Settings(split_root=str(split.root),cache_dir=str(tmp_path/'adaptive'),backend='transformers')
    raw=Ingestor(split,tmp_path/'adaptive').catalog('shop_a',normalize=False)
    backend=scripted_backend([policy_proposal(),binding_proposal()])
    first,_=BusinessPreparer(settings,split).prepare_for_business(raw,'shop_a',backend,time.monotonic()+30)
    readme=split.root/'data/shop_a/README.md'
    readme.write_text(README+'\nGhi chú được cập nhật.',encoding='utf-8')
    raw=Ingestor(split,tmp_path/'adaptive').catalog('shop_a',normalize=False)
    backend=scripted_backend([binding_proposal()])
    second,_=BusinessPreparer(settings,split).prepare_for_business(raw,'shop_a',backend,time.monotonic()+30)
    assert first.fingerprint!=second.fingerprint and len(backend.messages)==1  # rulebook is reused


def test_unknown_column_names_and_file_name_do_not_block_a_new_mapping(business,tmp_path,scripted_backend):
    split,_=business
    folder=split.root/'data/shop_a'
    original=folder/'journal.csv'
    rows=list(csv.reader(original.open(encoding='utf-8')))
    renamed={old:f'field_{i:02}' for i,old in enumerate(HEADERS)}
    original.unlink()
    with (folder/'new_export.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow([renamed[h] for h in HEADERS]);writer.writerows(rows[1:])
    (folder/'README.md').write_text('Một định dạng mới không theo mẫu ghi chú cũ.',encoding='utf-8')
    raw=Ingestor(split,tmp_path/'new').catalog('shop_a',normalize=False)
    proposal=binding_proposal()
    proposal['roles']={k:renamed[v] for k,v in ROLES.items()}
    proposal['evidence']={k:renamed[v] for k,v in ROLES.items()}
    backend=scripted_backend([policy_proposal(),proposal])
    settings=Settings(split_root=str(split.root),cache_dir=str(tmp_path/'new'),backend='transformers')
    ready,_=BusinessPreparer(settings,split).prepare_for_business(raw,'shop_a',backend,time.monotonic()+30)
    assert BusinessTools(ready,RULES,'2025-01-31').metric({'name':'revenue','period':{'kind':'year_to_date'}})['rows']==[[400]]
    assert '"kind": "other"' in backend.messages[1][1]['content']  # The candidate failed, the proposal was still accepted.


def test_dimension_attribute_is_typed_and_parse_errors_are_not_treated_as_false(business,tmp_path):
    split,_=business
    folder=split.root/'data/shop_a'
    path=folder/'flags.csv'
    path.write_text('ItemKey,RxFlag\nTrà (L),TRUE\nCà phê (L),false\nTrà (M),0\n',encoding='utf-8')
    for malformed in (False,True):
        if malformed:
            path.write_text(path.read_text(encoding='utf-8').replace('false','not sure'),encoding='utf-8')
        raw=Ingestor(split,tmp_path/'typed').catalog('shop_a',normalize=False)
        survey=survey_catalog(raw)
        source=next(t for t in raw.tables if t['file']=='journal.csv')
        dim=next(t for t in raw.tables if t['file']=='flags.csv')
        source_survey=next(t for t in survey['tables'] if t['name']==source['name'])
        dim_survey=next(t for t in survey['tables'] if t['name']==dim['name'])
        b=validate_binding(binding_proposal(),source,source_survey,README,raw)
        d=validate_binding({'kind':'other','roles':{'product_name':'ItemKey'},'evidence':{'product_name':'ItemKey'},
                            'attributes':{'prescription':{'column':'RxFlag','type':'boolean','evidence':'RxFlag'}}},dim,dim_survey,README,raw)
        catalog=prepare_catalog(raw,RULES,README,tmp_path/'typed',{'policy':read_policy(RULES).to_dict(),'bindings':[b,d],'unresolved':[]})
        tools=BusinessTools(catalog,RULES,'2025-01-31')
        clean=next(x['clean_table'] for x in catalog.semantic['bindings'] if x['table']==dim['name'])
        text='Doanh thu hàng không kê đơn từ đầu năm'
        p=plan(text,joins=[{'table':clean,'left':'product_name','right':clean+'.ItemKey'}],
               filters=[{'column':clean+'.RxFlag','op':'not_true','evidence':'không kê đơn'}])
        if malformed:
            with pytest.raises(DataMateError,match='lỗi parse'):
                execute(tools,text,p)
        else:
            assert execute(tools,text,p)==300
            assert tools.executor.execute(f'SELECT RxFlag FROM "{dim["name"]}" WHERE ItemKey=?',['Trà (L)'])['rows']==[['TRUE']]
            assert tools.executor.execute(f'SELECT RxFlag FROM "{clean}" WHERE ItemKey=?',['Trà (L)'])['rows']==[[1]]


def test_unknown_plan_fields_are_not_silently_dropped(business):
    _,tools=business
    text='Doanh thu từ đầu năm'
    with pytest.raises(DataMateError,match='không tự bỏ qua'):
        execute(tools,text,plan(text,discount_filter='some-new-condition'))


def test_series_preserves_requested_group_order_in_one_period(business):
    _,tools=business
    text='Doanh thu Trà rồi Cà phê từ đầu năm'
    groups=[[{'column':'product_name','op':'eq','value':'Trà (L)','evidence':'Trà'}],
            [{'column':'product_name','op':'eq','value':'Cà phê (L)','evidence':'Cà phê'}]]
    assert execute(tools,text,plan(text,operation='series',answer_type='list',series_filters=groups,output_order=['Trà','Cà phê']))==[100,300]


def test_quantity_units_and_sheet_year_survive_adaptive_validation(business,tmp_path):
    from openpyxl import Workbook
    split,_=business
    folder=split.root/'data/shop_a';(folder/'journal.csv').unlink()
    wb=Workbook();ws=wb.active;ws.title='Dữ liệu 2025'
    ws.append(HEADERS);ws.append(['a','01/01','Xong','Trà (L)','2 ly',100,'ly','An',30,'2025-01-01T07:00:00'])
    wb.save(folder/'new.xlsx');wb.close()
    raw=Ingestor(split,tmp_path/'raw').catalog('shop_a',normalize=False)
    survey=survey_catalog(raw)
    b=validate_binding(binding_proposal({'Xong':'completed'}),raw.tables[0],survey['tables'][0],README,raw)
    catalog=prepare_catalog(raw,RULES,README,tmp_path/'raw',{'policy':read_policy(RULES).to_dict(),'bindings':[b],'unresolved':[]})
    tools=BusinessTools(catalog,RULES,'2025-01-31')
    assert tools.executor.execute('SELECT order_date,base_quantity FROM dm_sales')['rows']==[['2025-01-01',2]]
    assert tools.executor.execute(f'SELECT Placed,Qty FROM "{raw.tables[0]["name"]}"')['rows']==[['01/01','2 ly']]


def test_historical_cutoff_comes_from_plan_and_single_day_period_is_not_leaked_to_metric(business):
    _,tools=business
    accepted={'measure':'debt','operation':'custom','type':'table','population':'debt',
              'filters':[],'resolved_periods':[],'date':'2025-01-10'}
    action={'name':'debt_rank','period':{'kind':'range','start':'2025-01-10','end':'2025-01-10'}}
    verify_metric_request(action,'Bảng công nợ tại ngày được hỏi','2025-01-31',tools.policy,plan=accepted)
    assert action['date']=='2025-01-10' and 'period' not in action


def test_trajectory_reuses_actual_prompt_and_does_not_train_preparation_as_question_actions(split_root,tmp_path,scripted_backend):
    from datamate.training.dataset import prepare_training, source_manifest
    split=DatasetSplit(split_root);question=split.questions()[0]
    backend=scripted_backend([{'action':'sql','sql':'SELECT SUM(num_en("Tiền")) FROM raw_0_sales'},
                              {'action':'final','type':'number'}])
    result=Agent(Settings(split_root=str(split_root),cache_dir=str(tmp_path/'cache'),backend='demo'),backend).run(question)
    assert result.answer.answer==4900
    record={'id':question.id,'business_id':question.business_id,'rulebook_sha256':split.rulebook_hash,**result.to_dict(question.id)}
    record['trace'].insert(0,{'phase':'preparation','raw':'not a question-action target','observation':{'error':'a repaired preparation step'}})
    logs=tmp_path/'logs';logs.mkdir();(logs/'one.json').write_text(json.dumps(record,ensure_ascii=False),encoding='utf-8')
    sources=tmp_path/'sources.json';sources.write_text(json.dumps({'sources':[{**source_manifest(split_root),'role':'competition_training'}]}),encoding='utf-8')
    output=tmp_path/'training'
    report=prepare_training(split_root,output,kind='trajectories',logs=logs,sources=sources)
    assert report['train_examples']+report['validation_examples']==1
    exported=[json.loads(line) for name in ('train.jsonl','validation.jsonl') for line in (output/name).read_text(encoding='utf-8').splitlines()]
    assert exported[0]['messages'][:2]==record['initial_messages']
    assert 'not a question-action target' not in json.dumps(exported)
