"""End-to-end control flow with scripted model outputs, not a model benchmark."""
import json

import pytest

from datamate.agent.language import abbreviation_hints
from datamate.agent.runner import Agent
from datamate.config import Settings
from datamate.data.binding import bind_table
from datamate.data.interpretation import evidence_in, merge_binding, merge_policy
from datamate.data.rules import read_policy
from datamate.debug_log import render_debug_log
from datamate.domain import Question
from test_pipeline_v3 import business, plan, README, RULES


def make_agent(split,tmp_path,backend,**options):
    return Agent(Settings(split_root=str(split.root),cache_dir=str(tmp_path/'fast'),backend='transformers',**options),backend)


def test_code_first_executes_and_projects_a_plan_in_one_model_call(business,tmp_path,scripted_backend):
    split,_=business
    text='Doanh thu từ đầu năm'
    p=plan(text);p.pop('evidence');p['evidence_id']='Q'
    agent=make_agent(split,tmp_path,scripted_backend([p]))
    result=agent.run(Question('arbitrary','shop_a',text,'2025-01-31'))
    assert result.answer.to_dict()=={'type':'number','answer':400}
    assert result.model_calls==1
    assert not any(e.get('phase')=='preparation' for e in result.trace)
    assert result.trace[0]['auto_final']['answer']==400
    assert result.trace[0]['execution']['sql'] and result.trace[0]['execution']['parameters']
    assert json.loads(result.initial_messages[1]['content'])['request_contract']['plan_required']
    log=render_debug_log({'id':'arbitrary','submission':result.answer.to_dict(),'trace':result.trace})
    assert 'Code chiếu đáp án' in log and result.trace[0]['execution']['sql'] in log


@pytest.mark.parametrize('measure,operation,kind,extra,expected',[
    ('quantity','sum','number',{},5),('gross_profit','sum','number',{},190),
    ('orders','ratio','percent',{'population':'all','numerator_filters':[{'column':'status','op':'eq','value':'cancelled','evidence_id':'Q'}]},33.33),
    ('revenue','sum','number',{'filters':[{'column':'order_timestamp','op':'shift','value':'afternoon','evidence_id':'Q'}]},300),
])
def test_one_call_keeps_units_cost_status_ratio_and_shift_guards(business,tmp_path,scripted_backend,measure,operation,kind,extra,expected):
    split,_=business
    text='Số lượng, lãi gộp, tỷ lệ đơn hủy hoặc doanh thu ca chiều từ đầu năm'
    p=plan(text,measure,operation,kind,**extra)
    result=make_agent(split,tmp_path,scripted_backend([p])).run(Question('unrelated','shop_a',text,'2025-01-31'))
    assert result.answer.answer==expected and result.model_calls==1


def test_turning_off_mapping_does_not_turn_off_planning(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu từ đầu năm'
    backend=scripted_backend([{'action':'metric','name':'revenue'},plan(text)])
    result=make_agent(split,tmp_path,backend,schema_mode='code_only').run(Question('x','shop_a',text,'2025-01-31'))
    assert result.model_calls==2 and result.answer.answer==400
    assert 'Cần action plan' in result.trace[0]['observation']['error']


def test_inline_sql_is_checked_and_projected_without_another_turn(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu từ đầu năm'
    query={'sql':'SELECT SUM(line_amount) AS total FROM dm_sales WHERE order_date BETWEEN ? AND ?',
           'parameters':['2025-01-01','2025-01-31'],'columns':['total']}
    result=make_agent(split,tmp_path,scripted_backend([plan(text,query=query)])).run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.answer==400 and result.model_calls==1
    assert result.trace[0]['sql_validation']['scope']=='agreement_with_compiled_plan_and_source_quality'


def test_wrong_sql_and_mutations_cannot_bypass_the_fast_path(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu từ đầu năm'
    query={'sql':'SELECT SUM(base_quantity) FROM dm_sales WHERE order_date BETWEEN ? AND ?',
           'parameters':['2025-01-01','2025-01-31']}
    result=make_agent(split,tmp_path,scripted_backend([plan(text,query=query),{'action':'calculate'}])).run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.answer==400 and result.model_calls==2
    assert result.trace[0]['observation']['error'] and result.trace[0]['execution']['rows']==[[5]]
    result=make_agent(split,tmp_path,scripted_backend([plan(text,query={'sql':'DELETE FROM dm_sales_all'}),plan(text)])).run(Question('y','shop_a',text,'2025-01-31'))
    assert result.answer.answer==400 and result.trace[0]['observation']['error']


def test_custom_inline_sql_and_self_review_remain_available(business,tmp_path,scripted_backend):
    split,_=business;text='Số mặt hàng khác nhau từ đầu năm'
    p=plan(text,'custom','custom',query={'sql':'SELECT COUNT(DISTINCT product_name) FROM dm_sales WHERE order_date BETWEEN ? AND ?',
                                       'parameters':['2025-01-01','2025-01-31']})
    result=make_agent(split,tmp_path,scripted_backend([p])).run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.answer==2 and result.model_calls==1
    text='Doanh thu từ đầu năm'
    result=make_agent(split,tmp_path,scripted_backend([plan(text),{'action':'final','type':'number','query_id':'r1'}]),mode='self_review').run(Question('y','shop_a',text,'2025-01-31'))
    assert result.model_calls==2 and result.answer.answer==400 and result.trace[0]['review_requested']


def test_exploratory_sql_is_returned_to_model_instead_of_becoming_an_answer(business,tmp_path,scripted_backend):
    split,_=business;text='Số mặt hàng khác nhau từ đầu năm'
    actions=[plan(text,'custom','custom'),
             {'action':'sql','sql':'SELECT COUNT(*) FROM dm_sales'},
             {'action':'sql','sql':'SELECT COUNT(DISTINCT product_name) FROM dm_sales WHERE order_date BETWEEN ? AND ?',
              'parameters':['2025-01-01','2025-01-31']},
             {'action':'final','type':'number','query_id':'r2'}]
    result=make_agent(split,tmp_path,scripted_backend(actions)).run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.answer==2 and result.model_calls==4
    assert result.trace[1]['observation']['rows']==[[3]] and not result.trace[1].get('accepted')


def test_targeted_patch_keeps_valid_fields_and_is_cached(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu từ đầu năm'
    path=split.root/'data/shop_a/journal.csv'
    path.write_text(path.read_text(encoding='utf-8').replace('Xong','Chốt'),encoding='utf-8')
    cite='R'+str(next(i for i,line in enumerate(README.splitlines()) if 'Chốt nghĩa' in line))
    patch={'action':'prepare','tables':{'T0':{'status_map':{'Chốt':{'value':'completed','evidence':cite}},
                                           'roles':{'made_up_role':{'column':'T0C0','evidence':'T0C0'}}}}}
    backend=scripted_backend([plan(text),patch])
    generate=backend.generate
    def with_thinking(*args,**kwargs):
        raw=generate(*args,**kwargs)
        return '<think>'+('outer' if len(backend.messages)==1 else 'repair')+'</think>'+raw
    backend.generate=with_thinking
    agent=make_agent(split,tmp_path,backend)
    result=agent.run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.answer==400 and result.model_calls==2
    from datamate.competition_log import competition_fields
    assert competition_fields(result)['thinking'] == ['outer','repair']
    repair=next(e for e in result.trace if e.get('preparation_stage')=='targeted_schema_repair')
    assert repair['observation']['baseline_preserved'] and repair['observation']['changed']
    assert repair['observation']['rejected_fields'] and not repair['observation']['remaining_needs']
    agent.backend=scripted_backend([plan(text)])
    again=agent.run(Question('different','shop_a',text,'2025-01-31'))
    assert again.answer.answer==400 and again.model_calls==1


def test_new_column_mapping_is_bounded_and_does_not_need_a_new_metric(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu từ đầu năm'
    path=split.root/'data/shop_a/journal.csv'
    path.write_text(path.read_text(encoding='utf-8').replace('Cash','XMoney'),encoding='utf-8')
    readme=README.replace('- **Cash**: thành tiền sau giảm','- **XMoney**: Ô này lưu khoản sau ưu đãi của từng mặt hàng.')
    (split.root/'data/shop_a/README.md').write_text(readme,encoding='utf-8')
    cite='R'+str(next(i for i,line in enumerate(readme.splitlines()) if '**XMoney**' in line))
    patch={'action':'prepare','tables':{'T0':{'kind':'sales','roles':{'line_amount':{'column':'T0C5','evidence':cite}}}}}
    class BudgetBackend(scripted_backend):
        supports_token_budget=True
        def generate(self,messages,remaining_seconds,**options):
            if options:
                assert options=={'max_new_tokens':512} and 0<remaining_seconds<=20
            return super().generate(messages,remaining_seconds)
    result=make_agent(split,tmp_path,BudgetBackend([plan(text),patch])).run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.answer==400 and result.model_calls==2
    repair=next(e for e in result.trace if e.get('preparation_stage')=='targeted_schema_repair')
    assert repair['observation']['changed'] and not repair['observation']['remaining_needs']


def test_failed_patch_is_not_retried_and_ambiguous_status_stays_unknown(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu từ đầu năm'
    path=split.root/'data/shop_a/journal.csv'
    path.write_text(path.read_text(encoding='utf-8').replace('Xong','Hoàn'),encoding='utf-8')
    patch={'action':'prepare','tables':{'T0':{'status_map':{'Hoàn':{'value':'completed','evidence':'T0C2'}}}}}
    end={'action':'final','type':'abstain','reason':'Trạng thái chưa rõ nghĩa'}
    result=make_agent(split,tmp_path,scripted_backend([plan(text),patch,end])).run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.type=='abstain' and result.model_calls==3
    # New Agent, same disk cache: no second patch call for the same missing data.
    agent=make_agent(split,tmp_path,scripted_backend([plan(text),end]))
    again=agent.run(Question('y','shop_a',text,'2025-01-31'))
    assert again.answer.type=='abstain' and again.model_calls==2
    repair=next(e for e in again.trace if e.get('preparation_stage')=='targeted_schema_repair')
    assert repair['cache_hit'] and repair['observation']['rejected_fields']


def test_unknown_labels_outside_the_requested_period_cost_no_repair_call(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu từ đầu năm'
    path=split.root/'data/shop_a/journal.csv'
    path.write_text(path.read_text(encoding='utf-8').replace('20/12/2024,Xong','20/12/2024,Hoàn'),encoding='utf-8')
    result=make_agent(split,tmp_path,scripted_backend([plan(text)])).run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.answer==400 and result.model_calls==1
    assert not any(e.get('phase')=='preparation' for e in result.trace)


def test_code_only_never_calls_schema_model_when_mapping_is_missing(business,tmp_path,scripted_backend):
    split,_=business;text='Doanh thu từ đầu năm'
    path=split.root/'data/shop_a/journal.csv'
    path.write_text(path.read_text(encoding='utf-8').replace('Xong','Hoàn'),encoding='utf-8')
    result=make_agent(split,tmp_path,scripted_backend([plan(text),{'action':'final','type':'abstain','reason':'Chưa rõ trạng thái'}]),schema_mode='code_only').run(Question('x','shop_a',text,'2025-01-31'))
    assert result.answer.type=='abstain' and result.model_calls==2
    assert not any(e.get('phase')=='preparation' for e in result.trace)


def test_markdown_citations_and_partial_policy_preserve_baseline():
    assert evidence_in('Tuần bắt đầu thứ Hai.', '| Tuần | bắt đầu **thứ Hai**. |')
    baseline=read_policy(RULES).to_dict()
    accepted,rejected=merge_policy({'week_start':{'value':0,'evidence':'Tuần bắt đầu thứ Hai.'},
                                   'shifts':{'value':{'morning':['25:00','27:00']},'evidence':'Ca sáng'}},RULES,baseline)
    assert accepted['week_start']==0 and accepted['shifts']==baseline['shifts'] and 'shifts' in rejected
    assert not evidence_in('Tuần bắt đầu Chủ nhật.',RULES)


def test_generic_header_conventions_and_readme_tables():
    table={'name':'raw_any','file':'arbitrary.csv','columns':['MaKH','Tên gọi','SLQuyDoi','ThoiGianLap','Tiền X','ĐVT']}
    readme='## Các cột trong `arbitrary.csv`\n### Chi tiết cột\n| Cột | Mô tả |\n| `Tiền X` | Là thành tiền của dòng bán |\n### File `different.csv`\n| `Tiền X` | Số lượng |'
    binding=bind_table(table,readme)
    assert binding['roles']['customer_id']=='MaKH'
    assert binding['roles']['customer_name']=='Tên gọi'
    assert binding['roles']['base_quantity']=='SLQuyDoi'
    assert binding['roles']['timestamp']=='ThoiGianLap'
    assert binding['roles']['line_amount']=='Tiền X'
    assert binding['roles']['unit']=='ĐVT'
    original='Dthu SP-X 12/03/2025 1.200.000đ ck hnay?'
    hints=abbreviation_hints(original)
    assert hints['aliases']['dthu']=='doanh thu' and len(hints['ambiguous_candidates']['ck'])==2
    assert original=='Dthu SP-X 12/03/2025 1.200.000đ ck hnay?'
