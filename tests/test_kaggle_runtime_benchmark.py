import json
from pathlib import Path
import sys
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from tools.prepare_kaggle_benchmark import digest, prepare
from tools.benchmark_runtime import Observer, run, override_loader_dtype
from tools.compare_runtime_benchmarks import compare


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def participant_fixture(root):
    dataset=root/'dataset'
    dataset.mkdir(parents=True)
    (dataset/'rulebook.md').write_text('fixture')
    (dataset/'calendar.csv').write_text('date\n2026-01-01\n')
    questions=[]
    for shop in range(4):
        folder=dataset/'data'/f'shop_{shop}'
        folder.mkdir(parents=True)
        (folder/'README.md').write_text('fixture '*(shop+1))
        for table in range(shop+1):
            (folder/f'table_{table}.csv').write_text('x\n1\n')
        for index in range(9):
            questions.append({'id':f'q{shop}_{index}','business_id':f'shop_{shop}',
                              'question':'Tổng tiền?','as_of_date':'2026-01-01'})
    (dataset/'questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in questions))
    save(root/'benchmark_manifest.json',{'benchmark_id':'fixture',
         'questions_sha256':digest(dataset/'questions.jsonl'),
         'files':{p.relative_to(root).as_posix():digest(p) for p in dataset.rglob('*') if p.is_file()}})
    (dataset/'answers.jsonl').write_text('SHOULD NOT BE READ OR COPIED')
    (dataset/'submission.jsonl').write_text('SHOULD NOT BE READ OR COPIED')
    return root


def test_frozen_subset_is_disjoint_repeatable_and_copies_no_predictions_or_gold(tmp_path):
    source=participant_fixture(tmp_path/'participant')
    before=digest(source/'dataset/questions.jsonl')
    report=prepare(source,tmp_path/'subset-a',size=8,businesses=2,seed=10)
    again=prepare(source,tmp_path/'subset-b',size=8,businesses=2,seed=10)
    assert report==again and digest(source/'dataset/questions.jsonl')==before
    groups=[json.loads((tmp_path/'subset-a'/name).read_text()) for name in ('main_ids.json','pilot_ids.json','reserve_ids.json')]
    assert len(groups[0])==8 and all(not set(a)&set(b) for i,a in enumerate(groups) for b in groups[i+1:])
    assert not (tmp_path/'subset-a/dataset/answers.jsonl').exists()
    assert not (tmp_path/'subset-a/dataset/submission.jsonl').exists()
    assert [x['business_id'] for x in report['businesses']]==['shop_0','shop_3']
    with pytest.raises(ValueError,match='empty'):
        prepare(source,tmp_path/'subset-a',size=8,businesses=2)


def test_source_corruption_is_not_silently_copied(tmp_path):
    source=participant_fixture(tmp_path/'participant')
    (source/'dataset/data/shop_3/table_0.csv').write_text('changed')
    with pytest.raises(ValueError,match='differs'):
        prepare(source,tmp_path/'subset',size=8,businesses=2)


def test_preflight_reaches_real_agent_prompt_without_loading_weights(split_root,tmp_path,monkeypatch):
    from tools.profile_runtime_context import profile
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert kwargs['enable_thinking'] is False
            return {'input_ids':list(range(100))}
    factory=SimpleNamespace(from_pretrained=lambda *a,**kw:Tokenizer())
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=factory))
    config=tmp_path/'config.json'
    save(config,{'backend':'transformers','split_root':str(split_root),'cache_dir':str(tmp_path/'cache'),
                 'model_dir':str(tmp_path/'tokenizer-only')})
    ids=tmp_path/'ids.json'; save(ids,['question_0'])
    report=profile(Path(__file__).resolve().parents[1],config,ids,tmp_path/'preflight.json')
    assert report['ready'] and report['questions'][0]['input_tokens']==100
    assert report['suggested_context_tokens']==16384


def test_observer_records_actual_app_output_and_does_not_replace_agent_loop(split_root,tmp_path,monkeypatch):
    monkeypatch.setattr(Observer,'_sample',lambda self:None)
    config=tmp_path/'config.json'; save(config,{'backend':'demo','split_root':str(split_root),'cache_dir':str(tmp_path/'cache')})
    ids=tmp_path/'ids.json'; save(ids,['question_0','question_1'])
    output=tmp_path/'run'
    result=run(Path(__file__).resolve().parents[1],config,ids,output)
    metrics=json.loads((output/'benchmark_metrics.json').read_text())
    assert result['written']==metrics['written']==2
    assert metrics['valid_complete_run'] and metrics['cached_tokens'] is None
    assert metrics['token_metadata_complete'] is False
    assert len((output/'question_timing.jsonl').read_text(encoding='utf-8').splitlines())==2
    assert len((output/'model_calls.jsonl').read_text().splitlines())==2


def test_dtype_override_changes_only_loader_configuration(monkeypatch):
    expected=object()
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(float16=expected))
    calls=[]
    def loader(target,path,**kwargs):
        calls.append((target,path,kwargs)); return 'loaded'
    backends=SimpleNamespace(load_target=loader)
    original=override_loader_dtype(backends,'float16')
    assert backends.load_target('target','local',dtype='bf16-emulated',device_map='auto')=='loaded'
    assert calls==[('target','local',{'dtype':expected,'device_map':'auto'})]
    assert original is loader


def comparison_fixture(root,backend='transformers'):
    root.mkdir()
    save(root/'benchmark_identity.json',{'ids':['q1'],'model_manifest_sha256':'same',
         'settings':{'backend':backend,'quantization':'none','model_key':'fixture'},'packages':{},'cuda_visible_devices':'0,1'})
    save(root/'benchmark_metrics.json',{'valid_complete_run':True,'inference_seconds':10,'cold_total_seconds':20,
         'token_metadata_complete':True,
         'backend_info':{'model_dtype':'torch.float16','class':'TransformersBackend' if backend=='transformers' else 'VLLMBackend'},
         'gpu_hardware':{'0':{'name':'T4','total_mib':15000}}})
    save(root/'run_identity.json',{'rulebook_sha256':'r','questions_sha256':'q','input_hashes':{'table':'x'}})
    (root/'submission.jsonl').write_text(json.dumps({'id':'q1','type':'number','answer':1})+'\n')
    (root/'model_calls.jsonl').write_text(json.dumps({'id':'q1','start_seconds':1,'messages_sha256':'same'})+'\n')
    return root


def test_comparison_does_not_invent_accuracy_and_invalidates_different_prompts(tmp_path):
    left=comparison_fixture(tmp_path/'left'); right=comparison_fixture(tmp_path/'right','vllm')
    result=compare(left,right)
    assert result['paired_comparison_valid'] and result['inference_speedup']==1
    assert result['scores_baseline_candidate'] is None and result['quality_not_worse_on_this_subset'] is None
    (right/'model_calls.jsonl').write_text(json.dumps({'id':'q1','start_seconds':1,'messages_sha256':'changed'})+'\n')
    result=compare(left,right)
    assert not result['paired_comparison_valid'] and result['inference_speedup'] is None


def test_comparison_rejects_different_input_tables_even_when_prompts_match(tmp_path):
    left=comparison_fixture(tmp_path/'left'); right=comparison_fixture(tmp_path/'right','vllm')
    save(right/'run_identity.json',{'rulebook_sha256':'r','questions_sha256':'q','input_hashes':{'table':'changed'}})
    result=compare(left,right)
    assert result['gates']['same_initial_requests'] and not result['gates']['same_dataset_hashes']
    assert result['inference_speedup'] is None


def test_comparison_rejects_accidental_bfloat16_baseline(tmp_path):
    left=comparison_fixture(tmp_path/'left'); right=comparison_fixture(tmp_path/'right','vllm')
    metrics=json.loads((left/'benchmark_metrics.json').read_text())
    metrics['backend_info']['model_dtype']='torch.bfloat16'
    save(left/'benchmark_metrics.json',metrics)
    result=compare(left,right)
    assert not result['gates']['same_explicit_float16_precision'] and result['inference_speedup'] is None


def test_observer_separates_time_before_generation_without_changing_the_call(tmp_path,monkeypatch):
    from tools import benchmark_runtime as runtime
    clock=[100.0]
    monkeypatch.setattr(runtime,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    observer=Observer(tmp_path/'observed')
    class Backend:
        last_generation=None
        def generate(self,messages,remaining_seconds):
            assert remaining_seconds==83
            clock[0]+=3
            self.last_generation={'input_tokens':10,'output_tokens':2,'stop_reason':'eos'}
            return '{"ok":true}'
    backend=observer.wrap_backend(Backend())
    def original(agent,question):
        clock[0]+=7  # Data/catalog preparation, outside generate.
        assert backend.generate([{'role':'user','content':'same input'}],83)=='{"ok":true}'
        clock[0]+=2  # Tools and other agent work.
        return SimpleNamespace(model_calls=1,answer=SimpleNamespace(type='number',reason=None))
    agent=SimpleNamespace(settings=SimpleNamespace(question_timeout_seconds=90))
    observer.wrap_agent_run(original)(agent,SimpleNamespace(id='q1',business_id='shop'))
    row=observer.questions[0]
    assert row['duration_seconds']==12
    assert row['first_model_delay_seconds']==7 and row['generation_wall_seconds']==3 and row['other_wall_seconds']==9
    assert observer.calls[0]['remaining_seconds_at_call']==83
    assert observer.calls[0]['output_sha256'] is not None
    report=observer.summarize(12,['q1'],{'written':1},None)
    assert report['execution_complete'] and report['valid_complete_run']


def test_completed_deadline_run_is_reportable_but_not_clean_speedup(tmp_path):
    left=comparison_fixture(tmp_path/'left');right=comparison_fixture(tmp_path/'right','vllm')
    metrics=json.loads((right/'benchmark_metrics.json').read_text())
    metrics.update(valid_complete_run=False,execution_complete=True,written=1,timeout_calls=1)
    save(right/'benchmark_metrics.json',metrics)
    (right/'submission.jsonl').write_text(json.dumps({'id':'q1','type':'abstain','reason':'deadline'})+'\n')
    (right/'question_timing.jsonl').write_text(json.dumps({'id':'q1','start_seconds':0,'duration_seconds':90})+'\n')
    (right/'model_calls.jsonl').write_text(json.dumps({'id':'q1','start_seconds':75,'duration_seconds':15,
        'messages_sha256':'same','stop_reason':'timeout_aborted','error':'GenerationTimeoutError: deadline'})+'\n')
    result=compare(left,right)
    assert result['execution_complete_both'] and result['identity_compatible']
    assert not result['paired_comparison_valid'] and result['inference_speedup'] is None
    assert result['observed_inference_time_ratio']==1 and result['engine_only_speedup'] is None
    assert result['outcome_counts_baseline_candidate'][1]=={'deadline_limited':1}
    row=result['per_question'][0]
    assert row['difference_kind']=='deadline_affected_difference'
    assert row['candidate']['first_model_delay_seconds']==75
    assert result['answered_only_baseline_ids']==['q1']


def test_same_prompt_different_model_action_is_distinct_from_deadline(tmp_path):
    left=comparison_fixture(tmp_path/'left');right=comparison_fixture(tmp_path/'right','vllm')
    (right/'submission.jsonl').write_text(json.dumps({'id':'q1','type':'number','answer':2})+'\n')
    for folder,value in ((left,1),(right,2)):
        (folder/'logs').mkdir()
        save(folder/'logs/q1.json',{'id':'q1','trace':[{'generation':{'stop_reason':'eos'},
               'llm_action':{'action':'plan','filters':[{'value':value}]}}]})
    result=compare(left,right)
    row=result['per_question'][0]
    assert row['difference_kind']=='different_non_abstain_answers'
    assert row['first_action_difference']['generation_index']==1
    assert row['first_action_difference']['same_request_messages']
    assert result['scores_baseline_candidate'] is None and result['quality_not_worse_on_this_subset'] is None


def test_tool_elapsed_time_changes_are_detected_without_rewriting_the_prompt(tmp_path):
    left=comparison_fixture(tmp_path/'left');right=comparison_fixture(tmp_path/'right','vllm')
    (right/'submission.jsonl').write_text(json.dumps({'id':'q1','type':'number','answer':2})+'\n')
    for folder,value in ((left,0.001),(right,0.02)):
        (folder/'logs').mkdir()
        save(folder/'logs/q1.json',{'id':'q1','trace':[{'generation':{'stop_reason':'eos'},
             'llm_action':{'action':'plan'},'model_observation':{'rows':[[1]],'elapsed_seconds':value}}]})
    before=(left/'logs/q1.json').read_bytes(),(right/'logs/q1.json').read_bytes()
    result=compare(left,right)
    feedback=result['per_question'][0]['first_feedback_difference']
    assert feedback=={'after_generation_index':1,'paths':['/elapsed_seconds'],'only_elapsed_seconds':True}
    assert before==((left/'logs/q1.json').read_bytes(),(right/'logs/q1.json').read_bytes())


def test_infrastructure_error_does_not_become_a_timeout_or_observed_speedup(tmp_path):
    left=comparison_fixture(tmp_path/'left');right=comparison_fixture(tmp_path/'right','vllm')
    metrics=json.loads((right/'benchmark_metrics.json').read_text())
    metrics.update(valid_complete_run=False,error='BackendUnavailableError: engine died')
    save(right/'benchmark_metrics.json',metrics)
    (right/'model_calls.jsonl').write_text(json.dumps({'id':'q1','start_seconds':1,'messages_sha256':'same',
        'error':'BackendUnavailableError: engine died'})+'\n')
    result=compare(left,right)
    assert not result['execution_complete_both'] and not result['gates']['no_infrastructure_errors']
    assert result['observed_inference_time_ratio'] is None and result['inference_speedup'] is None


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
def test_invalid_times_never_produce_a_speedup(tmp_path,value):
    left=comparison_fixture(tmp_path/'left');right=comparison_fixture(tmp_path/'right','vllm')
    metrics=json.loads((right/'benchmark_metrics.json').read_text())
    metrics['inference_seconds']=value
    save(right/'benchmark_metrics.json',metrics)
    result=compare(left,right)
    assert not result['gates']['finite_positive_times'] and result['inference_speedup'] is None


def test_observer_keeps_completion_separate_from_deadline_quality(tmp_path):
    observer=Observer(tmp_path/'observed')
    observer.questions=[{'id':'q1','duration_seconds':90,'error':None,'answer_type':'abstain'}]
    observer.calls=[{'id':'q1','error':'GenerationTimeoutError: deadline','stop_reason':'timeout_aborted',
                     'input_tokens':100,'output_tokens':0,'cached_tokens':None}]
    report=observer.summarize(90,['q1'],{'written':1},None)
    assert report['execution_complete'] and not report['valid_complete_run']
    assert report['timeout_calls']==1 and 'generation_timeouts' in report['measurement_warnings']


def test_kaggle_notebook_matches_copyable_guide_and_has_no_gpu_outputs():
    from tools.create_kaggle_benchmark_notebook import build
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/DataMate_Kaggle_Runtime_Benchmark.ipynb').read_text(encoding='utf-8'))
    assert notebook==build(root)
    code=[c for c in notebook['cells'] if c['cell_type']=='code']
    assert len(code)==2  # Only settings and the self-contained stage; Save & Run All needs no interactive state.
    for cell in code:
        compile(''.join(cell['source']),cell['id'],'exec')
        assert cell['outputs']==[] and cell['execution_count'] is None


def test_observer_can_run_the_frozen_old_application_unchanged(split_root,tmp_path):
    root=Path(__file__).resolve().parents[1]
    frozen=root/'outputs/packages/kaggle-baseline-source.zip'
    if not frozen.exists():
        pytest.skip('Local frozen baseline source kit is not stored in Git')
    baseline=tmp_path/'baseline'
    with zipfile.ZipFile(frozen) as archive:
        archive.extractall(baseline)
    config=tmp_path/'old-config.json'
    save(config,{'backend':'demo','split_root':str(split_root),'cache_dir':str(tmp_path/'old-cache')})
    ids=tmp_path/'ids.json'; save(ids,['question_0'])
    output=tmp_path/'old-output'
    result=subprocess.run([sys.executable,str(root/'tools/benchmark_runtime.py'),'--project-root',str(baseline),
                           '--config',str(config),'--ids',str(ids),'--output',str(output)],
                          capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode==0,result.stderr
    identity=json.loads((output/'benchmark_identity.json').read_text(encoding='utf-8'))
    metrics=json.loads((output/'benchmark_metrics.json').read_text(encoding='utf-8'))
    assert identity['source_root']==str(baseline.resolve()) and metrics['valid_complete_run']
    assert 'vllm' not in identity['settings']  # Proves the old Settings really was loaded.
