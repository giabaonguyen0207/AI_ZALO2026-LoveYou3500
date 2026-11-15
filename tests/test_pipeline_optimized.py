"""CPU-only regression fixtures for the integrated pipeline; no model imports."""
from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from datamate.config import RAGSettings, Settings
from datamate.domain import DataMateError, Question
from datamate.models.registry import inventory_from_settings, model_paths, relocate_models
from datamate.retrieval.correction import VietnameseCorrector, validate_variant
from datamate.retrieval.hybrid import rank_candidates
from datamate.retrieval.contracts import RetrievalFatalError


@pytest.mark.parametrize('original,candidate',[
    ('dthu thang 3 la bn','doanh thu tháng 3 là bao nhiêu'),
    ('sp ban dc nhieu nhat thang 3','sản phẩm bán được nhiều nhất tháng 3'),
    ('sp ko ban thang 3','sản phẩm không bán tháng 3'),
])
def test_correction_accepts_conservative_diacritic_and_abbreviation_variants(original,candidate):
    assert validate_variant(original,candidate) is None


@pytest.mark.parametrize('original,candidate',[
    ('dthu thang 3 la bn','doanh thu tháng 4 là bao nhiêu'),
    ('dthu ngay 01/02/2025','doanh thu ngày 02/01/2025'),
    ('sp ko ban thang 3','sản phẩm bán tháng 3'),
    ('dthu thang 3','doanh thu năm 3'),
    ('doanh thu','lợi nhuận'),
    ('sl 3 kg','số lượng 3 hộp'),
    ('sl "Sữa X"','số lượng "Sữa Y"'),
    ('dthu cua Lan','doanh thu của Lâm'),
    ('sl SP01','số lượng SP02'),
    ('dthu > 100','doanh thu < 100'),
])
def test_correction_rejects_changed_business_conditions(original,candidate):
    assert validate_variant(original,candidate)


def fake_corrector(monkeypatch, response, reason=None):
    import datamate.retrieval.correction as module
    monkeypatch.setattr(module,'verify_artifact',lambda *args:{'fixture':True})
    obj=VietnameseCorrector(RAGSettings(correction_enabled=True,mode='dense'))
    monkeypatch.setattr(obj,'load',lambda:None)
    monkeypatch.setattr(obj,'_generate',lambda *args:(response,reason))
    return obj


def test_correction_caches_by_text_and_does_not_invent_a_second_model_call(monkeypatch):
    obj=fake_corrector(monkeypatch,'doanh thu tháng 3 là bao nhiêu')
    variant,event=obj.correct('dthu thang 3 la bn',5)
    assert variant and event['model_call_attempted']
    assert obj.correct('dthu thang 3 la bn',5)[1]['status']=='cache_hit'
    assert obj.calls==1 and obj.cache_hits==1
    assert obj.correct('dthu thang 3 la bn',0)[0] is None
    obj.close()
    assert not obj.cache


def test_correction_timeout_and_rejected_variants_keep_original(monkeypatch):
    obj=fake_corrector(monkeypatch,'doanh thu tháng 4')
    assert obj.correct('dthu thang 3',5)[0] is None
    obj=fake_corrector(monkeypatch,None,'timeout')
    assert obj.correct('dthu thang 3',5)[1]['status']=='timeout'
    assert not obj.cache  # A slow call does not poison the next run.


def test_correction_cuda_failure_is_fatal_not_silently_ignored(monkeypatch):
    obj=fake_corrector(monkeypatch,None)
    def fail(*args): raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(obj,'_generate',fail)
    with pytest.raises(RetrievalFatalError,match='GPU failure'):obj.correct('dthu',5)


def test_hybrid_lexical_term_can_recover_a_schema_missed_by_dense_rank():
    chunks=[SimpleNamespace(text='Quy tắc doanh thu chung',metadata={'chunk_id':'a'}),
            SimpleNamespace(text='Schema MA_KH thanh toán công nợ',metadata={'chunk_id':'b'}),
            SimpleNamespace(text='Lịch âm dương',metadata={'chunk_id':'c'})]
    assert rank_candidates(chunks,['MA_KH'],[0.9,0.1,0.2],hybrid=False,limit=1)==[0]
    assert rank_candidates(chunks,['MA_KH'],[0.9,0.1,0.2],hybrid=True,limit=1)==[1]
    assert rank_candidates(chunks,['nonexistent'],[0.9,0.1,0.2],hybrid=True,limit=3)==[0,2,1]


def test_all_four_models_are_counted_and_relocated_without_changing_interface():
    root=Path(__file__).resolve().parents[1]
    settings=Settings.load(root/'configs/submission-optimized.json',use_env=False)
    total=inventory_from_settings(settings)
    assert total['total_parameters']==11_188_614_897
    assert len(model_paths(settings))==4
    paths={key:'/new/'+key for key in model_paths(settings)}
    relocate_models(settings,paths)
    assert settings.rag.correction_dir=='/new/vietnamese-error-correction'
    assert settings.split_root=='/data' and settings.result_dir=='/result'
    assert settings.effective_workers==settings.vllm.max_num_seqs==2
    assert settings.vllm.enable_prefix_caching and settings.vllm.cache_prompt_prefix
    assert settings.quantization=='none' and settings.dtype=='float16'


@pytest.mark.parametrize('patch',[{'correction_enabled':'true'}, {'hybrid_retrieval':1},
    {'correction_max_tokens':1024}, {'correction_timeout_seconds':float('nan')}])
def test_bad_correction_configuration_is_rejected(patch):
    with pytest.raises(DataMateError):RAGSettings(mode='dense',**patch).validate()


def test_one_recovery_can_make_progress_but_never_bypasses_answer_validation(split_root,tmp_path,scripted_backend):
    from datamate.agent.runner import Agent
    bad={'action':'sql','sql':'SELECT nonexistent_column FROM calendar'}
    backend=scripted_backend([bad,bad,{'action':'sql','sql':'SELECT 42 AS value'},
                              {'action':'final','type':'number','query_id':'r1'}])
    settings=Settings(split_root=str(split_root),cache_dir=str(tmp_path/'recover'),backend='transformers',
                      require_plan=False,max_steps=2,recovery_steps=2)
    backend.counts_as_model=True
    # This fixture exercises projection of a read-only result with no sales metric.
    result=Agent(settings,backend).run(Question('q','shop_a','Giá trị thử nghiệm?', '2026-06-02'))
    assert result.answer.answer==42
    assert sum(bool(e.get('bounded_recovery')) for e in result.trace)==1
    assert len(backend.messages)==4
    assert json.loads(backend.messages[2][-1]['content'])['tool_result']['recovery']


def test_repeated_failures_after_single_recovery_still_abstain(split_root,tmp_path,scripted_backend):
    from datamate.agent.runner import Agent
    bad={'action':'sql','sql':'SELECT nonexistent_column FROM calendar'}
    backend=scripted_backend([bad]*8)
    settings=Settings(split_root=str(split_root),cache_dir=str(tmp_path/'stop'),backend='transformers',
                      require_plan=False,max_steps=6,recovery_steps=2)
    result=Agent(settings,backend).run(Question('q','shop_a','Tổng?', '2026-06-02'))
    assert result.answer.type=='abstain' and len(backend.messages)==4
    assert sum(bool(e.get('bounded_recovery')) for e in result.trace)==1


def test_agent_keeps_original_evidence_and_supplies_correction_only_as_hint(split_root,tmp_path,scripted_backend):
    from datamate.agent.runner import Agent
    original='dthu thang 3 la bn'
    variant='doanh thu tháng 3 là bao nhiêu'
    class Retriever:
        def correct_question(self,text,remaining):
            assert text==original
            return variant,dict(phase='question_correction',model_call_attempted=True,raw=variant)
        def retrieve_variants(self,text,hint,business,catalog,remaining):
            assert text==original and hint==variant and business=='shop_a'
            return [],dict(status='ok')
    backend=scripted_backend([{'action':'final','type':'abstain'}])
    settings=Settings(split_root=str(split_root),cache_dir=str(tmp_path/'hint'),backend='transformers',
        rag=RAGSettings(mode='dense',correction_enabled=True))
    result=Agent(settings,backend,retriever=Retriever()).run(Question('q','shop_a',original,'2026-06-02'))
    payload=json.loads(result.initial_messages[1]['content'])
    assert payload['request']['question']==original
    assert payload['request_contract']['question_normalization_hint']['text']==variant
    assert result.model_calls==2


def test_pretrained_correction_is_verified_separately_from_team_training(monkeypatch,tmp_path):
    import tools.package_solution as package
    calls=[]
    monkeypatch.setattr(package,'verify_artifact',lambda path,key:calls.append((path,key)))
    package.verify_training_link({'inference_artifacts':{}},'vietnamese-error-correction',tmp_path)
    assert calls
    with pytest.raises((ValueError,FileNotFoundError)):
        package.verify_training_link({'inference_artifacts':{}},'qwen35-9b',tmp_path)


def test_retrieval_uses_original_and_variant_then_reranks_against_original(split_root,tmp_path,monkeypatch):
    import datamate.retrieval.core as core
    from datamate.data.split import DatasetSplit
    class Matrix:
        def __init__(self,rows):self.rows=rows
        def __getitem__(self,i):return self.rows[i]
        @property
        def T(self):return Matrix(list(zip(*self.rows)))
        def __matmul__(self,other):
            if isinstance(other,Matrix):
                columns=list(zip(*other.rows))
                return Matrix([[sum(a*b for a,b in zip(row,col)) for col in columns] for row in self.rows])
            return [sum(a*b for a,b in zip(row,other)) for row in self.rows]
        def max(self,axis):
            assert axis==1
            return [max(row) for row in self.rows]
    encoded=[];reranked=[]
    class Encoder:
        def encode(self,texts,remaining_seconds):
            encoded.extend(texts)
            return Matrix([[1,0],[0,1]][:len(texts)])
        def score(self,query,passages,remaining_seconds):
            reranked.append(query)
            return list(range(len(passages),0,-1))
        def close(self):pass
    monkeypatch.setattr(core,'verify_artifact',lambda *a:{'fixture':True})
    monkeypatch.setattr(core,'validate_vectors',lambda values,expected_rows:values)
    settings=RAGSettings(mode='dense_rerank',hybrid_retrieval=True,context_k=2)
    obj=core.ContextRetriever(settings,DatasetSplit(split_root),cache_dir=tmp_path/'retrieval',encoder=Encoder())
    chunks=[core.Chunk('schema dthu','source1','shop_a','digest1',metadata={'chunk_id':'one'}),
            core.Chunk('doanh thu','source2','shop_a','digest2',metadata={'chunk_id':'two'})]
    monkeypatch.setattr(obj,'_index',lambda *args:(chunks,Matrix([[1,0],[0,1]])))
    selected,info=obj.retrieve_variants('dthu','doanh thu','shop_a',SimpleNamespace(fingerprint='input'),5)
    assert encoded==['dthu','doanh thu'] and reranked==['dthu']
    assert len({s['chunk_id'] for s in selected})==2
    assert info['query_variants']==encoded and info['hybrid_retrieval']
    obj.close()


def test_staged_correction_records_actual_tensors_but_caps_inventory_conservatively(tmp_path,monkeypatch):
    import struct
    import datamate.models.artifacts as artifacts
    from datamate.models.registry import get_model
    path=tmp_path/'correction';path.mkdir()
    header=json.dumps({'weights':{'shape':[3],'dtype':'F16','data_offsets':[0,6]}}).encode()
    (path/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+b'\0'*6)
    # This fixture tests counting independently; production always checks the pinned SHA256.
    monkeypatch.setattr(artifacts,'verify_checkpoint_weight_hash',lambda *a,**kw:None)
    manifest=artifacts.seal_artifact(path,get_model('vietnamese-error-correction'))
    assert manifest['tensor_parameters']==3 and manifest['parameter_upper_bound']==400_000_000
    artifacts.verify_artifact(path,'vietnamese-error-correction')
    manifest['tensor_parameters']=4
    (path/'artifact_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(DataMateError,match='tensor count'):
        artifacts.verify_artifact(path,'vietnamese-error-correction')


@pytest.mark.parametrize('length,ids,expected',[(20,[2,8,2],None),(300,[2,8,2],'input_too_long'),
                                            (20,[2,8,7],'incomplete_output')])
def test_seq2seq_generation_rejects_overlong_input_and_incomplete_output(monkeypatch,length,ids,expected):
    import contextlib
    import sys
    obj=fake_corrector(monkeypatch,None)
    # Exercise the actual adapter using small fake tensors, without importing torch/transformers.
    monkeypatch.delattr(obj,'_generate')
    class Batch(dict):
        def to(self,device):return self
    class Tokenizer:
        def __call__(self,*a,**kw):return Batch(input_ids=SimpleNamespace(shape=(1,length)))
        def decode(self,*a,**kw):return 'doanh thu'
    class Model:
        generation_config=SimpleNamespace(eos_token_id=2)
        def generate(self,**kw):
            assert kw['do_sample'] is False and kw['num_beams']==1
            return [SimpleNamespace(tolist=lambda:ids)]
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(StoppingCriteria=object,StoppingCriteriaList=list))
    obj.tokenizer=Tokenizer();obj.model=Model();obj.torch=SimpleNamespace(inference_mode=contextlib.nullcontext)
    candidate,reason=obj._generate('dthu',time.monotonic()+10)
    assert reason==expected
    assert candidate==('doanh thu' if expected is None else None)
