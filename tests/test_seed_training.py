import csv
import hashlib
import json
from pathlib import Path
import types
import zipfile

import pytest

from datamate.data.split import DatasetSplit,read_jsonl
from datamate.data.sql import SQLExecutor
from datamate.domain import DataMateError,json_text
from datamate.models.artifacts import sha256
from datamate.training.colab import gpu_preflight,make_seed_training_config,unpack_seed_archive
from datamate.training.dataset import business_holdout,source_manifest
from datamate.training.finetune import tokenize_conversations
from datamate.training.seed import _digest,inline_generated_ctes,load_seed_sources,prepare_seed,verify_seed_dataset
from test_pipeline_v3 import RULES


class CharacterTokenizer:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False,**kwargs):
        text=''.join('<'+m['role']+'>'+m['content']+'</'+m['role']+'>' for m in messages)
        return text+('<assistant>' if add_generation_prompt else '')
    def __call__(self,text,**kwargs): return {'input_ids':[ord(char) for char in text]}


@pytest.fixture
def seed_source(split_root,tmp_path):
    records=[];references=[]
    for index,business in enumerate(('shop_a','shop_b','shop_c'),1):
        folder=split_root/'data'/business
        with (folder/'sales.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.writer(handle)
            writer.writerow(['Mã đơn hàng','Ngày bán','Trạng thái','Tên hàng','Số lượng','Thành tiền','Đơn vị tính'])
            writer.writerow(['order-'+str(index),'2025-01-02','Xong','Trà',1,index*1000,'ly'])
        q={'id':'question_'+str(index),'business_id':business,'question':'Doanh thu từ đầu năm tới hôm nay?',
           'as_of_date':'2025-01-31'}
        records.append(q);references.append({'id':q['id'],'type':'number','answer':index*1000})
    (split_root/'rulebook.md').write_text(RULES,encoding='utf-8')
    for name,rows in [('questions',records),('answers',references)]:
        (split_root/(name+'.jsonl')).write_text(''.join(json_text(r)+'\n' for r in rows),encoding='utf-8')
    approved={**source_manifest(split_root),'role':'competition_training'}
    sources=tmp_path/'sources.json';sources.write_text(json_text({'sources':[approved]}),encoding='utf-8')
    split=DatasetSplit(split_root)
    training,validation=business_holdout(split.questions(),.2,42)
    run=tmp_path/'legacy-run';(run/'logs').mkdir(parents=True)
    (run/'selected_questions.jsonl').write_text(''.join(json_text(r)+'\n' for r in records),encoding='utf-8')
    identity={'source_fingerprint':approved['fingerprint'],'question_ids':[r['id'] for r in records]}
    manifest={**identity,'identity_hash':hashlib.sha256(json_text(identity,sort_keys=True).encode()).hexdigest()}
    (run/'run_manifest.json').write_text(json_text(manifest),encoding='utf-8')
    for q,answer in zip(records,references):
        # The legacy log may omit prompt/hash; its bundle is checked instead.
        row={**q,'submission':answer,'trace':[{'raw':json_text({'action':'final','type':'number','query_id':'r1'}),'accepted':True}]}
        (run/'logs'/(q['id']+'.json')).write_text(json_text(row),encoding='utf-8')
    reviews={'reviews':[{'question_id':q.id,'question_sha256':_digest(q.prompt_dict()),
                         'recipe':{'task':'revenue','periods':[{'kind':'year_to_date'}]}}
                        for q in split.questions() if q.business_id in training]}
    review_path=tmp_path/'reviews.json';review_path.write_text(json_text(reviews),encoding='utf-8')
    return types.SimpleNamespace(split=split,run=run,sources=sources,reviews=review_path,
                                 approved=approved,training=training,validation=validation)


def build_fixture(source,output,monkeypatch):
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *args,**kwargs:CharacterTokenizer())
    return prepare_seed(source.split.root,output,reviews=source.reviews,runs=[source.run],
                        sources=source.sources,cache_dir=output.parent/'catalogs',
                        tokenizer_dirs={'gemma4-12b':'fixture'},max_length=65536)


def test_seed_replays_before_reference_scoring_and_keeps_holdout(seed_source,tmp_path,monkeypatch):
    import datamate.training.seed as seed
    output=tmp_path/'prepared';original=seed.read_jsonl;decoded={'answers':False}
    def read(path):
        if Path(path).name=='answers.jsonl':
            assert (output/'execution_audit.jsonl').is_file()
            decoded['answers']=True
        return original(path)
    compiler=seed.compile_recipe
    def compile_without_answers(*args,**kwargs):
        assert not decoded['answers']
        return compiler(*args,**kwargs)
    monkeypatch.setattr(seed,'read_jsonl',read)
    monkeypatch.setattr(seed,'compile_recipe',compile_without_answers)
    report=build_fixture(seed_source,output,monkeypatch)
    manifest=verify_seed_dataset(output)
    assert decoded['answers'] and report['train_examples']==2 and report['validation_seed_questions']==1
    assert report['actual_model_inference_calls']==0 and not report['gpu_training_performed']
    assert set(manifest['train_businesses']).isdisjoint(manifest['validation_businesses'])
    for row in read_jsonl(output/'train.jsonl'):
        assert row['business_id'] in seed_source.training
        targets=[json.loads(m['content']) for m in row['messages'] if m['role']=='assistant']
        assert len(targets)==1 and targets[0]['action']=='plan' and 'answer' not in targets[0]
    assert not (output/'validation.jsonl').read_text()


def test_legacy_bundle_tampering_and_wrong_source_are_rejected(seed_source):
    s=seed_source
    wrong={**s.approved,'fingerprint':'not-the-training-source'}
    with pytest.raises(DataMateError,match='TRAIN'): load_seed_sources(s.split,[s.run],wrong)
    path=next((s.run/'logs').glob('*.json'));row=json.loads(path.read_text(encoding='utf-8'))
    row['as_of_date']='2024-01-01';path.write_text(json_text(row),encoding='utf-8')
    with pytest.raises(DataMateError,match='TRAIN gốc'): load_seed_sources(s.split,[s.run],s.approved)


def test_seed_does_not_accept_annotations_for_validation(seed_source,tmp_path,monkeypatch):
    rows=json.loads(seed_source.reviews.read_text(encoding='utf-8'))
    q=next(q for q in seed_source.split.questions() if q.business_id in seed_source.validation)
    rows['reviews'].append({'question_id':q.id,'question_sha256':_digest(q.prompt_dict()),'recipe':{'task':'revenue'}})
    seed_source.reviews.write_text(json_text(rows),encoding='utf-8')
    with pytest.raises(DataMateError,match='phía train'):
        build_fixture(seed_source,tmp_path/'rejected',monkeypatch)
    assert not (tmp_path/'rejected').exists()


def test_pending_annotation_stays_out_of_training(seed_source,tmp_path,monkeypatch):
    rows=json.loads(seed_source.reviews.read_text(encoding='utf-8'))
    rows['reviews'][0]['recipe']={'task':'pending','category':'ledger_review','note':'Chưa đủ căn cứ'}
    seed_source.reviews.write_text(json_text(rows),encoding='utf-8')
    output=tmp_path/'prepared';report=build_fixture(seed_source,output,monkeypatch)
    assert report['train_examples']==1 and report['review_queue']==1
    assert 'ledger_review' in read_jsonl(output/'repair_queue.jsonl')[0]['reason']


def test_verified_seed_rejects_hash_change_and_validation_rows(seed_source,tmp_path,monkeypatch):
    output=tmp_path/'prepared';build_fixture(seed_source,output,monkeypatch)
    path=output/'train.jsonl';rows=read_jsonl(path)
    rows[0]['business_id']=next(iter(seed_source.validation))
    path.write_text(''.join(json_text(r)+'\n' for r in rows),encoding='utf-8')
    with pytest.raises(DataMateError,match='Hash'): verify_seed_dataset(output)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    manifest['files']['train.jsonl']=sha256(path)
    (output/'manifest.json').write_text(json_text(manifest),encoding='utf-8')
    with pytest.raises(DataMateError,match='validation'): verify_seed_dataset(output)


def test_loss_mask_does_not_supervise_a_failed_assistant_turn():
    messages=[{'role':'system','content':'rules'},{'role':'user','content':'question'},
              {'role':'assistant','content':'BAD_ACTION'},{'role':'user','content':'tool error'},
              {'role':'assistant','content':'GOOD_REPAIR'}]
    rows,skipped=tokenize_conversations([{'messages':messages,'trainable_assistant_indices':[4]}],CharacterTokenizer(),1000)
    assert not skipped and len(rows)==1
    target=''.join(chr(token) for token in rows[0]['labels'] if token!=-100)
    prefix=''.join(chr(token) for token,label in zip(rows[0]['input_ids'],rows[0]['labels']) if label==-100)
    assert target=='GOOD_REPAIR</assistant>' and 'BAD_ACTION' in prefix and 'BAD_ACTION' not in target
    with pytest.raises(DataMateError,match='trainable_assistant_indices'):
        tokenize_conversations([{'messages':messages,'trainable_assistant_indices':[3]}],CharacterTokenizer(),1000)


def test_overlength_target_is_reported_without_truncation():
    rows,skipped=tokenize_conversations([{'messages':[{'role':'user','content':'long'*50},
        {'role':'assistant','content':'correct_action'}]}],CharacterTokenizer(),20)
    assert rows==[] and skipped==1


@pytest.mark.parametrize('sql,parameters,expected',[
    ('WITH p AS (SELECT ? AS x), d AS (SELECT x-? AS y FROM p) SELECT y FROM d',[11,3],[[8]]),
    ("WITH p AS (SELECT '(' AS s,? AS x) SELECT x FROM p",[9],[[9]]),
    ("WITH p AS (SELECT '?' AS s,? AS x) SELECT x FROM p",[9],[[9]]),
])
def test_cte_normalization_keeps_bindings_and_quoted_parentheses(tmp_path,sql,parameters,expected):
    import sqlite3
    path=tmp_path/'db.sqlite';sqlite3.connect(path).close();executor=SQLExecutor(path)
    converted=inline_generated_ctes(sql)
    assert executor.execute(sql,parameters)['rows']==expected
    assert executor.execute(converted,parameters)['rows']==expected
    assert not converted.startswith('WITH ')


def archive_fixture(dataset,archive,lock):
    manifest=verify_seed_dataset(dataset)
    with zipfile.ZipFile(archive,'w') as handle:
        for name in sorted(set(manifest['files'])|{'manifest.json'}): handle.write(dataset/name,name)
    lock.write_text(json_text({'archive_sha256':sha256(archive),
                              'dataset_manifest_sha256':sha256(dataset/'manifest.json')}),encoding='utf-8')


def test_colab_archive_reuses_verified_data_and_blocks_wrong_upload(seed_source,tmp_path,monkeypatch):
    dataset=tmp_path/'prepared';build_fixture(seed_source,dataset,monkeypatch)
    archive,lock=tmp_path/'seed.zip',tmp_path/'lock.json';archive_fixture(dataset,archive,lock)
    target=tmp_path/'colab-data'
    assert unpack_seed_archive(archive,target,lock,seed_source.sources)==target
    assert unpack_seed_archive(archive,target,lock,seed_source.sources)==target
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    manifest['runtime_source_hashes']['agent/runner.py']='0'*64
    (dataset/'manifest.json').write_text(json_text(manifest),encoding='utf-8')
    archive_fixture(dataset,archive,lock)
    with pytest.raises(DataMateError,match='Source ZIP khác'):
        unpack_seed_archive(archive,target,lock,seed_source.sources)
    archive.write_bytes(archive.read_bytes()+b'changed')
    with pytest.raises(DataMateError,match='ZIP khác'): unpack_seed_archive(archive,target,lock,seed_source.sources)


def test_colab_configs_smoke_longest_and_report_actual_gpu(seed_source,tmp_path,monkeypatch):
    dataset=tmp_path/'prepared';build_fixture(seed_source,dataset,monkeypatch)
    smoke=make_seed_training_config('gemma4-12b','model',dataset,'out',smoke=True)
    pilot=make_seed_training_config('gemma4-12b','model',dataset,'out',max_steps=5)
    assert smoke['smoke_longest_only'] and smoke['max_steps']==1 and smoke['gradient_accumulation_steps']==1
    assert pilot['max_steps']==5 and pilot['gradient_accumulation_steps']==8 and pilot['reject_overlength']
    assert pilot['method']=='qlora' and pilot['max_length']==65536
    with pytest.raises(DataMateError,match='chỉ fine-tune Gemma'):
        make_seed_training_config('qwen35-9b','model',dataset,'out')
    cuda=types.SimpleNamespace(is_available=lambda:True,get_device_properties=lambda _:types.SimpleNamespace(total_memory=24*1024**3),
                               get_device_name=lambda _:'fixture GPU',is_bf16_supported=lambda:True)
    torch=types.SimpleNamespace(cuda=cuda,__version__='fixture',version=types.SimpleNamespace(cuda='fixture'))
    report=gpu_preflight(torch)
    assert report['gpu']=='fixture GPU' and report['total_vram_gib']==24 and report['memory_fit_verified'] is False
    cuda.get_device_properties=lambda _:types.SimpleNamespace(total_memory=16*1024**3)
    with pytest.raises(DataMateError,match='dưới ngưỡng'): gpu_preflight(torch)


def test_seed_notebook_compiles_and_has_small_pilot_defaults():
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/DataMate_Colab_Seed72_100_QLoRA.ipynb').read_text(encoding='utf-8'))
    import ast
    setup=next(c for c in notebook['cells'] if c.get('id')=='seed-setup')
    desired={'MODEL_KEY','MAX_STEPS','RUN_EVALUATION'}
    values={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(''.join(setup['source'])).body
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in desired}
    assert values=={'MODEL_KEY':'gemma4-12b','MAX_STEPS':5,'RUN_EVALUATION':False}
    for cell in notebook['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']),cell['id'],'exec')
