import ast
import json
from pathlib import Path
import zipfile

import pytest

from datamate.application import run_batch
from datamate.config import Settings
from tools.colab_review import stage_split, extract_archive, export_submission, diagnostics
from tools.create_review_notebook import build


def test_training_stage_excludes_gold_selects_questions_and_is_repeatable(split_root,tmp_path):
    snapshot = (split_root/'questions.jsonl').read_bytes()
    staged,report=stage_split(split_root,tmp_path/'scratch',kind='train',limit=2)
    assert report['questions']==2 and report['businesses']==2
    assert not (staged/'answers.jsonl').exists()
    assert not (staged/'data/shop_c').exists()
    assert (split_root/'questions.jsonl').read_bytes()==snapshot
    assert stage_split(split_root,tmp_path/'scratch',kind='train',limit=2)==(staged,report)
    with pytest.raises(ValueError,match='Public'):
        stage_split(split_root,tmp_path/'scratch',kind='public',limit=2)


def test_nested_split_zip_and_traversal_rejection(split_root,tmp_path):
    archive=tmp_path/'train.zip'
    with zipfile.ZipFile(archive,'w') as stream:
        for path in split_root.rglob('*'):
            if path.is_file():stream.write(path,Path('train/train')/path.relative_to(split_root))
    staged,report=stage_split(archive,tmp_path/'scratch',kind='train',limit=1)
    assert report['questions']==1 and not (staged/'answers.jsonl').exists()
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as stream:stream.writestr('../outside.txt','no')
    with pytest.raises(ValueError,match='Unsafe'):
        extract_archive(bad,tmp_path/'unpack')
    assert not (tmp_path/'outside.txt').exists()


def test_export_after_completed_resume_and_diagnostics(split_root,tmp_path):
    result=tmp_path/'result'
    settings=Settings(backend='demo',split_root=str(split_root),cache_dir=str(tmp_path/'cache'),result_dir=str(result))
    run_batch(settings)
    run_batch(settings,resume=True)  # Previously written=all, newly written=zero.
    exported=tmp_path/'submission.jsonl'
    report=export_submission(split_root,result,exported)
    assert report['questions']==3 and report['types']=={'abstain':3}
    assert report['quality_evaluated'] is False
    assert diagnostics(result)['questions_logged']==3
    assert all(set(json.loads(line))=={'id','type'} for line in exported.read_text().splitlines())


def test_review_notebook_parses_and_keeps_frozen_stack_and_interface(tmp_path):
    path=build(tmp_path)
    notebook=json.loads(path.read_text(encoding='utf-8'))
    code=[''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code']
    for source in code:ast.parse(source)
    assert 'runtime-freeze.txt' in code[0]
    assert "'--upgrade'" not in code[1]
    assert "'-r',str(FROZEN_ENV),'-c',str(FROZEN_ENV)" in code[1]
    assert "'--config',str(CONFIG)" in code[3]
    assert "'gpu_memory_utilization'] = 0.75" in code[2]
    assert not any(cell.get('outputs') for cell in notebook['cells'])
    maintained=Path(__file__).resolve().parents[1]/'notebooks/DataMate_Colab_Review.ipynb'
    assert json.loads(maintained.read_text(encoding='utf-8'))==notebook


@pytest.mark.parametrize('key', ['qwen35-9b','gemma4-12b'])
@pytest.mark.parametrize('profile', ['default','optimized','correction','hybrid'])
def test_colab_selected_model_reaches_runtime_config_without_changing_budgets(tmp_path,monkeypatch,key,profile):
    from types import SimpleNamespace
    from tools.create_review_notebook import PREPARE
    from tools.colab_review import write_json
    import tools.colab_model_cache as cache
    root=Path(__file__).resolve().parents[1]
    save=tmp_path/'save';save.mkdir()
    folders={}
    for model in (key,'vietnamese-embedding','vietnamese-reranker','vietnamese-error-correction'):
        folder=tmp_path/model;folder.mkdir()
        (folder/'artifact_manifest.json').write_text('{}')
        folders[model]=folder
    copied=[]
    def copy(source,scratch,model):
        copied.append(model)
        return source
    monkeypatch.setattr(cache,'prepare_local_model',copy)
    calls=[]
    scope=dict(Path=Path,json=json,MODEL_KEY=key,GENERATOR_SOURCE=folders[key],
        EMBEDDING_SOURCE=folders['vietnamese-embedding'],RERANKER_SOURCE=folders['vietnamese-reranker'],
        CORRECTION_SOURCE=folders['vietnamese-error-correction'],RUNTIME_ENV={},
        stage_split=lambda *a,**kw:(tmp_path/'split',dict(fingerprint='input',questions=20,businesses=1)),
        DATA_SOURCE=tmp_path,SPLIT_KIND='train',LIMIT=20,SAVE=save,WORK=root,
        digest=lambda p:'732f292bfb088f8dc90b0fe91e255a60886772292614cf0b1af58219d097b054',
        write_json=write_json,source_hash='testsource',PROFILE=profile,RUNTIME_PYTHON='fake-python',
        OFFLINE_ENV={},subprocess=SimpleNamespace(run=lambda *a,**kw:calls.append(a)))
    exec(compile(PREPARE,'notebook-prepare','exec'),scope)
    config=json.loads((save/'config.json').read_text())
    assert config['model_key']==key and config['model_dir']==str(folders[key])
    assert copied==[key,'vietnamese-embedding','vietnamese-reranker']+(['vietnamese-error-correction'] if profile in {'optimized','correction'} else [])
    assert config['dtype']=='float16' and config['quantization']=='none'
    assert config['max_context_tokens']==16384 and config['question_workers']==(2 if profile=='optimized' else 1)
    assert config['vllm']['gpu_memory_utilization']==0.75
    assert config['rag']['device']=='cuda:0' and config['backend']=='vllm'
    assert scope['model_inventory']['total_parameters']<15_000_000_000 and calls
    assert config['rag'].get('correction_enabled',False)==(profile in {'optimized','correction'})
    assert config['rag'].get('hybrid_retrieval',False)==(profile in {'optimized','hybrid'})
    if profile=='optimized':
        assert config['vllm']['max_num_seqs']==2 and config['vllm']['enable_prefix_caching']
        assert config['vllm']['cache_prompt_prefix'] and config['recovery_steps']==2
