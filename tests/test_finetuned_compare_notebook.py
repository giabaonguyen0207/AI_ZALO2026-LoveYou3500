import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from test_completed_adapter_inference import completed


def test_notebook_generated_cells_compile_and_use_common_dependency_stack(tmp_path):
    from tools.create_finetuned_compare_colab import build,SETUP,INSTALL,PREPARE,COMPARE
    path=build(tmp_path)
    nb=json.loads(path.read_text(encoding='utf-8'))
    cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert [''.join(c['source']) for c in cells]==[SETUP,INSTALL,PREPARE,COMPARE]
    for c in cells:
        compile(''.join(c['source']),c['id'],'exec')
        assert c['outputs']==[]
    assert '.[llm,vllm,rag]' in INSTALL and 'peft==0.21.2' in INSTALL


def test_comparison_cell_passes_only_merged_checkpoint_to_shared_runner(tmp_path):
    from tools.create_finetuned_compare_colab import COMPARE
    ids=tmp_path/'ids.json';ids.write_text('["q1"]')
    settings=tmp_path/'settings.json'
    settings.write_text(json.dumps({'model_key':'qwen35-9b'}))
    calls=[]
    namespace={'WORK':Path(__file__).resolve().parents[1],'sys':sys,
        'subprocess':SimpleNamespace(run=lambda cmd,**kw:calls.append((cmd,kw))),
        'IDS_FILE':ids,'REFERENCES_FILE':None,'TRAIN_ROOT':tmp_path/'split',
        'MODEL_DIR':tmp_path/'local-merged','EMBEDDING_DIR':tmp_path/'embedding','RERANKER_DIR':tmp_path/'reranker',
        'COMPARE_OUTPUT':tmp_path/'results','DTYPE':'float16','Path':Path,'json':json}
    exec(compile(COMPARE,'compare','exec'),namespace)
    assert len(calls)==1
    command,options=calls[0]
    assert command[command.index('--model-dir')+1]==str(namespace['MODEL_DIR'])
    assert command[2].endswith('tools/compare_system.py') or command[2].endswith('tools\\compare_system.py')
    assert '--references' not in command


def test_preparation_uses_cpu_child_and_checks_checkpoint_before_merge():
    from tools.create_finetuned_compare_colab import PREPARE
    # The CPU child exporter, checksum verification and the real model-copy
    # helper are independently exercised; this checks notebook wiring/order.
    assert PREPARE.index('verify_package(')<PREPARE.index('"merge"')
    assert 'checkpoint_inference.py' in PREPARE and 'prepare_local_model' in PREPARE
    assert 'merge_and_unload' not in PREPARE  # No model loaded in notebook kernel.


def test_finetuned_profile_pins_completed_report_and_whole_package():
    from tools.create_finetuned_compare_colab import PROFILE,SETUP,PREPARE
    assert PROFILE['source_kind']=='completed_adapter'
    assert PROFILE['verified_requested_total_epochs']==1.5
    assert len(PROFILE['archive_sha256'])==64
    assert PROFILE['source_training_report_sha256']=='9a09f391253ab39dd520d5862758f33f52cd461ee4f7830379816b35ac6d3d85'
    assert 'EXPECTED_PACKAGE_SHA256' in SETUP and 'EXPECTED_TRAINING_REPORT_SHA256' in SETUP
    assert PREPARE.index('sha256(ADAPTER_ZIP)')<PREPARE.index('extract_package(')
    assert 'completed_adapter' in PREPARE


def test_prepare_rejects_wrong_zip_before_merge_or_copy(tmp_path):
    from tools.create_finetuned_compare_colab import PREPARE
    wrong=tmp_path/'wrong.zip';wrong.write_bytes(b'not the selected artifact')
    with pytest.raises(RuntimeError,match='differs'):
        exec(compile(PREPARE,'prepare','exec'),{'time':time,'ADAPTER_ZIP':wrong,'EXPECTED_PACKAGE_SHA256':'0'*64})


def test_prepare_and_compare_use_one_completed_merged_artifact(completed,tmp_path,monkeypatch,split_root):
    import datamate.models.artifacts as artifacts
    import tools.colab_model_cache as cache
    from tools.checkpoint_inference import package_completed_adapter
    from tools.create_finetuned_compare_colab import PREPARE,COMPARE
    run,parent=completed
    packet=tmp_path/'complete.zip'
    manifest=package_completed_adapter(run,packet,parent_adapter=parent)
    merged=tmp_path/'stored-merged';merged.mkdir()
    (merged/'artifact_manifest.json').write_text('{}')
    storage=tmp_path/'drive'
    for key in ('vietnamese-embedding','vietnamese-reranker'):
        p=storage/'models'/key;p.mkdir(parents=True)
        (p/'artifact_manifest.json').write_text('{}')
    copies=[];commands=[]
    def copy_model(source,cache_root,key):
        copies.append(Path(source))
        return Path(cache_root)/(Path(source).name+'-local')
    monkeypatch.setattr(cache,'prepare_local_model',copy_model)
    monkeypatch.setattr(artifacts,'verify_artifact',lambda *args:{})
    save=tmp_path/'saved';save.mkdir()
    ids=tmp_path/'ids.json';ids.write_text('["question_0"]')
    namespace={'ADAPTER_ZIP':packet,'EXPECTED_PACKAGE_SHA256':artifacts.sha256(packet),
       'EXPECTED_ADAPTER_SHA256':manifest['files']['adapter/adapter_model.safetensors'],
       'EXPECTED_TRAINING_REPORT_SHA256':manifest['source_training_report_sha256'],
       'WORK':tmp_path/'work','BASE_SOURCE_DIR':tmp_path/'missing-base',
       'MERGED_STORE':merged,'SAVE':save,'STORAGE':storage,'time':time,'json':json,
       'sys':sys,'Path':Path,'subprocess':SimpleNamespace(STDOUT=-2,run=lambda cmd,**kw:commands.append(cmd)),
       'IDS_FILE':ids,'REFERENCES_FILE':None,'TRAIN_ROOT':split_root,
       'DTYPE':'float16','COMPARE_OUTPUT':save/'comparison'}
    exec(compile(PREPARE,'prepare','exec'),namespace)
    assert copies[0]==merged and namespace['BASE_SOURCE_DIR'] not in copies
    assert commands[0][commands[0].index('--output')+1]==str(merged)
    exec(compile(COMPARE,'compare','exec'),namespace)
    assert len(commands)==2
    command=commands[-1]
    assert command[command.index('--model-dir')+1]==str(namespace['MODEL_DIR'])
    assert command[2].endswith('compare_system.py')


def test_missing_benchmark_ids_fail_before_expensive_model_preparation(completed,tmp_path):
    from tools.checkpoint_inference import package_completed_adapter
    from tools.create_finetuned_compare_colab import PREPARE
    from datamate.models.artifacts import sha256
    run,parent=completed
    packet=tmp_path/'complete.zip'
    manifest=package_completed_adapter(run,packet,parent_adapter=parent)
    namespace={'time':time,'WORK':tmp_path/'work','ADAPTER_ZIP':packet,'EXPECTED_PACKAGE_SHA256':sha256(packet),
       'EXPECTED_ADAPTER_SHA256':manifest['files']['adapter/adapter_model.safetensors'],
       'EXPECTED_TRAINING_REPORT_SHA256':manifest['source_training_report_sha256'],
       'IDS_FILE':tmp_path/'missing-ids.json','TRAIN_ROOT':tmp_path/'data','json':json,'Path':Path,
       'MERGED_STORE':tmp_path/'not-merged','BASE_SOURCE_DIR':tmp_path/'missing-base'}
    with pytest.raises(FileNotFoundError,match='IDS_FILE'):
        exec(compile(PREPARE,'prepare','exec'),namespace)
    assert not namespace['MERGED_STORE'].exists()


def test_colab_setup_accepts_storage_and_dataset_paths_without_other_platform_controls(tmp_path,monkeypatch):
    import zipfile
    from types import ModuleType
    from tools.create_finetuned_compare_colab import SETUP
    storage=tmp_path/'drive';storage.mkdir()
    data=tmp_path/'split';data.mkdir();(data/'questions.jsonl').write_text('')
    packet=storage/'datamate-colab-source.zip'
    with zipfile.ZipFile(packet,'w') as z:z.writestr('src/datamate/__init__.py','# fixture source')
    google=ModuleType('google');google.__path__=[]
    colab=ModuleType('google.colab');colab.drive=SimpleNamespace(mount=lambda *args:None)
    monkeypatch.setitem(sys.modules,'google',google)
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    monkeypatch.delitem(sys.modules,'datamate',raising=False)
    monkeypatch.setattr(sys,'path',sys.path.copy())
    monkeypatch.chdir(tmp_path)
    for name in ('HF_HOME','TORCH_HOME','PIP_CACHE_DIR'):
        monkeypatch.setenv(name,os.environ.get(name,''))
    monkeypatch.setenv('DATAMATE_STORAGE_DIR',str(storage))
    monkeypatch.setenv('DATAMATE_WORK_DIR',str(tmp_path/'runtime'))
    monkeypatch.setenv('DATAMATE_SPLIT_ROOT',str(data))
    state={}
    exec(compile(SETUP,'setup','exec'),state)
    assert state['STORAGE']==storage and state['TRAIN_ROOT']==data
    assert (state['WORK']/'src/datamate/__init__.py').is_file()
    assert not {'BACKEND','RAG_MODE','RESUME'} & set(state)
