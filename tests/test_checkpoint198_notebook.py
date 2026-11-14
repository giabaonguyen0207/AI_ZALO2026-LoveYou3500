import json
from pathlib import Path
from types import SimpleNamespace
import sys


def test_notebook_generated_cells_compile_and_use_common_dependency_stack(tmp_path):
    from tools.create_checkpoint198_colab import build,SETUP,INSTALL,PREPARE,COMPARE
    path=build(tmp_path)
    nb=json.loads(path.read_text(encoding='utf-8'))
    cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert [''.join(c['source']) for c in cells]==[SETUP,INSTALL,PREPARE,COMPARE]
    for c in cells:
        compile(''.join(c['source']),c['id'],'exec')
        assert c['outputs']==[]
    assert '.[llm,vllm,rag]' in INSTALL and 'peft==0.21.2' in INSTALL


def test_comparison_cell_passes_only_merged_checkpoint_to_shared_runner(tmp_path):
    from tools.create_checkpoint198_colab import COMPARE
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
    from tools.create_checkpoint198_colab import PREPARE
    # The CPU child exporter, checksum verification and the real model-copy
    # helper are independently exercised; this checks notebook wiring/order.
    assert PREPARE.index('verify_package(')<PREPARE.index('"merge"')
    assert 'checkpoint_inference.py' in PREPARE and 'prepare_local_model' in PREPARE
    assert 'merge_and_unload' not in PREPARE  # No model loaded in notebook kernel.
