import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('quantization',['none','8bit','4bit'])
def test_selected_precision_applies_to_both_models_and_separate_run(tmp_path,quantization):
    from tools.build_sql100_colab import build
    path=tmp_path/'quant.ipynb'
    build(path,'a'*64,'source.zip','data.zip',profile='l4',selectable_quantization=True)
    cells=json.loads(path.read_text(encoding='utf-8'))['cells']
    code=[''.join(c['source']) for c in cells if c['cell_type']=='code']
    for source in code:ast.parse(source)
    first=code[0].replace('QUANTIZATION = "4bit"',f'QUANTIZATION = "{quantization}"')
    namespace={};exec(first,namespace)
    name=namespace['RUN_NAME'];exec(first,namespace)
    assert namespace['RUN_NAME']==name and name.endswith('-'+quantization)
    store=tmp_path/'merged';store.mkdir();(store/'artifact_manifest.json').write_text('{}')
    saved={}
    namespace.update({'release':{'adapter_files':{'training_report.json':'digest'}},'MERGED_STORE':store,
        'verify_artifact':lambda *a:{'lineage':{'merged_lora':True,'training_report_sha256':'digest'}},
        'time':SimpleNamespace(monotonic=lambda:0),'BASE_STORE':tmp_path/'base',
        'stage_local':lambda source,target:target,'write_json':lambda *a:None,'RESULT':tmp_path/name,
        'invocation':'test','BENCH':tmp_path/'data',
        'ensure_training_config':lambda path,config:saved.update({path.stem:config})})
    exec(next(c for c in code if c.startswith('expected_report_sha=')),namespace)
    assert len(saved)==2
    for config in saved.values():
        assert config['quantization']==quantization and config['max_context_tokens']==16384
        assert config['max_new_tokens']==1536 and config['split_root']==str(tmp_path/'data/input')
        assert name in config['result_dir']
    with pytest.raises(AssertionError):exec(code[0].replace('QUANTIZATION = "4bit"','QUANTIZATION = "invalid"'),{})


def test_builder_preserves_explicit_precision_default(tmp_path):
    from tools.build_sql100_colab import build
    path=tmp_path/'none.ipynb'
    build(path,'a'*64,'source.zip','data.zip',profile='l4',selectable_quantization=True,default_quantization='none')
    cells=json.loads(path.read_text(encoding='utf-8'))['cells']
    source=next(''.join(c['source']) for c in cells if c['cell_type']=='code' and ''.join(c['source']).startswith('from pathlib import Path'))
    namespace={};exec(source,namespace)
    assert namespace['QUANTIZATION']=='none' and namespace['RUN_NAME'].endswith('-none')
