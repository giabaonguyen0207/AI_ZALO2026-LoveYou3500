import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from datamate.domain import DataMateError


def l4_cells(tmp_path):
    from tools.build_sql100_colab import build
    path=tmp_path/'l4.ipynb'
    build(path,'a'*64,'source-l4.zip','same-data.zip',profile='l4')
    nb=json.loads(path.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']


def test_l4_generated_configs_use_identical_nf4_and_keep_context(tmp_path):
    cells=l4_cells(tmp_path)
    merge=next(c for c in cells if c.startswith('expected_report_sha='))
    store=tmp_path/'models/merged';store.mkdir(parents=True)
    (store/'artifact_manifest.json').write_text('{}')
    recorded={}
    namespace={'release':{'adapter_files':{'training_report.json':'digest'}},'MERGED_STORE':store,
        'verify_artifact':lambda *a:{'lineage':{'merged_lora':True,'training_report_sha256':'digest'}},
        'time':SimpleNamespace(monotonic=lambda:0),'Path':Path,'BASE_STORE':tmp_path/'base',
        'stage_local':lambda source,target:target,'write_json':lambda *a:None,'RESULT':tmp_path/'result',
        'invocation':'test','RUN_NAME':'l4-run','BENCH':tmp_path/'same-data',
        'ensure_training_config':lambda path,config:recorded.update({path.stem:config})}
    exec(compile(merge,'<L4 merge/config cell>','exec'),namespace)
    assert len(recorded)==2
    a,b=recorded['baseline-config'],recorded['finetuned-config']
    for config in (a,b):
        assert config['quantization']=='4bit'
        assert config['max_context_tokens']==16384 and config['max_new_tokens']==1536
        assert config['split_root']==str(tmp_path/'same-data/input')
    shared=lambda c:{k:v for k,v in c.items() if k not in {'model_dir','result_dir','cache_dir'}}
    assert shared(a)==shared(b)


def test_l4_merge_cell_invokes_cpu_child_without_hiding_parent_gpu(tmp_path):
    merge=next(c for c in l4_cells(tmp_path) if c.startswith('expected_report_sha='))
    commands=[]
    namespace={'release':{'adapter_files':{'training_report.json':'digest'}},'MERGED_STORE':tmp_path/'merged',
        'verify_artifact':lambda *a:{'lineage':{'merged_lora':True,'training_report_sha256':'digest'}},
        'time':SimpleNamespace(monotonic=lambda:0),'Path':Path,'BASE_STORE':tmp_path/'base',
        'stage_local':lambda source,target:target,'write_json':lambda *a:None,'RESULT':tmp_path/'result',
        'invocation':'test','RUN_NAME':'l4-run','BENCH':tmp_path/'data','ADAPTER':tmp_path/'adapter',
        'WORK':tmp_path/'work','sys':sys,'uuid4':uuid4,'run_logged':lambda command,**kw:commands.append(command),
        'ensure_training_config':lambda *a:None}
    original=dict(os.environ)
    exec(compile(merge,'<L4 CPU merge command>','exec'),namespace)
    assert len(commands)==1 and Path(commands[0][2]).name=='merge_sql100_cpu.py'
    assert '--base' in commands[0] and '--adapter' in commands[0]
    assert dict(os.environ)==original


def test_l4_gpu_gate_accepts_23_gib_and_rejects_16_gib(tmp_path):
    source=next(c for c in l4_cells(tmp_path) if c.startswith('import torch'))
    asserts=[n for n in ast.parse(source).body if isinstance(n,ast.Assert)]
    gpu_asserts=asserts[:2]
    def check(gib):
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,
            get_device_properties=lambda i:SimpleNamespace(total_memory=gib*1024**3)))
        exec(compile(ast.Module(body=gpu_asserts,type_ignores=[]),'<GPU preflight>','exec'),{'torch':torch})
    check(23)
    with pytest.raises(AssertionError):check(16)


def test_cpu_merge_ram_policy_and_environment_isolation(monkeypatch):
    from tools.merge_sql100_cpu import check_cpu_ram
    assert check_cpu_ram(9653104368,40*1024**3)==32
    with pytest.raises(DataMateError,match='High RAM'):check_cpu_ram(9653104368,12*1024**3)
    pytest.importorskip('torch')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','parent-device')
    script=Path(__file__).resolve().parents[1]/'tools/merge_sql100_cpu.py'
    environment=dict(os.environ)
    environment['PYTHONPATH']=str(script.parents[1]/'src')+os.pathsep+environment.get('PYTHONPATH','')
    result=subprocess.run([sys.executable,'-X','utf8',str(script),'--probe'],capture_output=True,text=True,env=environment)
    assert result.returncode==0,result.stderr
    probe=json.loads(result.stdout)
    assert probe['cuda_visible_devices']=='' and probe['cuda_available'] is False
    assert probe['accelerate_use_cpu']=='true'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='parent-device'


def test_cpu_ram_check_accounts_for_container_limit(tmp_path):
    from tools.merge_sql100_cpu import effective_ram_bytes,check_cpu_ram
    (tmp_path/'memory.max').write_text(str(24*1024**3))
    (tmp_path/'memory.current').write_text(str(4*1024**3))
    available=effective_ram_bytes(100*1024**3,tmp_path)
    assert available==20*1024**3
    with pytest.raises(DataMateError):check_cpu_ram(9653104368,available)
    (tmp_path/'memory.max').write_text('max')
    assert effective_ram_bytes(40*1024**3,tmp_path)==40*1024**3


def test_actual_small_peft_bf16_cpu_merge_preserves_output():
    torch=pytest.importorskip('torch');peft=pytest.importorskip('peft')
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.proj=torch.nn.Linear(8,8,bias=False)
        def forward(self,x):return self.proj(x)
    torch.manual_seed(42)
    model=peft.get_peft_model(Tiny().to(dtype=torch.bfloat16),peft.LoraConfig(r=2,lora_alpha=4,target_modules=['proj']))
    with torch.no_grad():
        model.base_model.model.proj.lora_A['default'].weight.fill_(.125)
        model.base_model.model.proj.lora_B['default'].weight.fill_(.05)
    model.eval();x=torch.randn(2,8,dtype=torch.bfloat16)
    expected=model(x).detach()
    merged=model.merge_and_unload()
    assert merged.proj.weight.device.type=='cpu' and merged.proj.weight.dtype==torch.bfloat16
    torch.testing.assert_close(merged(x),expected,rtol=.03,atol=.005)
