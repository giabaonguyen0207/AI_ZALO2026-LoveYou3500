import json
from pathlib import Path

import pytest

from datamate.domain import DataMateError


def test_selection_is_label_independent_proportional_and_order_independent():
    from tools.sql100 import select_questions
    rows=[{'id':f'q_{b}_{i}','business_id':b,'question':'test','as_of_date':'2026-01-01'}
          for b,n in [('A',8),('B',12)] for i in range(n)]
    a=select_questions(rows,[r['id'] for r in rows],10)
    b=select_questions(list(reversed(rows)),[r['id'] for r in rows],10)
    assert a==b and len(a)==10
    assert sum(r['business_id']=='A' for r in a)==4
    with pytest.raises(DataMateError):select_questions(rows,['missing'],1)


def test_sealed_benchmark_rejects_changed_inputs_or_gold_in_split(tmp_path):
    from tools.sql100 import verify_benchmark,hash_file,write_json
    root=tmp_path/'bench';(root/'input/data').mkdir(parents=True)
    q=[{'id':'a','business_id':'biz','question':'x','as_of_date':'2026-01-01'}]
    (root/'input/questions.jsonl').write_text(json.dumps(q[0])+'\n')
    (root/'input/rulebook.md').write_text('rules')
    (root/'input/calendar.csv').write_text('calendar')
    write_json(root/'question_ids.json',['a'])
    files={p.relative_to(root).as_posix():hash_file(p) for p in root.rglob('*') if p.is_file()}
    write_json(root/'benchmark_manifest.json',{'sample_size':1,'files':files})
    assert verify_benchmark(root)['sample_size']==1
    (root/'input/answers.jsonl').write_text('{}')
    with pytest.raises(DataMateError):verify_benchmark(root)
    (root/'input/answers.jsonl').unlink()
    (root/'input/rulebook.md').write_text('changed')
    with pytest.raises(DataMateError):verify_benchmark(root)


def test_run_resume_does_not_repeat_questions_and_rejects_changed_identity(tmp_path,monkeypatch):
    import tools.sql100 as runner
    from datamate.config import Settings
    from datamate.domain import Answer,RunResult
    root=tmp_path/'data';(root/'input/data/biz').mkdir(parents=True)
    (root/'input/rulebook.md').write_text('rules');(root/'input/calendar.csv').write_text('calendar')
    qs=[{'id':str(i),'business_id':'biz','question':'x','as_of_date':'2026-01-01'} for i in range(3)]
    (root/'input/questions.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in qs))
    runner.write_json(root/'question_ids.json',[q['id'] for q in qs])
    runner.write_json(root/'benchmark_manifest.json',{'sample_size':3,'files':{
        p.relative_to(root).as_posix():runner.hash_file(p) for p in root.rglob('*') if p.is_file()}})
    model=tmp_path/'model';model.mkdir();(model/'artifact_manifest.json').write_text('{}')
    config=Settings(model_dir=str(model),split_root=str(root/'input'),cache_dir=str(tmp_path/'cache'),result_dir=str(tmp_path/'out')).to_dict()
    cp=tmp_path/'config.json';runner.write_json(cp,config)
    calls=[]
    class Agent:
        def run(self,q,remaining):
            calls.append(q.id)
            if q.id=='1' and len(calls)==2:raise RuntimeError('runtime lost')
            return RunResult(Answer('number',1),0.01,[],0)
    monkeypatch.setattr(runner,'create_agent',lambda settings:Agent())
    monkeypatch.setattr(runner,'runtime_identity',lambda:{'fixture':'cpu'})
    with pytest.raises(RuntimeError,match='runtime lost'):runner.run_variant(cp,root)
    summary=runner.run_variant(cp,root)
    assert summary['completed']==3 and calls.count('0')==1
    assert summary['timing_incomplete_after_runtime_loss'] and summary['wall_seconds'] is None
    before=len(calls);runner.run_variant(cp,root);assert len(calls)==before
    config['max_new_tokens']=999;runner.write_json(cp,config)
    with pytest.raises(DataMateError,match='identity'):runner.run_variant(cp,root)


def test_paired_report_rejects_partial_or_unfair_runs_and_scores_same_ids(tmp_path,monkeypatch):
    import tools.sql100 as runner
    from datamate.config import Settings
    from datamate.domain import Answer,RunResult
    root=tmp_path/'bench';(root/'input/data/biz').mkdir(parents=True);(root/'evaluation').mkdir()
    (root/'input/rulebook.md').write_text('rules');(root/'input/calendar.csv').write_text('calendar')
    q={'id':'a','business_id':'biz','question':'x','as_of_date':'2026-01-01'}
    (root/'input/questions.jsonl').write_text(json.dumps(q)+'\n')
    (root/'evaluation/references.jsonl').write_text('{"id":"a","type":"number","answer":1}\n')
    runner.write_json(root/'question_ids.json',['a'])
    runner.write_json(root/'benchmark_manifest.json',{'sample_size':1,'files':{
        p.relative_to(root).as_posix():runner.hash_file(p) for p in root.rglob('*') if p.is_file()}})
    class Agent:
        def __init__(self,value):self.value=value
        def run(self,q,remaining):return RunResult(Answer('number',self.value),.01,[],1)
    monkeypatch.setattr(runner,'create_agent',lambda settings:Agent(1 if settings.model_dir.endswith('base') else 2))
    monkeypatch.setattr(runner,'runtime_identity',lambda:{'fixture':'cpu'})
    for label in ('base','fine'):
        model=tmp_path/label;model.mkdir();runner.write_json(model/'artifact_manifest.json',{'fixture':label})
        config=Settings(model_dir=str(model),split_root=str(root/'input'),cache_dir=str(tmp_path/('cache-'+label)),
                        result_dir=str(tmp_path/('out-'+label))).to_dict()
        path=tmp_path/(label+'.json');runner.write_json(path,config);runner.run_variant(path,root)
    result=runner.paired_report(root,tmp_path/'out-base',tmp_path/'out-fine',tmp_path/'review')
    assert result['baseline']['correct']==1 and result['finetuned']['wrong']==1
    assert result['delta_pa']==-2
    assert len(json.loads((tmp_path/'review/regressions.json').read_text()))==1
    # A separately sealed benchmark with the same IDs must not reuse these runs.
    import shutil
    other=tmp_path/'other';shutil.copytree(root,other)
    (other/'input/rulebook.md').write_text('different rules')
    meta=json.loads((other/'benchmark_manifest.json').read_text())
    meta['files']['input/rulebook.md']=runner.hash_file(other/'input/rulebook.md')
    runner.write_json(other/'benchmark_manifest.json',meta)
    with pytest.raises(DataMateError,match='benchmark'):
        runner.paired_report(other,tmp_path/'out-base',tmp_path/'out-fine',tmp_path/'other-review')
    record=next((tmp_path/'out-fine/records').glob('*.json'));record.unlink()
    with pytest.raises(DataMateError,match='Chưa đủ'):runner.paired_report(root,tmp_path/'out-base',tmp_path/'out-fine',tmp_path/'review')
    runner.run_variant(tmp_path/'fine.json',root)
    ip=tmp_path/'out-fine/evaluation_identity.json';identity=json.loads(ip.read_text())
    identity['settings']['max_new_tokens']=999;runner.write_json(ip,identity)
    with pytest.raises(DataMateError,match='cấu hình'):runner.paired_report(root,tmp_path/'out-base',tmp_path/'out-fine',tmp_path/'review')


def test_generated_notebook_has_separate_processes_and_frozen_drive_root(tmp_path):
    from tools.build_sql100_colab import build
    import ast
    path=tmp_path/'compare.ipynb';build(path,'a'*64,'source.zip','data.zip')
    nb=json.loads(path.read_text(encoding='utf-8'))
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert '/content/drive/MyDrive/DataMateSQLSFT' in code
    assert 'checkpoint-190' not in code and 'smoke_adapter' not in code
    assert 'baseline' in code and 'finetuned' in code
    assert 'run_logged' in code and 'sql100.py' in code
    import sys
    from uuid import uuid4
    commands=[]
    namespace={'sys':sys,'WORK':tmp_path,'BENCH':tmp_path/'data','RESULT':tmp_path/'result',
               'configs':{'baseline':tmp_path/'base.json','finetuned':tmp_path/'fine.json'},'uuid4':uuid4,
               'run_logged':lambda command,**kwargs:commands.append(command)}
    for c in nb['cells']:
        src=''.join(c['source'])
        if src.startswith('run_logged') and '"run",' in src:exec(compile(src,'<notebook run cell>','exec'),namespace)
    assert len(commands)==2
    assert commands[0][-1]==commands[1][-1]==str(tmp_path/'data')
    assert commands[0][commands[0].index('--config')+1]!=commands[1][commands[1].index('--config')+1]
