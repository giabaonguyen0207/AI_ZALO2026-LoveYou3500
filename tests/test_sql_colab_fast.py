import json
import sys

from datamate.models.artifacts import sha256


def test_notebook_auto_resumes_and_reuses_completed_output(tmp_path):
    from tools.build_sql_colab_fast import build
    notebook=tmp_path/'fast.ipynb'
    build(notebook,'a'*64,'source.zip','dataset.zip')
    cells=json.loads(notebook.read_text(encoding='utf-8'))['cells']
    cell=next(''.join(c['source']) for c in cells if c['cell_type']=='code' and ''.join(c['source']).startswith('config_path=OUT'))
    output=tmp_path/'run'
    output.mkdir()
    profile={'output_dir':str(output/'adapter'),'max_steps':5}
    namespace={'OUT':output,'profile':profile,'RESUME':'auto','sys':sys,'json':json,'WORK':tmp_path}
    def run(command, **kwargs):
        folder=output/'adapter'
        folder.mkdir()
        weights=folder/'adapter_model.safetensors'
        weights.write_bytes(b'completed pilot')
        report={'training_config':profile,'adapter_files':{weights.name:sha256(weights)},'metrics':{},
                'train_sequences':40,'validation_sequences':0,'trainable_parameters':1,'peak_cuda_bytes':0,
                'received_resume':command[command.index('--resume')+1]}
        (folder/'training_report.json').write_text(json.dumps(report),encoding='utf-8')
    namespace['run_logged']=run
    exec(compile(cell,'<fast notebook train cell>','exec'),namespace)
    assert namespace['report']['received_resume']=='auto'
    namespace['run_logged']=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('must not train twice'))
    exec(compile(cell,'<completed notebook rerun>','exec'),namespace)


def test_notebook_rejects_changed_config_without_overwriting_saved_config(tmp_path):
    from tools.build_sql_colab_fast import build
    import pytest
    from datamate.domain import DataMateError
    notebook=tmp_path/'fast.ipynb'
    build(notebook,'a'*64,'source.zip','dataset.zip')
    cells=json.loads(notebook.read_text(encoding='utf-8'))['cells']
    cell=next(''.join(c['source']) for c in cells if c['cell_type']=='code' and ''.join(c['source']).startswith('config_path=OUT'))
    output=tmp_path/'run'
    output.mkdir()
    original='{"max_steps":5}'
    (output/'training_config.json').write_text(original,encoding='utf-8')
    with pytest.raises(DataMateError,match='RUN_TAG'):
        exec(compile(cell,'<changed profile>','exec'),{'OUT':output,'profile':{'max_steps':-1},'json':json})
    assert (output/'training_config.json').read_text(encoding='utf-8')==original
