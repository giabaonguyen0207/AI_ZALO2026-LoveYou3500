import copy
import json
from pathlib import Path
import zipfile

import pytest

from datamate.domain import DataMateError
from datamate.models.artifacts import sha256
from datamate.models.registry import get_model
from datamate.training.checkpoints import build_run_identity


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def make_adapter(folder,*,epoch,step):
    folder.mkdir(parents=True)
    (folder/'adapter_config.json').write_text('{"peft_type":"LORA","r":16}')
    (folder/'adapter_model.safetensors').write_bytes(('fixture-'+str(epoch)).encode())
    for name in ('tokenizer.json','tokenizer_config.json'):
        (folder/name).write_text('{}')
    (folder/'chat_template.jinja').write_text('checkpoint template')
    config={'model_key':'qwen35-9b','epochs':epoch,'max_steps':-1,'method':'qlora'}
    identity=build_run_identity(config,'f'*64,'d'*64)
    report={'base_source':get_model('qwen35-9b').to_dict(),'training_config':config,'run_identity':identity,
            'global_step':step,'metrics':{'epoch':epoch},'training_manifest_sha256':'d'*64,
            'adapter_files':{n:sha256(folder/n) for n in ('adapter_config.json','adapter_model.safetensors')}}
    return report


@pytest.fixture
def completed(tmp_path):
    parent=tmp_path/'parent/adapter'
    report=make_adapter(parent,epoch=1,step=298)
    save(parent/'training_report.json',report)
    run=tmp_path/'stage2'
    adapter=run/'adapter'
    report=make_adapter(adapter,epoch=.5,step=198)
    initial={n:sha256(parent/n) for n in ('adapter_config.json','adapter_model.safetensors','training_report.json')}
    report['training_config']['initial_adapter_files']=initial
    report['run_identity']['training']['initial_adapter_files']=initial
    report['initial_adapter']={'adapter_files':initial,'parent_global_step':298}
    save(adapter/'training_report.json',report)
    save(adapter/'run_identity.json',report['run_identity'])
    save(run/'training_config.json',report['training_config'])
    (adapter/'checkpoint-198').mkdir()
    (adapter/'checkpoint-198/optimizer.pt').write_bytes(b'not an inference artifact')
    (run/'smoke_adapter').mkdir()
    return run,parent


def test_completed_adapter_preserves_original_reports_and_verified_epoch_chain(completed,tmp_path):
    from tools.checkpoint_inference import package_completed_adapter,extract_package,verify_package
    run,parent=completed
    adapter=run/'adapter'
    before={p.name:sha256(p) for p in adapter.iterdir() if p.is_file()}
    archive=tmp_path/'complete.zip'
    manifest=package_completed_adapter(run,archive,parent_adapter=parent)
    assert manifest['source_kind']=='completed_adapter' and manifest['training_finalization_verified']
    assert manifest['total_epochs_independently_verified'] and manifest['verified_requested_total_epochs']==1.5
    root=extract_package(archive,tmp_path/'unpacked')
    assert verify_package(root)['source_global_step']==198
    assert (root/'adapter/training_report.json').read_bytes()==(adapter/'training_report.json').read_bytes()
    assert (root/'provenance/parent_training_report.json').read_bytes()==(parent/'training_report.json').read_bytes()
    with zipfile.ZipFile(archive) as packet:
        assert not any('optimizer' in n or 'smoke' in n or 'checkpoint-' in n or n.endswith('.bin') for n in packet.namelist())
    assert before=={p.name:sha256(p) for p in adapter.iterdir() if p.is_file()}


@pytest.mark.parametrize('failure',['weights','config','parent','model_key','run_identity'])
def test_rejects_unverified_completed_adapter_or_parent(completed,tmp_path,failure):
    from tools.checkpoint_inference import package_completed_adapter
    run,parent=completed
    adapter=run/'adapter'
    if failure=='weights': (adapter/'adapter_model.safetensors').write_bytes(b'changed')
    elif failure=='config': save(run/'training_config.json',{'epochs':1})
    elif failure=='parent': (parent/'training_report.json').write_text('{}')
    elif failure=='model_key':
        report=json.loads((adapter/'training_report.json').read_text())
        report['base_source']['revision']='other'
        save(adapter/'training_report.json',report)
    else: save(adapter/'run_identity.json',{'base_manifest_sha256':'other'})
    with pytest.raises(DataMateError):
        package_completed_adapter(run,tmp_path/'bad.zip',parent_adapter=parent)
    assert not (tmp_path/'bad.zip').exists()


def test_without_parent_report_does_not_claim_verified_total_epochs(completed,tmp_path):
    from tools.checkpoint_inference import package_completed_adapter
    run,_=completed
    manifest=package_completed_adapter(run,tmp_path/'no-parent.zip')
    assert manifest['total_epochs_independently_verified'] is False
    assert manifest['verified_requested_total_epochs'] is None


def test_completed_package_detects_report_tamper_and_metadata_mismatch(completed,tmp_path):
    from tools.checkpoint_inference import package_completed_adapter,extract_package,verify_package
    run,parent=completed
    archive=tmp_path/'complete.zip'
    package_completed_adapter(run,archive,parent_adapter=parent)
    root=extract_package(archive,tmp_path/'unpacked')
    manifest=json.loads((root/'checkpoint_inference.json').read_text())
    manifest['source_global_step']=1
    save(root/'checkpoint_inference.json',manifest)
    with pytest.raises(DataMateError):verify_package(root)


def test_epoch_claim_must_match_immutable_training_identity(completed,tmp_path):
    from tools.checkpoint_inference import package_completed_adapter
    run,parent=completed
    report=json.loads((run/'adapter/training_report.json').read_text())
    report['training_config']['epochs']=.25
    save(run/'adapter/training_report.json',report)
    save(run/'training_config.json',report['training_config'])
    with pytest.raises(DataMateError,match='identity|Identity'):
        package_completed_adapter(run,tmp_path/'inconsistent.zip',parent_adapter=parent)


def test_interrupted_extract_is_not_published_and_can_be_retried(completed,tmp_path,monkeypatch):
    import tools.checkpoint_inference as tool
    run,parent=completed
    packet=tmp_path/'complete.zip'
    tool.package_completed_adapter(run,packet,parent_adapter=parent)
    destination=tmp_path/'unpacked'
    def interrupted(*args,**kwargs): raise OSError('interrupted copy')
    with monkeypatch.context() as patch:
        patch.setattr(tool.shutil,'copyfileobj',interrupted)
        with pytest.raises(OSError,match='interrupted'):
            tool.extract_package(packet,destination)
    assert not destination.exists()
    assert tool.verify_package(tool.extract_package(packet,destination))['source_global_step']==198


def test_merged_cache_identity_includes_tokenizer_and_template(completed,tmp_path,monkeypatch):
    import tools.checkpoint_inference as tool
    run,parent=completed
    first=tmp_path/'first.zip'
    tool.package_completed_adapter(run,first,parent_adapter=parent)
    first_root=tool.extract_package(first,tmp_path/'first')
    (run/'adapter/chat_template.jinja').write_text('another tokenizer template')
    second=tmp_path/'second.zip'
    tool.package_completed_adapter(run,second,parent_adapter=parent)
    second_root=tool.extract_package(second,tmp_path/'second')
    output=tmp_path/'merged';output.mkdir();(output/'artifact_manifest.json').write_text('{}')
    monkeypatch.setattr(tool,'verify_artifact',lambda *args:{'lineage':{
        'merged_lora':True,'training_report_sha256':sha256(first_root/'adapter/training_report.json'),
        'base_manifest_sha256':'f'*64,'inference_package_sha256':sha256(first_root/'checkpoint_inference.json')}})
    assert tool.merge_package(first_root,tmp_path/'base',output)['reused'] is True
    with pytest.raises(DataMateError,match='another|package'):
        tool.merge_package(second_root,tmp_path/'base',output)
