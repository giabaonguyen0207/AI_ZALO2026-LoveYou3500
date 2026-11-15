import json
from pathlib import Path
import zipfile

import pytest

from datamate.domain import DataMateError
from datamate.models.artifacts import sha256
from datamate.training.checkpoints import seal_checkpoint
from test_training_checkpoints import checkpoint


@pytest.fixture
def source(tmp_path):
    folder = checkpoint(tmp_path/'run',198)
    (folder/'adapter_config.json').write_text(json.dumps({'peft_type':'LORA','r':16}))
    (folder/'tokenizer.json').write_text('{}')
    (folder/'tokenizer_config.json').write_text('{}')
    (folder/'trainer_state.json').write_text(json.dumps({'global_step':198,'epoch':.5003158559696779}))
    identity = {'training':{'model_key':'qwen35-9b','epochs':.5,'initial_adapter_files':{'training_report.json':'parent'}},
                'base_manifest_sha256':'f'*64,'dataset_manifest_sha256':'d'*64}
    seal_checkpoint(folder,identity)
    return folder


def test_package_retains_adapter_tokenizer_and_provenance_without_optimizer(source,tmp_path):
    from tools.checkpoint_inference import package_checkpoint,extract_package,verify_package
    original = {p.name:sha256(p) for p in source.iterdir()}
    archive = tmp_path/'checkpoint.zip'
    result = package_checkpoint(source,archive,reported_total_epochs=1.5)
    assert result['source_global_step']==198 and result['reported_total_epochs']==1.5
    assert result['total_epochs_independently_verified'] is False
    with zipfile.ZipFile(archive) as z:
        assert {'adapter/adapter_model.safetensors','adapter/tokenizer.json','adapter/training_report.json',
                'source_checkpoint_complete.json','checkpoint_inference.json'} <= set(z.namelist())
        assert not any(n.endswith(('.pt','.pth','.bin')) for n in z.namelist())
    root = extract_package(archive,tmp_path/'unpacked')
    manifest = verify_package(root)
    assert manifest['expected_base_manifest_sha256']=='f'*64
    report = json.loads((root/'adapter/training_report.json').read_text())
    assert report['kind']=='trainer_checkpoint_snapshot' and report['training_finalization_verified'] is False
    assert report['source_checkpoint']['global_step']==198
    assert original == {p.name:sha256(p) for p in source.iterdir()}
    assert extract_package(archive,root)==root


@pytest.mark.parametrize('file',['adapter_model.safetensors','tokenizer.json','trainer_state.json'])
def test_package_rejects_modified_checkpoint(source,tmp_path,file):
    from tools.checkpoint_inference import package_checkpoint
    (source/file).write_bytes(b'changed')
    with pytest.raises(DataMateError): package_checkpoint(source,tmp_path/'bad.zip')
    assert not (tmp_path/'bad.zip').exists()


def test_upload_corruption_is_detected(source,tmp_path):
    from tools.checkpoint_inference import package_checkpoint,extract_package,verify_package
    archive=tmp_path/'checkpoint.zip'
    package_checkpoint(source,archive)
    root=extract_package(archive,tmp_path/'unpacked')
    (root/'adapter/adapter_model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(DataMateError): verify_package(root)


def test_merge_rejects_other_sealed_base_before_loading(source,tmp_path,monkeypatch):
    from tools import checkpoint_inference as tool
    archive=tmp_path/'checkpoint.zip'
    tool.package_checkpoint(source,archive)
    root=tool.extract_package(archive,tmp_path/'unpacked')
    base=tmp_path/'base';base.mkdir()
    (base/'artifact_manifest.json').write_text('{}')
    monkeypatch.setattr(tool,'verify_artifact',lambda *args:{})
    with pytest.raises(DataMateError,match='base|Base'):
        tool.merge_package(root,base,tmp_path/'merged')
    assert not (tmp_path/'merged').exists()


def test_existing_merged_model_with_other_adapter_is_rejected(source,tmp_path,monkeypatch):
    from tools import checkpoint_inference as tool
    archive=tmp_path/'checkpoint.zip'
    tool.package_checkpoint(source,archive)
    root=tool.extract_package(archive,tmp_path/'unpacked')
    output=tmp_path/'merged';output.mkdir()
    (output/'artifact_manifest.json').write_text('{}')
    monkeypatch.setattr(tool,'verify_artifact',lambda *args:{'lineage':{'training_report_sha256':'other'}})
    with pytest.raises(DataMateError): tool.merge_package(root,tmp_path/'base',output)
