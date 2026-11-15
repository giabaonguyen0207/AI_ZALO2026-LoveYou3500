import json

import pytest

from tools.reproduce_training import prepare_config


def test_reproduction_uses_frozen_data_and_new_parent_without_overwriting_evidence(tmp_path):
    root = tmp_path/'evidence'; (root/'configs').mkdir(parents=True)
    original = {'base_dir':'/old/base','dataset_dir':'/old/data','output_dir':'/old/out',
                'epochs':0.5,'learning_rate':1e-5,'initial_adapter':'/old/parent',
                'initial_adapter_files':{'old':'old'}}
    (root/'configs/stage2.json').write_text(json.dumps(original))
    work = tmp_path/'new-run'; parent = work/'stage1/adapter'; parent.mkdir(parents=True)
    for name in ('adapter_config.json','adapter_model.safetensors','training_report.json'):
        (parent/name).write_text('fresh parent')
    config = prepare_config(root,tmp_path/'base',work,'stage2')
    assert config['epochs'] == 0.5 and config['learning_rate'] == 1e-5
    assert config['dataset_dir'] == str((root/'data/stage2').resolve())
    assert config['initial_adapter'] == str(parent.resolve())
    assert len(config['initial_adapter_files']) == 3
    assert json.loads((root/'configs/stage2.json').read_text()) == original
    (work/'stage2/adapter').mkdir(parents=True)
    with pytest.raises(ValueError,match='overwrite'):
        prepare_config(root,tmp_path/'base',work,'stage2')
