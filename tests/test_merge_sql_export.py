import json
from pathlib import Path

import pytest

torch=pytest.importorskip('torch')
from safetensors.torch import save_file,load_file
from datamate.domain import DataMateError
from datamate.models.artifacts import parameter_count


def fixture(tmp_path):
    base=tmp_path/'base';out=tmp_path/'out';base.mkdir();out.mkdir()
    tensors={'model.weight':torch.ones(3,4),'mtp.fc.weight':torch.arange(6).reshape(2,3).float()}
    save_file(tensors,base/'model.safetensors')
    save_file({'model.weight':torch.full((3,4),2.)},out/'model.safetensors')
    return base,out,tensors


@pytest.mark.parametrize('sharded',[False,True])
def test_export_retains_original_mtp_and_exact_parameter_inventory(tmp_path,sharded):
    from tools.merge_sql_export import preserve_auxiliary_tensors
    base,out,tensors=fixture(tmp_path)
    if sharded:
        (out/'model.safetensors').rename(out/'model-00001-of-00001.safetensors')
        (out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':48},
            'weight_map':{'model.weight':'model-00001-of-00001.safetensors'}}))
    receipt=preserve_auxiliary_tensors(base,out)
    assert parameter_count(base)==parameter_count(out)==18
    assert receipt['preserved_parameters']==6
    index=json.loads((out/'model.safetensors.index.json').read_text())
    assert set(index['weight_map'])==set(tensors)
    assert not (out/'model.safetensors').exists()  # loader must use complete index
    assert torch.equal(load_file(out/index['weight_map']['mtp.fc.weight'])['mtp.fc.weight'],tensors['mtp.fc.weight'])
    assert torch.equal(load_file(out/index['weight_map']['model.weight'])['model.weight'],torch.full((3,4),2.))


@pytest.mark.parametrize('failure',['missing_active','extra','shape'])
def test_export_rejects_unexplained_missing_extra_or_changed_shape(tmp_path,failure):
    from tools.merge_sql_export import preserve_auxiliary_tensors
    base,out,_=fixture(tmp_path)
    tensors={'model.weight':torch.ones(3,4)}
    if failure=='missing_active':tensors={}
    elif failure=='extra':tensors['unexpected.weight']=torch.ones(2)
    else:tensors['model.weight']=torch.ones(2,6)
    save_file(tensors,out/'model.safetensors')
    with pytest.raises(DataMateError):preserve_auxiliary_tensors(base,out)


def test_actual_qwen_cpu_lora_export_with_mtp(monkeypatch,tmp_path):
    import dataclasses,shutil
    peft=pytest.importorskip('peft')
    from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration,PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models
    import datamate.models.artifacts as artifacts
    from datamate.models.registry import get_model
    from tools.merge_sql_export import merge_verified
    config=Qwen3_5Config(text_config={'vocab_size':31,'hidden_size':32,'intermediate_size':64,
        'num_hidden_layers':1,'layer_types':['full_attention'],'num_attention_heads':4,
        'num_key_value_heads':2,'head_dim':8,'rope_parameters':{'rope_type':'default','rope_theta':10000.,
        'partial_rotary_factor':.5,'mrope_section':[1,1,0]}},vision_config={'depth':1,'hidden_size':16,
        'intermediate_size':32,'num_heads':2,'out_hidden_size':32,'num_position_embeddings':16,
        'patch_size':2,'spatial_merge_size':1,'deepstack_visual_indexes':[]})
    base=tmp_path/'base';adapter=tmp_path/'adapter';out=tmp_path/'merged'
    model=Qwen3_5ForConditionalGeneration(config).to(dtype=torch.bfloat16);model.save_pretrained(base)
    mtp=torch.arange(6,dtype=torch.bfloat16).reshape(2,3)
    save_file({'mtp.fc.weight':mtp},base/'aux.safetensors')
    spec=dataclasses.replace(get_model('qwen35-9b'),parameters=parameter_count(base))
    monkeypatch.setattr(artifacts,'get_model',lambda key:spec)
    artifacts.seal_artifact(base,spec)
    model=peft.get_peft_model(model,peft.LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj'],task_type='CAUSAL_LM'))
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name:p.fill_(.1)
    model.save_pretrained(adapter)
    tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'<unk>':0,'x':1},unk_token='<unk>')),unk_token='<unk>')
    tok.save_pretrained(adapter)
    (adapter/'training_report.json').write_text(json.dumps({'base_source':spec.to_dict()}))
    result=merge_verified('qwen35-9b',base,adapter,out)
    assert result['tensor_parameters']==spec.parameters
    assert artifacts.verify_artifact(out,'qwen35-9b')['lineage']['auxiliary_export']['preserved_parameters']==6
    index=json.loads((out/'model.safetensors.index.json').read_text())
    assert torch.equal(load_file(out/index['weight_map']['mtp.fc.weight'])['mtp.fc.weight'],mtp)
    reloaded=Qwen3_5ForConditionalGeneration.from_pretrained(out,local_files_only=True,dtype=torch.bfloat16)
    key='model.language_model.layers.0.self_attn.q_proj.weight'
    assert not torch.equal(reloaded.state_dict()[key],load_file(base/'model.safetensors')[key])
