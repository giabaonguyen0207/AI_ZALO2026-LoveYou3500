import json
from pathlib import Path

import pytest

from datamate.domain import DataMateError


def checkpoint(root, step):
    path = root/f'checkpoint-{step}'
    path.mkdir(parents=True)
    for name in ('adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
                 'rng_state.pth','training_args.bin'):
        (path/name).write_bytes(b'fixture payload')
    (path/'trainer_state.json').write_text(json.dumps({'global_step':step}), encoding='utf-8')
    return path


def test_auto_resume_skips_unfinished_or_corrupted_latest_checkpoint(tmp_path):
    from datamate.training.checkpoints import seal_checkpoint, select_checkpoint
    identity = {'base':'revision-A','dataset':'hash-A','max_steps':20}
    first = checkpoint(tmp_path, 1)
    seal_checkpoint(first, identity)
    latest = checkpoint(tmp_path, 5)
    assert select_checkpoint(tmp_path, 'auto', identity) == first
    seal_checkpoint(latest, identity)
    assert select_checkpoint(tmp_path, 'auto', identity) == latest
    (latest/'optimizer.pt').write_bytes(b'truncated')
    assert select_checkpoint(tmp_path, 'auto', identity) == first


def test_incomplete_checkpoint_cannot_be_marked_complete(tmp_path):
    from datamate.training.checkpoints import seal_checkpoint
    path = checkpoint(tmp_path, 1)
    (path/'rng_state.pth').unlink()
    with pytest.raises(DataMateError): seal_checkpoint(path, {'dataset':'A'})
    assert not (path/'checkpoint_complete.json').exists()


def test_resume_rejects_changed_run_or_external_checkpoint(tmp_path):
    from datamate.training.checkpoints import seal_checkpoint, select_checkpoint
    root = tmp_path/'run'
    path = checkpoint(root, 5)
    seal_checkpoint(path, {'max_steps':5,'dataset':'A'})
    with pytest.raises(DataMateError): select_checkpoint(root, 'auto', {'max_steps':50,'dataset':'A'})
    with pytest.raises(DataMateError): select_checkpoint(tmp_path/'other', str(path), {'max_steps':5,'dataset':'A'})


def test_valid_checkpoint_requires_matching_step_and_keeps_previous_marker(tmp_path):
    from datamate.training.checkpoints import seal_checkpoint, select_checkpoint
    path = checkpoint(tmp_path, 5)
    seal_checkpoint(path, {'dataset':'A'})
    marker = (path/'checkpoint_complete.json').read_bytes()
    with pytest.raises(DataMateError): seal_checkpoint(path, {'dataset':'different'})
    assert (path/'checkpoint_complete.json').read_bytes() == marker
    (path/'trainer_state.json').write_text('{"global_step":4}', encoding='utf-8')
    with pytest.raises(DataMateError): select_checkpoint(tmp_path, str(path), {'dataset':'A'})


def test_callback_saves_first_step_periodic_time_and_marks_after_save(tmp_path):
    from types import SimpleNamespace
    from datamate.training.checkpoints import make_checkpoint_callback
    callback = make_checkpoint_callback({'dataset':'A'}, interval_seconds=0)
    state = SimpleNamespace(global_step=1)
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(SimpleNamespace(), state, control)
    assert control.should_save
    path = checkpoint(tmp_path, 1)
    callback.on_save(SimpleNamespace(output_dir=str(tmp_path)), state, control)
    assert (path/'checkpoint_complete.json').is_file()


def test_real_trainer_interruption_and_resume_matches_uninterrupted_training(tmp_path):
    pytest.importorskip('accelerate')
    torch = pytest.importorskip('torch')
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from datamate.training.checkpoints import make_checkpoint_callback, select_checkpoint

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)
            self.dropout = torch.nn.Dropout(0.1)
        def forward(self, input_ids, labels):
            logits = self.layer(self.dropout(input_ids.float()))
            return {'loss':torch.nn.functional.cross_entropy(logits, labels), 'logits':logits}

    class Interrupt(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.global_step == 1: raise RuntimeError('simulated runtime loss')

    identity = {'max_steps':3,'dataset':'fixed-tiny-fixture','seed':42}
    rows = [{'input_ids':[float(i), 0.2, 0.3], 'labels':i%2} for i in range(8)]
    def trainer(directory, interrupt=False):
        torch.manual_seed(123)
        args = TrainingArguments(output_dir=str(directory), use_cpu=True, max_steps=3,
            per_device_train_batch_size=2, learning_rate=0.01, seed=42, data_seed=42,
            save_steps=1, report_to='none', disable_tqdm=True)
        callbacks = [make_checkpoint_callback(identity, interval_seconds=600)]
        if interrupt: callbacks.append(Interrupt())
        return Trainer(model=Model(), args=args, train_dataset=rows, callbacks=callbacks)

    baseline = trainer(tmp_path/'baseline')
    baseline.train()
    first = trainer(tmp_path/'interrupted', interrupt=True)
    with pytest.raises(RuntimeError, match='simulated runtime loss'): first.train()
    path = select_checkpoint(tmp_path/'interrupted', 'auto', identity)
    resumed = trainer(tmp_path/'interrupted')
    resumed.train(resume_from_checkpoint=str(path))
    assert resumed.state.global_step == baseline.state.global_step == 3
    for name, weight in baseline.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], weight, rtol=0, atol=0)


def test_identity_covers_training_duration_and_normalizes_defaults():
    from datamate.training.checkpoints import build_run_identity
    a = build_run_identity({'model_key':'qwen35-9b'}, 'base-A', 'data-A')
    b = build_run_identity({'model_key':'qwen35-9b','max_steps':-1,'epochs':2}, 'base-A', 'data-A')
    assert a == b
    assert a != build_run_identity({'model_key':'qwen35-9b','max_steps':5}, 'base-A','data-A')
    assert a != build_run_identity({'model_key':'qwen35-9b','epochs':1}, 'base-A','data-A')
    assert a != build_run_identity({'model_key':'qwen35-9b'}, 'base-A','data-B')


def test_corrupted_checkpoint_is_preserved_away_from_future_save_path(tmp_path):
    from datamate.training.checkpoints import seal_checkpoint, select_checkpoint
    identity = {'dataset':'A'}
    first = checkpoint(tmp_path, 1)
    seal_checkpoint(first, identity)
    broken = checkpoint(tmp_path, 5)
    assert select_checkpoint(tmp_path, 'auto', identity, quarantine=True) == first
    assert not broken.exists()
    preserved = list(tmp_path.glob('incomplete-checkpoint-5-*'))
    assert len(preserved) == 1 and (preserved[0]/'optimizer.pt').read_bytes() == b'fixture payload'
    replacement = checkpoint(tmp_path, 5)
    seal_checkpoint(replacement, identity)
    assert select_checkpoint(tmp_path, 'auto', identity) == replacement


def test_completed_report_checks_weights_before_reusing_result(tmp_path):
    from datamate.training.checkpoints import completed_report
    from datamate.models.artifacts import sha256
    config = {'max_steps':5}
    weights = tmp_path/'adapter_model.safetensors'
    weights.write_bytes(b'learned adapter')
    report = {'training_config':config,'adapter_files':{weights.name:sha256(weights)}}
    assert completed_report(tmp_path, config) is None
    (tmp_path/'training_report.json').write_text(json.dumps(report), encoding='utf-8')
    assert completed_report(tmp_path, config) == report
    weights.write_bytes(b'corrupt')
    with pytest.raises(DataMateError): completed_report(tmp_path, config)


def test_same_runtime_cannot_launch_duplicate_training_for_one_output(tmp_path):
    from datamate.training.checkpoints import training_lock
    with training_lock(tmp_path):
        with pytest.raises(DataMateError,match='đang chạy'):
            with training_lock(tmp_path): pass
    with training_lock(tmp_path): pass


def test_actual_peft_adapter_resume_preserves_optimizer_and_rng(tmp_path):
    pytest.importorskip('accelerate')
    peft=pytest.importorskip('peft')
    torch=pytest.importorskip('torch')
    from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainerCallback, TrainingArguments
    from datamate.training.checkpoints import make_checkpoint_callback, select_checkpoint
    identity={'model':'tiny-gpt2-lora','max_steps':3,'seed':42}
    rows=[{'input_ids':[1,2,i%10+3,4,5], 'labels':[-100,-100,i%10+3,4,5]} for i in range(8)]
    class Interrupt(TrainerCallback):
        def on_save(self,args,state,control,**kwargs):
            if state.global_step==1: raise RuntimeError('lost runtime with PEFT')
    def make(output,interrupt=False):
        torch.manual_seed(123)
        model=GPT2LMHeadModel(GPT2Config(vocab_size=17,n_positions=16,n_embd=16,n_layer=1,n_head=2))
        model=peft.get_peft_model(model,peft.LoraConfig(r=2,lora_alpha=4,lora_dropout=.05,fan_in_fan_out=True,
                                                      target_modules=['c_attn'],task_type='CAUSAL_LM'))
        args=TrainingArguments(output_dir=str(output),use_cpu=True,max_steps=3,per_device_train_batch_size=1,
            gradient_accumulation_steps=2,seed=42,save_steps=1,report_to='none',disable_tqdm=True)
        callbacks=[make_checkpoint_callback(identity)]
        if interrupt: callbacks.append(Interrupt())
        return Trainer(model=model,args=args,train_dataset=rows,callbacks=callbacks)
    baseline=make(tmp_path/'base')
    baseline.train()
    interrupted=make(tmp_path/'recover',True)
    with pytest.raises(RuntimeError,match='lost runtime'): interrupted.train()
    selected=select_checkpoint(tmp_path/'recover','auto',identity)
    resumed=make(tmp_path/'recover')
    resumed.train(resume_from_checkpoint=str(selected))
    expected=peft.get_peft_model_state_dict(baseline.model)
    actual=peft.get_peft_model_state_dict(resumed.model)
    assert resumed.state.global_step==3
    for name,weight in expected.items():
        torch.testing.assert_close(actual[name],weight,rtol=0,atol=0)
    # Loss immediately after the final checkpoint must not create an extra update.
    from datamate.training.checkpoints import recover_completed_checkpoint
    final_path=select_checkpoint(tmp_path/'recover','auto',identity)
    finalized=make(tmp_path/'recover')
    metrics=recover_completed_checkpoint(finalized,final_path,identity)
    assert metrics['completion_recovered_from_checkpoint'] is True
    assert finalized.state.global_step==3
    for name,weight in expected.items():
        torch.testing.assert_close(peft.get_peft_model_state_dict(finalized.model)[name],weight,rtol=0,atol=0)


@pytest.mark.parametrize('name,content', [('checkpoint_complete.json','null'),
                                        ('checkpoint_complete.json','[]'),('trainer_state.json','null')])
def test_malformed_json_schema_falls_back_to_previous_checkpoint(tmp_path,name,content):
    from datamate.training.checkpoints import seal_checkpoint,select_checkpoint
    identity={'dataset':'A'}
    first=checkpoint(tmp_path,1)
    seal_checkpoint(first,identity)
    last=checkpoint(tmp_path,5)
    seal_checkpoint(last,identity)
    (last/name).write_text(content,encoding='utf-8')
    assert select_checkpoint(tmp_path,'auto',identity)==first


def test_truncated_root_metadata_recovers_from_verified_checkpoint(tmp_path):
    from datamate.training.checkpoints import seal_checkpoint,prepare_run,ensure_training_config
    identity={'dataset':'A','max_steps':5}
    path=checkpoint(tmp_path/'adapter',1)
    seal_checkpoint(path,identity)
    (path.parent/'run_identity.json').write_text('{"dataset":',encoding='utf-8')
    assert prepare_run(path.parent,'auto',identity)==path
    assert json.loads((path.parent/'run_identity.json').read_text(encoding='utf-8'))==identity
    profile={'max_steps':5}
    config_path=tmp_path/'training_config.json'
    config_path.write_text('{"max_steps":',encoding='utf-8')
    ensure_training_config(config_path,profile)
    assert json.loads(config_path.read_text(encoding='utf-8'))==profile
    assert len(list(tmp_path.glob('training_config.json.incomplete-*')))==1


def test_zero_byte_weight_file_cannot_be_sealed(tmp_path):
    from datamate.training.checkpoints import seal_checkpoint
    path=checkpoint(tmp_path,1)
    (path/'adapter_model.safetensors').write_bytes(b'')
    with pytest.raises(DataMateError): seal_checkpoint(path,{'dataset':'A'})
