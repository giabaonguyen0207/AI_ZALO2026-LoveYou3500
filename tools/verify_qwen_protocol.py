"""Offline token/stop integration and log replay; NOT a full-model benchmark."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading

from datamate.agent.protocol import parse_action
from datamate.agent.runner import Agent
from datamate.config import Settings
from datamate.data.split import DatasetSplit
from datamate.domain import DataMateError, json_text
from datamate.models.backends import TransformersBackend, generation_token_settings
from datamate.models.registry import get_model
from datamate.training.dataset import require_approved_source
from tools.diagnose_train import diagnostic_plan, source_hashes

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_count(tokenizer, messages):
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'keys'): ids=ids['input_ids']
    if hasattr(ids,'tolist'): ids=ids.tolist()
    return len(ids[0] if ids and isinstance(ids[0],list) else ids)


def forced_generation(tokenizer, generation_config):
    """Real Qwen3.5/Transformers generation, tiny random weights, forced tokens.

    The continuation deliberately contains a fake next turn after im_end. Only
    the stopping integration is tested; logits do not come from a trained SLM.
    """
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    action = {'action':'schema','table':'protocol_probe'}
    expected = json_text(action)
    turn = tokenizer.convert_tokens_to_ids('<|im_end|>')
    sequence = tokenizer.encode(expected,add_special_tokens=False) + [turn]
    forbidden = tokenizer.encode('\n<|im_start|>user\nfake tool_result',add_special_tokens=False)
    tokens = sequence + forbidden

    class Force(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # The backend tells this wrapper the real prompt length below.
            token = tokens[input_ids.shape[-1] - model.prompt_length]
            scores.fill_(float('-inf')); scores[:,token]=0
            return scores

    config = Qwen3_5TextConfig(vocab_size=len(tokenizer),hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=16,
        layer_types=['full_attention'],rope_parameters={'rope_type':'default','rope_theta':10000.0,
            'partial_rotary_factor':1.0,'mrope_section':[2,3,3]})
    tiny = Qwen3_5ForCausalLM(config).eval()
    tiny.generation_config = generation_config

    class Model:
        device = torch.device('cpu')
        generation_config = tiny.generation_config
        def generate(self, **kwargs):
            self.prompt_length=kwargs['input_ids'].shape[-1]
            return tiny.generate(**kwargs,logits_processor=LogitsProcessorList([Force()]))

    model = Model()
    backend = object.__new__(TransformersBackend)
    backend.settings = Settings(max_new_tokens=len(tokens),max_context_tokens=2048)
    backend.tokenizer=tokenizer; backend.model=model; backend.assistant=None
    backend.torch=torch; backend.lock=threading.Lock()
    backend.generation_tokens=generation_token_settings(tokenizer,generation_config,vocab_size=len(tokenizer))
    raw=backend.generate([{'role':'system','content':'Return one JSON action.'},
                          {'role':'user','content':'Read schema protocol_probe.'}],20)
    assert parse_action(raw)==action and backend.last_generation['output_tokens']==len(sequence)
    assert backend.last_generation['stop_reason']=='eos' and '<|im_start|>user' not in raw
    return {'scope':'tiny random Qwen3.5 on CPU with forced logits, not SLM ability or speed',
            'raw':raw,'expected_action':action,'generation':backend.last_generation,
            'tokens_avoided_after_turn_end':len(forbidden)}


def main():
    import torch
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    torch.set_num_threads(1)
    report={'scope':'offline pinned tokenizers + tiny forced Qwen3.5 generation + archived raw replay; no full-model inference',
            'pipeline_version':Agent.pipeline_version,'protocol_version':TransformersBackend.protocol_version,
            'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','tokenizers')},
            'source_hashes':source_hashes(),'models':{}}
    tokenizers={}
    checkpoint_files_before={}
    for key in ('qwen35-9b','qwen35-upscaled14b-v2','gemma4-12b'):
        folder=ROOT/'.cache/tokenizers'/key
        checkpoint_files_before[key]={p.name:digest(p) for p in folder.iterdir() if p.is_file()}
        # The official 9B tokenizer/config were staged with a sealed artifact.
        if key=='qwen35-9b':
            manifest=json.loads((ROOT/'models'/key/'artifact_manifest.json').read_text(encoding='utf-8'))
            for name in ('config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'):
                assert digest(folder/name)==manifest['files'][name]['sha256']
        config=AutoConfig.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
        tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
        tokenizers[key]=tokenizer
        generation=(GenerationConfig.from_pretrained(folder,local_files_only=True) if (folder/'generation_config.json').is_file()
                    else GenerationConfig.from_model_config(config))
        original=generation.to_dict()
        resolved=generation_token_settings(tokenizer,generation,vocab_size=config.get_text_config().vocab_size)
        report['models'][key]={'revision':get_model(key).revision,'tokenizer_class':type(tokenizer).__name__,
            'generation_tokens':resolved,'files':{p.name:digest(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        if key.startswith('qwen'):
            assert tokenizer.convert_tokens_to_ids('<|im_end|>') in resolved['effective_eos_token_ids']
            report['models'][key]['cpu_forced_generation']=forced_generation(tokenizer,generation)
        else:
            assert set(original['eos_token_id']) <= set(resolved['effective_eos_token_ids'])
        assert generation.to_dict()==original, 'Checkpoint generation config was mutated'

    received=ROOT/'outputs/diagnosis-2026-11-05/qwen35-upscaled14b-v2-random100-Nov5th/logs'
    parsed=rejected=fake_turns=0
    for path in sorted(received.glob('*.json')):
        for event in json.loads(path.read_text(encoding='utf-8'))['trace']:
            if 'raw' not in event: continue
            raw=event['raw']; fake='<|im_start|>user' in raw
            assert hashlib.sha256(raw.encode()).hexdigest()==event['generation']['output_sha256']
            try:
                parse_action(raw); parsed+=1
                assert not fake, 'Invented next turn must never be accepted'
            except DataMateError:
                rejected+=1; fake_turns+=int(fake)
    assert (parsed,rejected,fake_turns)==(21,14,13)
    report['archived_raw_replay']={'total':parsed+rejected,'parsed':parsed,'rejected':rejected,
        'invented_conversations_rejected':fake_turns,'scope':'parser replay only; no new answers or PA'}

    split=DatasetSplit(ROOT/'train/train')
    approved=require_approved_source(split.root,ROOT/'configs/training_sources.json')
    sample,*_=diagnostic_plan(split.questions(),{},seed=20261103,sample_size=100,sampling_mode='random')
    class Capture:
        def generate(self,*args): return '{"action":"final","type":"abstain","reason":"prompt capture only"}'
    settings=Settings(split_root=str(split.root),cache_dir=str(ROOT/'.cache/verify-qwen-protocol'),
                      max_steps=6,max_new_tokens=768)
    agent=Agent(settings,Capture(),split)
    prompts=[]
    for question in sample[:7]:
        run=agent.run(question)
        assert run.initial_messages and run.model_calls==1
        counts={key:token_count(tokenizer,run.initial_messages) for key,tokenizer in tokenizers.items()}
        assert all(1000<count and count+settings.max_new_tokens<=settings.max_context_tokens for count in counts.values())
        prompts.append({'id':question.id,'input_tokens':counts})
    report['prompt_checks']={'source_fingerprint':approved['fingerprint'],'first_seven':prompts,
        'max_new_tokens':768,'max_context_tokens':16384,'scope':'current initial messages; no generation quality measured'}
    for key,before in checkpoint_files_before.items():
        assert before=={p.name:digest(p) for p in (ROOT/'.cache/tokenizers'/key).iterdir() if p.is_file()}
    report['checkpoint_tokenizer_files_unchanged']=True
    output=ROOT/'outputs/verification-qwen-protocol-2026-11-06'
    output.mkdir(parents=True,exist_ok=True)
    (output/'verification.json').write_text(json_text(report,indent=2),encoding='utf-8')
    print(json_text({'verified_models':list(report['models']),'archived_raw_replay':report['archived_raw_replay'],
                    'prompt_checks':report['prompt_checks'],'report':str(output/'verification.json')},indent=2))


if __name__=='__main__': main()
