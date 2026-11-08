"""Stop-token and action-contract regressions; no full checkpoint is loaded."""
from copy import deepcopy
import json
import threading
from types import SimpleNamespace

import pytest

from datamate.agent.planning import validate_plan
from datamate.agent.prompts import PLAN_EXAMPLES, SYSTEM
from datamate.agent.protocol import parse_action
from datamate.agent.runner import Agent
from datamate.config import Settings
from datamate.domain import DataMateError, Question
from datamate.models.backends import TransformersBackend, generation_token_settings
from test_pipeline_v3 import business, plan, README


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    chat_template = '<|im_start|>assistant ... <|im_end|>'
    all_special_ids = [0, 1, 2, 3]
    special = {'<|endoftext|>':1, '<|im_end|>':2, '<turn|>':3}

    def __len__(self): return 32
    def convert_tokens_to_ids(self, token): return self.special.get(token)
    def convert_ids_to_tokens(self, token): return next((k for k,v in self.special.items() if v==token), None)
    def encode(self, token, **kwargs): return [self.special[token]]


@pytest.mark.parametrize('checkpoint_eos', [1, [1], [1,2], (1,2)])
def test_stop_ids_merge_and_preserve_checkpoint_and_zero_pad(checkpoint_eos):
    config = SimpleNamespace(eos_token_id=checkpoint_eos, pad_token_id=None)
    original = deepcopy(config.__dict__)
    resolved = generation_token_settings(Tokenizer(), config, vocab_size=32)
    assert resolved['effective_eos_token_ids'] == [1,2]
    assert resolved['effective_pad_token_id'] == 0
    assert resolved['chat_turn_token_ids'] == {'<|im_end|>':2}
    assert config.__dict__ == original


def test_turn_eos_is_added_even_if_a_new_tokenizer_defaults_to_document_eos():
    tokenizer = Tokenizer(); tokenizer.eos_token_id = 1
    assert generation_token_settings(tokenizer,SimpleNamespace(eos_token_id=1))['effective_eos_token_ids'] == [1,2]
    tokenizer.chat_template = 'assistant <turn|>'; tokenizer.eos_token_id = 1
    assert generation_token_settings(tokenizer,SimpleNamespace(eos_token_id=[1,3]))['effective_eos_token_ids'] == [1,3]


@pytest.mark.parametrize('bad', [-1, 32, True, '2', [1,-2]])
def test_bad_checkpoint_eos_is_not_silently_ignored(bad):
    with pytest.raises(DataMateError, match='token ID'):
        generation_token_settings(Tokenizer(), SimpleNamespace(eos_token_id=bad))


def test_missing_eos_and_non_atomic_turn_marker_fail_before_generation():
    tokenizer = Tokenizer(); tokenizer.eos_token_id = None; tokenizer.chat_template = None
    with pytest.raises(DataMateError, match='thiếu EOS'):
        generation_token_settings(tokenizer, SimpleNamespace(eos_token_id=None))
    tokenizer = Tokenizer(); tokenizer.encode = lambda *a,**kw: [2,2]
    with pytest.raises(DataMateError, match='special token đơn'):
        generation_token_settings(tokenizer, SimpleNamespace(eos_token_id=1))


@pytest.mark.parametrize('tail', ['<|im_end|>\n<|endoftext|>', '<|im_end|>  <|im_end|>\n<|endoftext|>',
                                  '<turn|><eos>', '<eos>'])
def test_parser_accepts_only_consecutive_terminal_markers(tail):
    expected = {'action':'final','type':'abstain','reason':'chưa đủ căn cứ'}
    assert parse_action(json.dumps(expected)+tail) == expected


@pytest.mark.parametrize('extra', [
    '<|im_end|>\n<|im_start|>user\ntool_result: 999<|im_end|><|endoftext|>',
    '<|im_end|>{"action":"final","type":"abstain"}<|endoftext|>',
    '\n{"action":"final","type":"abstain"}<|endoftext|>',
    '<|im_end|>some prose<|endoftext|>',
])
def test_parser_rejects_invented_turns_and_multiple_actions(extra):
    with pytest.raises(DataMateError):
        parse_action('{"action":"schema","table":"a"}'+extra)


def test_stop_settings_reach_real_backend_generate_and_its_log():
    torch = pytest.importorskip('torch')
    from transformers import BatchEncoding

    class ChatTokenizer(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs['enable_thinking'] is False
            return BatchEncoding({'input_ids':torch.tensor([[4,5]]), 'attention_mask':torch.tensor([[1,1]])})
        def decode(self, tokens, **kwargs): return '<|im_end|>' if tokens.tolist()==[2] else 'prompt'

    class Model:
        device = 'cpu'
        generation_config = SimpleNamespace(eos_token_id=1, pad_token_id=None)
        def generate(self, **kwargs):
            assert kwargs['eos_token_id']==[1,2] and kwargs['pad_token_id']==0
            assert kwargs['max_new_tokens']==8 and kwargs['do_sample'] is False
            return torch.tensor([[4,5,2]])

    backend = object.__new__(TransformersBackend)
    backend.settings = Settings(max_new_tokens=8)
    backend.lock = threading.Lock(); backend.torch = torch
    backend.tokenizer = ChatTokenizer(); backend.model = Model(); backend.assistant = None
    backend.generation_tokens = generation_token_settings(backend.tokenizer, backend.model.generation_config)
    assert backend.generate([{'role':'user','content':'hello'}],10) == '<|im_end|>'
    assert backend.last_generation['stop_reason']=='eos'
    assert backend.last_generation['effective_eos_token_ids']==[1,2]
    assert backend.model.generation_config.eos_token_id==1


def test_prompt_examples_pass_the_same_validator_used_at_inference(business):
    _,tools = business
    for example in PLAN_EXAMPLES:
        q = Question('arbitrary','shop_a',example['question'],'2025-01-31')
        assert validate_plan(example['plan'],q,tools,README)['action']=='plan'
    assert '"revenue|quantity' not in SYSTEM and '"eq|ne' not in SYSTEM


@pytest.mark.parametrize('mutation,field', [
    ({'operation':'sum|rank'},'plan.operation'),
    ({'periods':['last_week']},'plan.periods[0]'),
    ({'filters':[{'column':'product_name','op':'<=','value':'A','evidence_id':'Q'}]},'plan.filters[0].op'),
    ({'filters':[{'column':'product_name','op':'eq','value':'A'}]},'plan.filters[0]'),
    ({'operation':'rank','type':'list','limit':3,'group_by':['customer_key']},'plan.group_by'),
])
def test_plan_errors_identify_the_field_without_dropping_conditions(business, mutation, field):
    _,tools = business; question = Question('arbitrary','shop_a','Doanh thu từ đầu năm','2025-01-31')
    action = plan(question.question,**mutation); original = deepcopy(action)
    with pytest.raises(DataMateError) as caught:
        validate_plan(action,question,tools,README)
    assert field in str(caught.value) and action==original


def test_varying_malformed_outputs_cannot_consume_every_model_step(business,tmp_path):
    split,_ = business
    class Backend:
        def __init__(self): self.calls=0
        def generate(self, messages, remaining_seconds):
            self.calls += 1
            return 'bad output variant '+str(self.calls)
    backend = Backend()
    result = Agent(Settings(split_root=str(split.root),cache_dir=str(tmp_path/'run'),max_steps=6),backend).run(
        Question('arbitrary','shop_a','Doanh thu từ đầu năm','2025-01-31'))
    assert result.model_calls==2 and result.answer.type=='abstain'
    assert result.trace[-1]['loop_stop_kind']=='consecutive_action_parse'


def test_a_valid_action_resets_consecutive_parse_failures(business,tmp_path):
    split,_ = business; text='Doanh thu từ đầu năm'
    outputs = iter(['bad one','{"action":"period","period":{"kind":"today"}}','bad two',
                    json.dumps(plan(text))+'<|im_end|>\n<|endoftext|>'])
    class Backend:
        def generate(self,*args): return next(outputs)
    result = Agent(Settings(split_root=str(split.root),cache_dir=str(tmp_path/'run')),Backend()).run(
        Question('arbitrary','shop_a',text,'2025-01-31'))
    assert result.model_calls==4 and result.answer.answer==400


def test_plan_feedback_allows_model_to_repair_the_specific_field(business,tmp_path):
    split,_=business; text='Doanh thu từ đầu năm'
    invalid=plan(text); invalid['period']=invalid.pop('periods')[0]
    class Backend:
        def __init__(self): self.calls=0
        def generate(self,messages,*args):
            self.calls+=1
            if self.calls==1: return json.dumps(invalid)+'<|im_end|><|endoftext|>'
            feedback=json.loads(messages[-1]['content'])['tool_result']
            assert feedback['error_kind']=='question_plan' and 'period' in feedback['error']
            assert 'periods là list' in feedback['repair']
            return json.dumps(plan(text))+'<|im_end|>'
    result=Agent(Settings(split_root=str(split.root),cache_dir=str(tmp_path/'run')),Backend()).run(
        Question('arbitrary','shop_a',text,'2025-01-31'))
    assert result.model_calls==2 and result.answer.answer==400
    assert 'period' in result.trace[0]['llm_action'] and 'periods' not in result.trace[0]['llm_action']


@pytest.mark.parametrize('key', ['qwen35-9b','qwen35-upscaled14b-v2'])
def test_both_qwen_presets_keep_fast_profile(key):
    from pathlib import Path
    settings=Settings.load(Path(__file__).resolve().parents[1]/'configs'/f'{key}.json')
    assert settings.max_steps==6 and settings.max_new_tokens==768 and settings.question_timeout_seconds==90
    assert settings.require_plan and settings.auto_execute_plan and not settings.adaptive_preparation
