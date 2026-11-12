import json
from types import SimpleNamespace

import pytest

from datamate.config import Settings
from datamate.domain import DataMateError
from datamate.models.backends import DemoBackend, TransformersBackend, build_backend
from datamate.models.chat_format import context_budget, encode_messages, input_ids
from datamate.models.contracts import GenerationTimeoutError
from datamate.models.runtime import runtime_identity
from tools.check_context import check


class Tokenizer:
    chat_template = 'test template'

    def apply_chat_template(self, messages, **options):
        assert options == dict(tokenize=True, add_generation_prompt=True, return_dict=True, enable_thinking=False)
        # Includes framing/generation tokens; not just message content length.
        return {'input_ids': [1, 2, 3, 4]}


def test_counter_and_preflight_share_the_generation_formatter():
    messages = [{'role': 'user', 'content': 'xin chào'}]
    backend = object.__new__(TransformersBackend)
    backend.tokenizer = Tokenizer()
    assert backend.count_tokens(messages) == len(input_ids(encode_messages(backend.tokenizer, messages))) == 4
    assert check(backend.tokenizer, messages, 2, 6)['remaining_tokens'] == 0
    with pytest.raises(DataMateError, match='input=4, output_reserved=3, limit=6'):
        check(backend.tokenizer, messages, 3, 6)


@pytest.mark.parametrize('values', [(1, True, 5), (1, 0, 5), (-1, 1, 5), (1, 1, 1)])
def test_invalid_context_budget_is_rejected(values):
    with pytest.raises(DataMateError):
        context_budget(*values)


def test_demo_is_an_explicit_non_model_and_does_not_fake_token_counts():
    backend = DemoBackend()
    assert not backend.counts_as_model and not backend.supports_token_budget
    assert json.loads(backend.generate([], 1, max_new_tokens=8))['type'] == 'abstain'
    with pytest.raises(DataMateError, match='tokenizer'):
        backend.count_tokens([])
    with pytest.raises(GenerationTimeoutError):
        backend.generate([], 0)
    json.dumps(backend.runtime_info())
    backend.close()
    backend.close()


def test_identity_is_stable_and_engine_options_invalidate_caches():
    baseline = runtime_identity(Settings())
    assert baseline == runtime_identity(Settings())
    assert baseline != runtime_identity(Settings(backend='demo'))
    assert baseline != runtime_identity(Settings(max_context_tokens=32768))
    assert 'chat_format.py' in baseline['sources']
    json.dumps(baseline)


def test_unknown_backend_never_silently_loads_transformers():
    with pytest.raises(DataMateError, match='chưa được triển khai'):
        build_backend(Settings(backend='typo'))


def test_output_budget_is_checked_before_weights_load():
    with pytest.raises(DataMateError, match='output budget'):
        Settings(max_context_tokens=2048, max_new_tokens=2048).validate()


def test_transformers_close_is_idempotent_without_touching_other_models():
    import threading
    backend = object.__new__(TransformersBackend)
    backend.lock = threading.Lock()
    backend._closed = False
    backend.settings = Settings()
    backend.inventory = {'models': []}
    backend.tokenizer = Tokenizer()
    backend.generation_tokens = {}
    backend.model = SimpleNamespace()
    backend.assistant = SimpleNamespace()
    backend.close()
    backend.close()
    assert backend.model is backend.assistant is None
    assert json.loads(json.dumps(backend.runtime_info()))['closed'] is True
    with pytest.raises(DataMateError, match='đã đóng'):
        backend.generate([], 1)
