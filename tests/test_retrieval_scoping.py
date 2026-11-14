"""Retained RAG features; business heuristics are intentionally not part of RAG."""
import pytest

from datamate.config import Settings
from datamate.domain import DataMateError, Question
from datamate.models.registry import inventory_from_settings
from datamate.retrieval.core import sections
from test_rag_integration import Encoder, Tokenizer, retriever


def test_rag_inventory_and_validation():
    settings = Settings(rag={'mode': 'dense_rerank'}).validate()
    assert [m['key'] for m in inventory_from_settings(settings)['models']] == [
        'qwen35-9b', 'vietnamese-embedding', 'vietnamese-reranker']
    assert inventory_from_settings(settings)['total_parameters'] == 10_788_614_897
    assert len(inventory_from_settings(Settings())['models']) == 1
    with pytest.raises(DataMateError):
        Settings(rag={'mode': 'dense_rerank'}, backend='demo').validate()
    with pytest.raises(DataMateError):
        Settings(rag={'mode': 'dense', 'embedding_key': 'qwen35-9b'}).validate()
    with pytest.raises(DataMateError, match='15B'):
        inventory_from_settings(Settings(model_key='qwen35-upscaled14b-v2', rag={'mode': 'dense_rerank'}).validate())


def test_corpus_and_query_do_not_cross_business(split_root, tmp_path, monkeypatch):
    for business in ('shop_a', 'shop_b'):
        (split_root / f'data/{business}/README.md').write_text(f'# {business}\nPrivate business description.', encoding='utf-8')
    obj, ingestor = retriever(tmp_path, split_root, monkeypatch)
    obj.prepare(['shop_a', 'shop_b'], ingestor=ingestor, remaining_seconds=10)
    a, _ = obj.retrieve('description', 'shop_a', ingestor.catalog('shop_a', normalize=False), 10)
    b, _ = obj.retrieve('description', 'shop_b', ingestor.catalog('shop_b', normalize=False), 10)
    assert any('shop_a' in c['text'] for c in a) and all('shop_b' not in c['text'] for c in a)
    assert any('shop_b' in c['text'] for c in b) and all('shop_a' not in c['text'] for c in b)
    assert all(c['scope'] in {'split', 'shop_a'} for c in a)
    obj.close()


def test_sections_use_embedding_tokens_and_retain_heading():
    tokenizer = Tokenizer()
    chunks = sections('# Kho\n' + ('Quy tắc lô hàng.\n' * 300), 'rulebook.md', 'split',
                      tokenizer=tokenizer, chunk_tokens=64, overlap=8)
    assert len(chunks) > 1
    assert all(c.text.startswith('# Kho') for c in chunks)
    assert all(len(tokenizer.encode(c.text)) <= 64 for c in chunks)
    assert len({c.metadata['chunk_id'] for c in chunks}) == len(chunks)


def test_rag_environment_overrides_nested_config(monkeypatch, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{"rag":{"mode":"none"}}', encoding='utf-8')
    monkeypatch.setenv('DATAMATE_RAG_MODE', 'dense')
    assert Settings.load(config).rag.mode == 'dense'


def test_none_empty_and_fallback_preserve_baseline_prompt(split_root, tmp_path, scripted_backend):
    from datamate.agent.runner import Agent

    class Empty:
        def __init__(self, status):
            self.status = status
        def retrieve(self, *args):
            return [], {'status': self.status}
        def close(self):
            pass

    question = Question('q', 'shop_a', 'Tổng tiền?', '2026-06-02')
    expected = None
    for mode, status in [('none', 'disabled'), ('dense', 'empty'), ('dense', 'fallback')]:
        settings = Settings(split_root=str(split_root), cache_dir=str(tmp_path / mode),
                            schema_mode='code_only', rag={'mode': mode}).validate()
        backend = scripted_backend([{'action': 'final', 'type': 'abstain'}])
        run = Agent(settings, backend, retriever=Empty(status)).run(question)
        assert run.answer.type == 'abstain'
        if expected is None:
            expected = run.initial_messages
        else:
            assert run.initial_messages == expected
