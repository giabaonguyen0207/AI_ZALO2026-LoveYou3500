"""Integration contracts; no full checkpoint or network is used."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from datamate.config import Settings
from datamate.data.ingest import Ingestor
from datamate.data.split import DatasetSplit
from datamate.domain import DataMateError, Question


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode('utf-8'))


class Encoder:
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []
        self.closed = False

    def load(self):
        pass

    def encode(self, texts, *, remaining_seconds):
        assert remaining_seconds > 0 and not self.closed
        self.calls.extend(texts)
        result = np.zeros((len(texts), 1024), dtype=np.float32)
        result[:, 0] = 1
        return result

    def close(self):
        self.closed = True


def retriever(tmp_path, split_root, monkeypatch, encoder=None):
    import datamate.retrieval.core as core
    monkeypatch.setattr(core, 'verify_artifact', lambda *args: {'source': {'revision': 'fixture'}, 'files': {}})
    settings = Settings(split_root=str(split_root), cache_dir=str(tmp_path / 'cache'),
                        rag={'mode': 'dense', 'context_k': 8, 'candidate_k': 12}).validate()
    split = DatasetSplit(split_root)
    obj = core.ContextRetriever(settings.rag, split, cache_dir=settings.cache_dir, encoder=encoder or Encoder())
    return obj, Ingestor(split, settings.cache_dir)


def test_actual_generator_tokens_limit_context_and_shrink_with_history():
    from datamate.retrieval.context import fit_context
    settings = Settings(max_context_tokens=4096, max_new_tokens=5,
                        rag={'mode': 'dense', 'context_tokens': 20}).validate()
    settings.max_context_tokens = 30  # Small fixture context, after normal validation.
    backend = SimpleNamespace(count_tokens=lambda rows: sum(len(m['content']) for m in rows))
    chunks = [{'text': 'A' * 8}, {'text': 'B' * 8}]
    history = []

    def build(context):
        text = 'Q' + ('<rag>' + ''.join(c['text'] for c in context) + '</rag>' if context else '')
        return [{'role': 'user', 'content': text}, *history]

    messages, selected, diagnostics = fit_context(chunks, build_messages=build, backend=backend, settings=settings)
    assert selected == [chunks[0]]
    assert diagnostics['context_tokens'] == 19
    history.append({'role': 'assistant', 'content': 'H' * 10})
    messages, selected, diagnostics = fit_context(selected, build_messages=build, backend=backend, settings=settings)
    assert selected == [] and messages == [{'role': 'user', 'content': 'Q'}, history[0]]


def test_empty_context_does_not_require_a_tokenizer_or_modify_messages():
    from datamate.retrieval.context import fit_context
    original = [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'Q'}]
    messages, selected, _ = fit_context([], build_messages=lambda context: original,
                                      backend=object(), settings=Settings())
    assert messages == original and selected == []


@pytest.mark.parametrize('vectors', [np.zeros((1, 1024)), np.ones((1, 512)), np.full((1, 1024), np.nan), np.ones((2, 1024))])
def test_encoder_output_rejects_zero_nan_wrong_dimension_or_cardinality(vectors):
    from datamate.retrieval.encoders import validate_vectors
    with pytest.raises(DataMateError):
        validate_vectors(vectors, expected_rows=1)


def test_cache_reuses_vectors_and_invalidates_changed_sources(split_root, tmp_path, monkeypatch):
    first_encoder = Encoder()
    first, ingestor = retriever(tmp_path, split_root, monkeypatch, first_encoder)
    first.prepare(['shop_a'], ingestor=ingestor, remaining_seconds=10)
    assert first_encoder.calls
    first.close()
    second_encoder = Encoder()
    second, ingestor = retriever(tmp_path, split_root, monkeypatch, second_encoder)
    second.prepare(['shop_a'], ingestor=ingestor, remaining_seconds=10)
    assert second_encoder.calls == []
    (split_root / 'data/shop_a/README.md').write_text('A new rule', encoding='utf-8')
    third_encoder = Encoder()
    third, ingestor = retriever(tmp_path, split_root, monkeypatch, third_encoder)
    third.prepare(['shop_a'], ingestor=ingestor, remaining_seconds=10)
    assert third_encoder.calls
    second.close()
    third.close()


def test_nested_schema_provenance_is_real_source_bytes(split_root, tmp_path, monkeypatch):
    from datamate.models.artifacts import sha256
    source = split_root / 'data/shop_a/nested/sales.csv'
    source.parent.mkdir()
    source.write_text('id,amount\na,10\n', encoding='utf-8-sig')
    obj, ingestor = retriever(tmp_path, split_root, monkeypatch)
    docs = obj._corpus('shop_a', ingestor.catalog('shop_a', normalize=False))
    chunks = [c.to_dict() for c in docs if c.source.endswith('nested/sales.csv')]
    assert chunks and all(c['source_sha256'] == sha256(source) for c in chunks)
    assert all(c['scope'] == 'shop_a' and c['source'].startswith('data/shop_a/') for c in chunks)
    obj.close()


def test_corrupted_cached_vectors_rebuild_instead_of_failing_startup(split_root, tmp_path, monkeypatch):
    from datamate.models.artifacts import sha256
    obj, ingestor = retriever(tmp_path,split_root,monkeypatch)
    obj.prepare(['shop_a'],ingestor=ingestor,remaining_seconds=10)
    obj.close()
    manifest = next((tmp_path/'cache').rglob('index_manifest.json'))
    vector_file = manifest.parent/'vectors.npy'
    np.save(vector_file,np.zeros((1,512)),allow_pickle=False)
    data = json.loads(manifest.read_text())
    data['files']['vectors.npy'] = sha256(vector_file)
    manifest.write_text(json.dumps(data))
    encoder = Encoder()
    obj, ingestor = retriever(tmp_path,split_root,monkeypatch,encoder)
    obj.prepare(['shop_a'],ingestor=ingestor,remaining_seconds=10)
    assert encoder.calls and obj.runtime_info()['cache_builds']==1
    obj.close()


def test_closed_retriever_cannot_reload_or_serve_requests(split_root, tmp_path, monkeypatch):
    from datamate.retrieval.contracts import RetrievalFatalError
    obj, ingestor = retriever(tmp_path, split_root, monkeypatch)
    obj.close()
    obj.close()
    with pytest.raises(RetrievalFatalError):
        obj.retrieve('q', 'shop_a', ingestor.catalog('shop_a', normalize=False), 10)


def test_retrieval_fatal_propagates_through_agent(split_root, tmp_path):
    from datamate.agent.runner import Agent
    from datamate.retrieval.contracts import RetrievalFatalError

    class FailedRetriever:
        def retrieve(self, *args):
            raise RetrievalFatalError('encoder OOM')

    settings = Settings(split_root=str(split_root), cache_dir=str(tmp_path / 'cache'),
                        schema_mode='code_only', rag={'mode': 'dense'}).validate()
    with pytest.raises(RetrievalFatalError, match='OOM'):
        Agent(settings, object(), retriever=FailedRetriever()).run(Question('q', 'shop_a', 'q', '2026-06-02'))


def test_loader_failure_is_atomic_and_reranker_is_retried(tmp_path, monkeypatch):
    from datamate.retrieval.encoders import VietnameseEncoders
    from datamate.retrieval.contracts import RetrievalFatalError
    import transformers
    calls = []
    model = SimpleNamespace(to=lambda *args, **kwargs: model, eval=lambda: model)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', lambda *args, **kwargs: model)

    def load_reranker(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError('broken checkpoint')
        return model

    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, 'from_pretrained', load_reranker)
    settings = Settings(rag={'mode': 'dense_rerank', 'device': 'cpu'}).validate()
    obj = VietnameseEncoders(settings.rag)
    with pytest.raises(RetrievalFatalError):
        obj.load()
    assert obj.encoder is None and obj.reranker is None
    obj.load()
    assert obj.reranker is model and len(calls) == 2
    obj.close()
