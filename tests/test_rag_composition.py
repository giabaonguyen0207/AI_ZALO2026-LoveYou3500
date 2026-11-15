from types import SimpleNamespace
import pytest

from datamate.config import Settings


def test_prepare_is_inside_startup_budget_before_generator_load(split_root, tmp_path, monkeypatch):
    import datamate.application as app
    now, order = [0.0], []
    monkeypatch.setattr(app.time, 'monotonic', lambda: now[0])

    class Retriever:
        def prepare(self, business_ids, *, ingestor, remaining_seconds):
            assert business_ids == ['shop_a'] and remaining_seconds == 1
            order.append('prepare')
            now[0] += 0.25

        def close(self):
            order.append('close_retriever')

    def backend(settings, *, remaining_seconds):
        order.append('backend')
        assert remaining_seconds == pytest.approx(0.75)
        return SimpleNamespace(close=lambda: order.append('close_backend'))

    monkeypatch.setattr(app, 'build_retriever', lambda *args: Retriever())
    monkeypatch.setattr(app, 'build_backend', backend)
    settings = Settings(backend='vllm', split_root=str(split_root), cache_dir=str(tmp_path / 'cache'),
                        result_dir=str(tmp_path / 'out'), rag={'mode': 'dense'})
    agent = app.create_agent(settings, business_ids=['shop_a'], remaining_seconds=1)
    assert order == ['prepare', 'backend']
    agent.close()
    assert order[-2:] == ['close_backend', 'close_retriever']


def test_failed_generator_initialization_closes_prepared_retriever(split_root, tmp_path, monkeypatch):
    import datamate.application as app
    closed = []
    retriever = SimpleNamespace(prepare=lambda *args, **kwargs: None, close=lambda: closed.append(True))
    monkeypatch.setattr(app, 'build_retriever', lambda *args: retriever)

    def fail(*args, **kwargs):
        raise RuntimeError('engine initialization failed')

    monkeypatch.setattr(app, 'build_backend', fail)
    settings = Settings(backend='vllm', split_root=str(split_root), cache_dir=str(tmp_path / 'cache'),
                        result_dir=str(tmp_path / 'out'), rag={'mode': 'dense'})
    with pytest.raises(RuntimeError, match='initialization'):
        app.create_agent(settings, business_ids=['shop_a'], remaining_seconds=1)
    assert closed == [True]


def test_full_inventory_paths_and_extras_are_shared_by_runtime_tools():
    from datamate.models.registry import model_paths, inference_extras, inventory_from_settings
    settings = Settings(model_dir='/main', rag={'mode': 'dense_rerank',
        'embedding_dir': '/embedding', 'reranker_dir': '/reranker'}).validate()
    assert model_paths(settings) == {'qwen35-9b': '/main', 'vietnamese-embedding': '/embedding', 'vietnamese-reranker': '/reranker'}
    assert inference_extras(settings) == 'llm,rag'
    assert inventory_from_settings(settings)['total_parameters'] == 10_788_614_897
    settings.backend = 'vllm'
    assert inference_extras(settings) == 'vllm,rag'
    settings.rag.mode = 'none'
    assert model_paths(settings) == {'qwen35-9b': '/main'} and inference_extras(settings) == 'vllm'
