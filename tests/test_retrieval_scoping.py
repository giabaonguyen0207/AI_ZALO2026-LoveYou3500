from types import SimpleNamespace
import threading
import sqlite3
import time

import pytest

from datamate.agent.tools import BusinessTools
from datamate.agent.protocol import answer_from_query
from datamate.config import Settings
from datamate.data.rules import read_policy
from datamate.domain import DataMateError
from datamate.models.registry import inventory_from_settings
from datamate.retrieval.core import Chunk, ContextRetriever, sections
from datamate.agent.planning import _forced_measure, _resolve_plan_period


def test_scalar_projection_uses_actual_column_but_multicolumn_stays_explicit():
    result = {'columns': ['round_half_up(numerator.value*100.0/denominator.value,2)'], 'rows': [[23.77]]}
    assert answer_from_query('percent', result, ['value']).answer == 23.77
    with pytest.raises(DataMateError):
        answer_from_query('number', {'columns': ['a', 'b'], 'rows': [[1, 2]]}, ['value'])


def test_rag_inventory_and_validation():
    settings = Settings(rag={'mode': 'dense_rerank'})
    settings.validate()
    names = [item['key'] for item in inventory_from_settings(settings)['models']]
    assert names == ['qwen35-9b', 'vietnamese-embedding', 'vietnamese-reranker']
    assert inventory_from_settings(settings)['total_parameters'] < 15_000_000_000
    assert len(inventory_from_settings(Settings())['models']) == 1
    with pytest.raises(DataMateError):
        Settings(rag={'mode': 'dense_rerank'}, backend='demo').validate()
    with pytest.raises(DataMateError):
        Settings(rag={'mode': 'dense', 'embedding_key': 'qwen35-9b'}).validate()


def test_corpus_keeps_business_scope_and_source(tmp_path):
    split = SimpleNamespace(rulebook='# Quy tắc\nTuần từ thứ Hai.', rulebook_hash='rule-hash',
                            readme=lambda business: f'# {business}\nMã đơn nối chi tiết.',
                            business_path=lambda business: tmp_path / business)
    retriever = ContextRetriever.__new__(ContextRetriever)
    retriever.settings = SimpleNamespace(mode='dense', candidate_k=10, context_k=10)
    retriever.split, retriever.index, retriever.reranker = split, {}, None
    retriever.lock = threading.RLock()
    retriever._encode = lambda texts: [[1.0, 0.0] for _ in texts]
    catalog = SimpleNamespace(fingerprint='raw-hash', tables=[{'name': 'raw_sales', 'file': 'sales.csv',
                                                                'columns': ['Mã đơn', 'Số lượng']}])
    first, _ = retriever.retrieve('Mã đơn', 'biz_a', catalog, 10)
    second, _ = retriever.retrieve('Mã đơn', 'biz_b', catalog, 10)
    assert any('biz_a' in item['text'] for item in first)
    assert all('biz_b' not in item['text'] for item in first)
    assert any('biz_b' in item['text'] for item in second)
    assert any(item['source'] == 'data/biz_b/sales.csv' for item in second)
    assert len(retriever.index) == 2


def test_sections_bound_large_documents_without_losing_heading():
    chunks = sections('# Kho\n' + ('Quy tắc lô hàng.\n' * 300), 'rulebook.md', 'split')
    assert len(chunks) > 1
    assert all(chunk.text.startswith('# Kho') for chunk in chunks)


def test_retrieval_boost_prefers_explicit_month_sheet_and_cancel_rule():
    sheet = Chunk('Bang raw_orders__t04_2026; cot: Trang thai', 'data/biz/orders.xlsx', 'biz', 'a')
    neighbour = Chunk('Bang raw_orders__t03_2026; cot: Trang thai', 'data/biz/orders.xlsx', 'biz', 'b')
    rule = Chunk('# Don hang\nTy le don huy trong ky.', 'rulebook.md', 'split', 'c')
    question = 'thang 4/2026 co bao nhieu % don bi huy'
    assert ContextRetriever._question_specific_boost(question, sheet) > ContextRetriever._question_specific_boost(question, neighbour)
    assert ContextRetriever._question_specific_boost(question, rule) > 0


def test_short_vietnamese_sales_money_intent_is_not_debt():
    assert _forced_measure('hnay cua hang ban dc bao nhiu tien vay') == 'revenue'
    assert _forced_measure('hom nay con no bao nhieu tien') is None


def test_resolved_iso_period_without_kind_is_normalized_to_range():
    policy = read_policy('')
    period = _resolve_plan_period({'start': '2026-04-01', 'end': '2026-04-30', 'inclusive': True,
                                   'as_of_date': '2026-07-13'}, '2026-07-13', policy)
    assert period['start'] == '2026-04-01' and period['end'] == '2026-04-30'


def test_rag_environment_overrides_nested_config(monkeypatch, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text('{"rag":{"mode":"none"}}', encoding='utf-8')
    monkeypatch.setenv('DATAMATE_RAG_MODE', 'dense')
    assert Settings.load(config).rag.mode == 'dense'


def test_fuzzy_lookup_reads_transaction_names_when_dimension_empty(tmp_path):
    path = tmp_path / 'facts.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE dm_sales_all (customer_key TEXT, customer_id TEXT, customer_name TEXT)')
        db.execute('CREATE TABLE dm_orders_all (customer_key TEXT, customer_id TEXT, customer_name TEXT)')
        db.execute('CREATE TABLE dm_customers (customer_id TEXT, customer_name TEXT)')
        db.execute("INSERT INTO dm_sales_all VALUES ('khach le', NULL, 'Khách lẻ')")
    catalog = SimpleNamespace(database=path, semantic={'policy': read_policy('').to_dict(), 'bindings': []})
    result = BusinessTools(catalog, '', '2026-05-01').lookup('customer', 'Khách lz')
    assert result['match_kind'] == 'fuzzy_candidates'
    assert result['candidates'][0]['key'] == 'khach le'
    assert result['ambiguous'] is False
    with pytest.raises(DataMateError):
        BusinessTools(catalog, '', '2026-05-01')._entity('customer', 'Khách lz')


def test_retrieval_resumes_partial_cpu_index():
    retriever = ContextRetriever.__new__(ContextRetriever)
    retriever.settings = SimpleNamespace(mode='dense', candidate_k=8, context_k=2)
    retriever.split = SimpleNamespace(rulebook_hash='rule')
    retriever.index, retriever.reranker, retriever.lock = {}, None, threading.RLock()
    retriever._corpus = lambda business, catalog: [Chunk(str(i), 'source', business, 'digest') for i in range(9)]
    encoded = []

    def fake_encode(texts):
        time.sleep(0.03)
        encoded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    retriever._encode = fake_encode
    catalog = SimpleNamespace(fingerprint='raw')
    _, first = retriever.retrieve('query', 'biz_a', catalog, 0.01)
    assert first['phase'] == 'index' and first['indexed_chunks'] == 4
    chunks, second = retriever.retrieve('query', 'biz_a', catalog, 1)
    assert second['status'] == 'ok' and len(chunks) == 2
    assert encoded == [str(i) for i in range(9)] + ['query']
