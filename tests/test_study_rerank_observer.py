from types import SimpleNamespace

import pytest

from scripts.study_operator.rerank_observer import observe, report


def test_observation_preserves_call_result_identity_and_restores_method():
    result = [SimpleNamespace(chunk_id='two', score=2.75), SimpleNamespace(chunk_id='one', score=-1.25)]
    calls = []

    class Reranker:
        def rerank(self, *args, **kwargs):
            calls.append((args, kwargs))
            return result

    reranker = Reranker()
    pipeline = SimpleNamespace(reranker=reranker)
    records = []
    with observe(pipeline, records):
        assert reranker.rerank('question', result, top_k=2) is result
    assert calls == [(('question', result), {'top_k': 2})]
    assert 'rerank' not in vars(reranker)
    assert records == [{'chunk_id': 'two', 'rerank_score': 2.75}, {'chunk_id': 'one', 'rerank_score': -1.25}]
    selected = report(records, SimpleNamespace(retrieved_chunks=[{'chunk_id': 'one'}]))
    assert selected['selected'] == [{'chunk_id': 'one', 'rerank_score': -1.25}]
    assert [row.chunk_id for row in result] == ['two', 'one']


def test_exception_restores_previous_instance_override():
    def failed(*args, **kwargs):
        raise RuntimeError('synthetic failure')
    reranker = SimpleNamespace(rerank=failed)
    with pytest.raises(RuntimeError):
        with observe(SimpleNamespace(reranker=reranker), []):
            reranker.rerank()
    assert reranker.rerank is failed


def test_control_without_retrieval_has_empty_observation():
    records = []
    with observe(SimpleNamespace(), records):
        pass
    assert report(records, SimpleNamespace(retrieved_chunks=[])) == {
        'candidates': [], 'selected': [], 'reranking_observed': False}
