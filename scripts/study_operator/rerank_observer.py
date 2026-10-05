"""Runner-only observation. Return the exact object produced by the reranker."""
from contextlib import contextmanager


@contextmanager
def observe(pipeline, records):
    reranker = getattr(pipeline, 'reranker', None)
    if reranker is None:
        yield
        return
    original = reranker.rerank
    had_override = 'rerank' in vars(reranker)
    previous_override = vars(reranker).get('rerank')

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        records.extend(dict(chunk_id=row.chunk_id, rerank_score=float(row.score)) for row in result)
        return result

    reranker.rerank = capture
    try:
        yield
    finally:
        if had_override:
            reranker.rerank = previous_override
        else:
            del reranker.rerank


def report(records, response):
    scores = {row['chunk_id']: row['rerank_score'] for row in records}
    return dict(candidates=records, selected=[dict(chunk_id=row['chunk_id'], rerank_score=scores.get(row['chunk_id']))
                                             for row in response.retrieved_chunks],
                reranking_observed=bool(records))
