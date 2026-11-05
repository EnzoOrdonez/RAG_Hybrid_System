"""P4b: tests that kill surviving mutants in the modules that compute reported results.

Modules: src/evaluation/retrieval_metrics.py, src/embedding/index/hybrid_index.py (RRF/linear fusion),
src/retrieval/hybrid_retriever.py (fusion routing), src/evaluation/hallucination_metrics.py (NLI faithfulness
aggregation) and src/evaluation/statistical_analysis.py. See docs/review/2026-11-05/p4/SALUD_TESTS.md.
"""
import math
from types import SimpleNamespace

import pytest

from src.evaluation import retrieval_metrics as rm
from src.evaluation import statistical_analysis as sa


# --- retrieval_metrics --------------------------------------------------------------------------

@pytest.mark.parametrize('fn', [lambda r, g: rm.recall_at_k(r, g, 3), lambda r, g: rm.precision_at_k(r, g, 3),
                                rm.mrr, lambda r, g: rm.ndcg_at_k(r, g, 3), rm.map_score])
def test_empty_inputs_score_exactly_zero(fn):
    assert fn([], ['a']) == 0.0
    assert fn(['a'], []) == 0.0


@pytest.mark.parametrize('k', [0, -1, 1.0, True])
def test_k_must_be_a_positive_int(k):
    with pytest.raises(ValueError):
        rm.precision_at_k(['a'], ['a'], k)


def test_k_one_is_valid():
    assert rm.precision_at_k(['a', 'b'], ['a'], 1) == 1.0


def test_ndcg_values_with_rank_discount():
    assert rm.ndcg_at_k(['x', 'a'], ['a'], 2) == pytest.approx((1 / math.log2(3)) / 1.0)
    assert rm.ndcg_at_k(['x', 'y'], ['a'], 2) == 0.0


def test_compute_all_default_k_values():
    keys = set(rm.compute_all_retrieval_metrics(['a', 'b'], ['a']))
    for k in (5, 10, 20):
        assert {f'recall@{k}', f'precision@{k}'} <= keys
    assert not any(key.endswith('@6') or key.endswith('@11') for key in keys)


# --- hybrid_index fusion -------------------------------------------------------------------------

def index_with(bm25, dense):
    from src.embedding.index.hybrid_index import HybridIndex
    index = object.__new__(HybridIndex)
    calls = {}
    def search_bm25(query, top_k):
        calls['bm25'] = (query, top_k)
        return bm25
    def search_dense(query, top_k):
        calls['dense'] = (query, top_k)
        return dense
    index.search_bm25, index.search_dense, index.calls = search_bm25, search_dense, calls
    return index


def test_rrf_scores_are_sum_of_reciprocal_ranks_with_k_60():
    index = index_with([('a', 9.0), ('b', 5.0)], [('b', .9), ('c', .8)])
    result = dict(index.search_hybrid('q'))
    assert result['a'] == pytest.approx(1 / 61)
    assert result['b'] == pytest.approx(1 / 62 + 1 / 61)
    assert result['c'] == pytest.approx(1 / 62)
    assert [cid for cid, _ in index.search_hybrid('q')] == ['b', 'a', 'c']
    assert index.calls == {'bm25': ('q', 50), 'dense': ('q', 50)}


def test_rrf_respects_top_k_and_custom_k():
    index = index_with([('a', 1.0), ('b', .5), ('c', .1)], [])
    assert [c for c, _ in index.search_hybrid('q', top_k=2, rrf_k=10)] == ['a', 'b']
    assert dict(index.search_hybrid('q', rrf_k=10))['a'] == pytest.approx(1 / 11)


def test_default_top_k_is_20():
    index = index_with([(f'd{i}', float(100 - i)) for i in range(30)], [])
    assert len(index.search_hybrid('q')) == 20


def test_bm25_leg_gets_expanded_query_and_dense_keeps_raw():
    index = index_with([], [])
    index.search_hybrid('raw', bm25_query='expanded')
    assert index.calls['bm25'][0] == 'expanded' and index.calls['dense'][0] == 'raw'


def test_unknown_fusion_is_rejected():
    with pytest.raises(ValueError):
        index_with([], []).search_hybrid('q', fusion='max')


def test_linear_fusion_weights_normalized_scores():
    index = index_with([('a', 10.0), ('b', 0.0)], [('b', 1.0), ('a', 0.0)])
    result = dict(index.search_hybrid('q', fusion='linear', alpha=0.3))
    assert result['a'] == pytest.approx(0.3) and result['b'] == pytest.approx(0.7)
    default = dict(index.search_hybrid('q', fusion='linear'))
    assert default['a'] == pytest.approx(0.5)


def test_normalize_scores_edges():
    from src.embedding.index.hybrid_index import HybridIndex
    assert HybridIndex._normalize_scores([]) == []
    assert HybridIndex._normalize_scores([('a', 3.0), ('b', 3.0)]) == [('a', 1.0), ('b', 1.0)]
    assert HybridIndex._normalize_scores([('a', 4.0), ('b', 2.0), ('c', 3.0)]) == [('a', 1.0), ('b', 0.0), ('c', 0.5)]


# --- hybrid_retriever routing --------------------------------------------------------------------

class FakeIndex:
    def __init__(self, n=8, providers=None):
        self.calls, self.n, self.providers = [], n, providers or {}

    def search_hybrid(self, **kwargs):
        self.calls.append(kwargs)
        return [(f'c{i}', 1.0 / (i + 1)) for i in range(self.n)]

    def get_chunk(self, cid):
        return {'text': 't' + cid, 'cloud_provider': self.providers.get(cid, 'aws')}


def processor(bm25_query='expanded', provider_filter=None):
    return SimpleNamespace(process=lambda q: SimpleNamespace(bm25_query=bm25_query, provider_filter=provider_filter))


def retriever(**kw):
    from src.retrieval.hybrid_retriever import HybridRetriever
    index = kw.pop('index', FakeIndex())
    return HybridRetriever(index, query_processor=kw.pop('qp', processor()), **kw), index


def test_retriever_defaults_and_rrf_routing():
    r, index = retriever()
    assert r.alpha == 0.5 and r.rrf_k == 60 and r.fusion_method == 'rrf'
    out = r.search('raw')
    call = index.calls[-1]
    assert call['query'] == 'raw' and call['bm25_query'] is None and call['fusion'] == 'rrf'
    assert call['top_k'] == 50 and call['top_k_candidates'] == 50 and call['rrf_k'] == 60 and call['alpha'] == 0.5
    assert [c.chunk_id for c in out] == ['c0', 'c1', 'c2', 'c3', 'c4']
    assert out[0].retrieval_method == 'hybrid_rrf' and out[0].chunk_text == 'tc0'


def test_expansion_reaches_only_the_bm25_leg():
    r, index = retriever()
    r.search('raw', use_expansion=True)
    assert index.calls[-1]['query'] == 'raw' and index.calls[-1]['bm25_query'] == 'expanded'


def test_linear_fusion_uses_processed_query_for_both_legs():
    r, index = retriever()
    r.search('raw', fusion='linear')
    assert index.calls[-1]['query'] == 'expanded' and index.calls[-1]['bm25_query'] is None


def test_explicit_zero_alpha_is_not_replaced_by_default():
    r, index = retriever()
    r.search('raw', alpha=0.0)
    assert index.calls[-1]['alpha'] == 0.0


def test_provider_filter_keeps_only_matching_chunks():
    index = FakeIndex(providers={'c0': 'gcp', 'c1': 'aws', 'c2': 'gcp'})
    r, _ = retriever(index=index, qp=processor(provider_filter=['gcp']))
    assert [c.chunk_id for c in r.search('raw', top_k=10)] == ['c0', 'c2']


def test_reranker_used_only_when_requested():
    class Reranker:
        def __init__(self):
            self.seen = None
        def rerank(self, query, candidates, top_k):
            self.seen = (query, len(candidates), top_k)
            return list(reversed(candidates))[:top_k]
    reranker = Reranker()
    r, _ = retriever(reranker=reranker)
    assert [c.chunk_id for c in r.search('raw', top_k=2)] == ['c7', 'c6'] and reranker.seen == ('raw', 8, 2)
    reranker.seen = None
    assert [c.chunk_id for c in r.search('raw', top_k=2, use_reranker=False)] == ['c0', 'c1'] and reranker.seen is None


def test_grid_search_default_grid():
    r, index = retriever()
    grid = r.grid_search_alpha('raw')
    assert set(grid) == {f'{f}_alpha_{a}' for f in ('linear', 'rrf') for a in (0.3, 0.5, 0.7)}
    assert all(len(v['top_ids']) == 5 for v in grid.values())
    assert all(call['top_k_candidates'] == 50 for call in index.calls)


# --- hallucination_metrics -----------------------------------------------------------------------

EMPTY = {'faithfulness': 0.0, 'hallucination_rate': 1.0, 'suggested_rubric': 1, 'total_claims': 0,
         'supported_claims': 0}


def test_empty_response_scores_worst_case_without_calling_detector():
    from src.evaluation.hallucination_metrics import compute_hallucination_metrics
    detector = SimpleNamespace(check=lambda *a: pytest.fail('detector called'))
    assert compute_hallucination_metrics('', [{'text': 'x'}], detector=detector) == dict(EMPTY, method='none')


def test_detector_failure_scores_worst_case():
    from src.evaluation.hallucination_metrics import compute_hallucination_metrics
    def boom(*a):
        raise RuntimeError('nli down')
    out = compute_hallucination_metrics('claim.', [], detector=SimpleNamespace(check=boom))
    assert out == dict(EMPTY, method='error', error='nli down')


def test_report_fields_are_mapped_and_missing_chunks_become_empty_list():
    from src.evaluation.hallucination_metrics import compute_hallucination_metrics
    seen = {}
    report = SimpleNamespace(faithfulness_score=.75, hallucination_rate=.25, suggested_rubric=4, total_claims=4,
                             supported_claims=3, contradicted_claims=0, unsupported_claims=1, method='nli',
                             processing_time_ms=12.0)
    def check(text, chunks):
        seen['chunks'] = chunks
        return report
    out = compute_hallucination_metrics('claim.', None, detector=SimpleNamespace(check=check))
    assert seen['chunks'] == []
    assert out == {'faithfulness': .75, 'hallucination_rate': .25, 'suggested_rubric': 4, 'total_claims': 4,
                   'supported_claims': 3, 'contradicted_claims': 0, 'unsupported_claims': 1, 'method': 'nli',
                   'processing_time_ms': 12.0}


# --- statistical_analysis ------------------------------------------------------------------------

@pytest.mark.parametrize('d,label', [(0.19, 'negligible'), (0.2, 'small'), (0.49, 'small'), (0.5, 'medium'),
                                     (0.79, 'medium'), (0.8, 'large')])
def test_cohens_d_label_boundaries(d, label):
    # differences with mean d and sample std 1: [d-1, d+1] has mean d and ddof=1 std sqrt(2)
    diffs = [d * math.sqrt(2) - 1, d * math.sqrt(2) + 1]
    value, got = sa.cohens_d([0.0, 0.0], diffs)
    assert value == pytest.approx(d) and got == label


def test_cohens_d_needs_two_pairs():
    assert sa.cohens_d([1.0], [2.0]) == (0.0, 'insufficient_data')


def test_paired_comparison_insufficient_and_significance_threshold():
    assert sa.paired_comparison([1.0, 2.0], [2.0, 3.0]) == {
        'test_name': 'insufficient_data', 'statistic': 0.0, 'p_value': 1.0, 'is_significant': False, 'n': 2}
    a = [0.1 * i for i in range(30)]
    b = [x + 0.5 + (0.01 if i % 2 else -0.01) for i, x in enumerate(a)]
    out = sa.paired_comparison(a, b)
    assert out['n'] == 30 and bool(out['is_significant']) == (out['p_value'] < 0.05) and out['p_value'] < 0.05
    same = sa.paired_comparison(a, a)
    # Finding P4-S1 (characterised, not fixed): zero differences take the t-test path and give p = NaN,
    # not the wilcoxon_zero_diff p = 1.0 branch. It never reads as significant.
    assert same['test_name'] == 'paired_t_test' and math.isnan(same['p_value']) and not same['is_significant']


def test_shapiro_wilk_edges():
    assert sa.shapiro_wilk_test([1.0, 2.0]) == (False, 0.0)
    normal, p = sa.shapiro_wilk_test([0.1, -0.3, 0.2, 0.05, -0.1, 0.4, -0.2, 0.0, 0.15, -0.05])
    assert bool(normal) == (p > 0.05) and p > 0.0
    big = [((i * 7919) % 1000) / 1000 for i in range(5001)]
    assert sa.shapiro_wilk_test(big)[1] == 0.0                     # Anderson branch, no exact p


def test_bh_and_holm_corrections_known_values():
    assert sa.apply_multiple_comparison_correction([]) == ([], [])
    p, sig = sa.apply_multiple_comparison_correction([0.01, 0.04, 0.03])
    assert p == pytest.approx([0.03, 0.04, 0.04]) and sig == [True, True, True]
    p, sig = sa.apply_multiple_comparison_correction([0.01, 0.04, 0.03], method='holm')
    assert p == pytest.approx([0.03, 0.06, 0.06]) and sig == [True, False, False]


def result(p, **kw):
    base = dict(metric_name='m', system_a='A', system_b='B', mean_a=0.123456789, mean_b=0.987654321,
                std_a=0.111115, std_b=0.222225, improvement=0.864197532, improvement_pct=700.123456,
                p_value=0.0123456789, test_name='wilcoxon', is_normal_a=False, is_normal_b=False,
                effect_size=0.456789123, effect_size_label='small', ci_lower=-0.0123456, ci_upper=0.0654321,
                is_significant=True, n_samples=10)
    base.update(kw, p_value=p)
    return sa.StatisticalResult(**base)


def test_to_dict_rounding_is_fixed():
    d = result(0.0123456789).to_dict()
    assert (d['mean_a'], d['mean_b'], d['std_a'], d['std_b']) == (0.1235, 0.9877, 0.1111, 0.2222)
    assert (d['improvement'], d['improvement_pct'], d['p_value'], d['p_raw']) == (0.8642, 700.12, 0.012346, 0.0123456789)
    assert (d['effect_size'], d['ci_95_lower'], d['ci_95_upper']) == (0.4568, -0.0123, 0.0654)


def test_corrections_annotate_family():
    assert sa.apply_corrections_to_results([]) == []
    rows = sa.apply_corrections_to_results([result(0.01), result(0.04), result(0.03)])
    assert [r.correction_family_size for r in rows] == [3, 3, 3]
    assert [round(r.p_bh, 4) for r in rows] == [0.03, 0.04, 0.04]


def test_bootstrap_ci_defaults_are_10000_resamples_95pct_seed_42():
    a = [math.sin(i * 1.7) for i in range(25)]
    b = [x + 0.1 + 0.3 * math.cos(i * 2.3) for i, x in enumerate(a)]
    default = sa.bootstrap_ci(a, b)
    assert default == sa.bootstrap_ci(a, b, n_bootstrap=10000, ci_level=0.95, seed=42)
    assert default != sa.bootstrap_ci(a, b, seed=43)
    assert default != sa.bootstrap_ci(a, b, ci_level=0.9)
    assert sa.bootstrap_ci([1.0, 2.0], [1.0, 2.0]) == (0.0, 0.0)
    lo, hi = default
    assert lo < hi


def test_mrr_and_map_are_zero_when_nothing_relevant_is_retrieved():
    assert rm.mrr(['x', 'y'], ['a']) == 0.0
    assert rm.map_score(['x', 'y'], ['a']) == 0.0
    assert rm.mrr(['x', 'a'], ['a']) == 0.5
