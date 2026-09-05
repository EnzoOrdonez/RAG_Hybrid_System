import math

import pytest

from src.evaluation.retrieval_metrics import compute_all_retrieval_metrics


def test_short_ranking_uses_historical_returned_denominator():
    result = compute_all_retrieval_metrics(["x", "a"], ["a", "b"], [5])
    assert result["precision@5"] == 0.5
    assert result["recall@5"] == 0.5
    assert result["mrr"] == 0.5
    assert result["map"] == 0.25
    assert result["ndcg@5"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


@pytest.mark.parametrize("retrieved,relevant,k", [(["a", "a"], ["a"], 5),
                                               (["a"], ["a", "a"], 5),
                                               (["a"], ["a"], 0),
                                               (["a"], ["a"], -1)])
def test_ambiguous_metrics_fail_before_reporting(retrieved, relevant, k):
    with pytest.raises(ValueError):
        compute_all_retrieval_metrics(retrieved, relevant, [k])


def test_paired_effect_does_not_label_a_constant_shift_as_zero():
    from src.evaluation.statistical_analysis import cohens_d
    assert cohens_d([1, 2, 3], [1, 2, 3]) == (0.0, "negligible")
    with pytest.raises(ValueError, match="undefined"):
        cohens_d([1, 2, 3], [2, 3, 4])
