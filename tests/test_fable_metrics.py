"""Auditoría Fable: métricas de P1 que no tenían tests de comportamiento.

  - Caracterización de `generation_metrics`, `latency_metrics` y de la corrección BH/Holm de
    `statistical_analysis`: fija los valores actuales con casos calculados a mano.
  - H-02 / FABLE-B2 y H-06 / FABLE-B6 (clase B): `xfail(strict=True)` a la espera de Enzo.

Solo numpy, scipy y statsmodels. Ningún modelo.
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.generation_metrics import (  # noqa: E402
    compute_all_generation_metrics, exact_match, f1_token, rouge_l)
from src.evaluation.latency_metrics import (  # noqa: E402
    compute_all_latency_metrics, compute_latency_stats)
from src.evaluation.statistical_analysis import (  # noqa: E402
    apply_multiple_comparison_correction, bootstrap_ci, cohens_d, paired_comparison,
    run_all_comparisons)

BASE = [0.1, 0.5, 0.2, 0.9, 0.4, 0.3, 0.8, 0.6]
BETTER = [0.3, 0.7, 0.5, 0.95, 0.6, 0.7, 0.9, 0.8]
TIED = [0.1, 0.25, 0.3, 0.42, 0.5, 0.61, 0.7, 0.85]


# ------------------------------------------------------------ generation_metrics
def test_exact_match_is_an_80_percent_token_overlap_not_string_equality():
    """El nombre engaña: basta que el 80 % de los tokens del gold aparezca en la respuesta."""
    gold = "S3 stores objects in buckets"
    assert exact_match("Amazon S3 stores objects in buckets durably", gold) == 1.0
    assert exact_match("S3 stores objects in regions", gold) == 1.0      # 4 de 5 = 0,8
    assert exact_match("S3 stores files in regions", gold) == 0.0        # 3 de 5
    assert exact_match("", gold) == 0.0 and exact_match(gold, "") == 0.0


def test_f1_token_works_on_token_sets_so_repetitions_count_once():
    # pred = {the, cat, sat, on, mat}; gold = {the, cat, ran}; comunes = 2
    assert f1_token("the cat sat on the mat", "the cat ran") == pytest.approx(0.5)
    assert f1_token("alpha beta", "gamma delta") == 0.0


def test_rouge_l_is_the_lcs_f1_over_tokens():
    # LCS(the cat sat on the mat, the cat ran on mat) = the cat on mat = 4; P = 4/6, R = 4/5
    assert rouge_l("the cat sat on the mat", "the cat ran on mat") == pytest.approx(8 / 11)


def test_generation_bundle_skips_bert_score_by_default():
    out = compute_all_generation_metrics("the cat sat", "the cat sat")
    assert out == {"exact_match": 1.0, "f1_token": 1.0, "rouge_l": 1.0}


# --------------------------------------------------------------- latency_metrics
def test_latency_stats_for_a_small_sample_use_every_value():
    stats = compute_latency_stats([10.0, 20.0, None, 30.0, float("nan"), 40.0])
    assert (stats["count"], stats["min"], stats["max"]) == (4, 10.0, 40.0)
    assert stats["p50"] == 25.0 and stats["mean"] == 25.0
    assert stats["std"] == pytest.approx(math.sqrt(125))        # desviación poblacional


def test_latency_stats_of_nothing_are_zeros_with_count_zero():
    assert compute_latency_stats([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0,
                                         "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}


def test_throughput_comes_from_the_mean_total_time():
    out = compute_all_latency_metrics([{"total_ms": 1000.0}, {"total_ms": 3000.0}])
    assert out["throughput_qpm"] == pytest.approx(30.0)
    assert out["total_ms"]["count"] == 2 and out["retrieval_ms"]["count"] == 0


# ---------------------------------------------------------- statistical_analysis
def test_bh_and_holm_match_hand_computed_values_and_keep_the_input_order():
    raw = [0.5, 0.001, 0.02]
    p_bh, sig_bh = apply_multiple_comparison_correction(raw, method="fdr_bh")
    p_holm, sig_holm = apply_multiple_comparison_correction(raw, method="holm")
    assert p_bh == pytest.approx([0.5, 0.003, 0.03])
    assert p_holm == pytest.approx([0.5, 0.003, 0.04])
    assert [bool(s) for s in sig_bh] == [False, True, True] == [bool(s) for s in sig_holm]
    assert apply_multiple_comparison_correction([]) == ([], [])


def test_effect_size_and_bootstrap_use_the_b_minus_a_convention_and_a_fixed_seed():
    d, label = cohens_d([1, 2, 3, 4], [2, 4, 3, 6])               # diff = [1, 2, 0, 2]
    assert d == pytest.approx(1.25 / math.sqrt(2.75 / 3)) and label == "large"
    assert bootstrap_ci(BASE, BETTER, n_bootstrap=500) == bootstrap_ci(BASE, BETTER, n_bootstrap=500)
    low, high = bootstrap_ci(BASE, BETTER, n_bootstrap=500)
    assert 0 < low < high
    assert bootstrap_ci([1, 2], [2, 3]) == (0.0, 0.0)             # n < 3: sin intervalo


def test_too_few_pairs_report_insufficient_data_instead_of_a_test():
    out = paired_comparison([0.1, float("nan"), 0.3], [0.2, 0.5, 0.4])   # 2 pares válidos
    assert (out["test_name"], out["p_value"], out["n"]) == ("insufficient_data", 1.0, 2)


def test_family_correction_covers_every_comparison_against_the_baseline():
    results = run_all_comparisons(
        {"base": {"ndcg": BASE, "mrr": BASE}, "hybrid": {"ndcg": BETTER, "mrr": BETTER[::-1]}},
        baseline_name="base")
    assert sorted(r.metric_name for r in results) == ["mrr", "ndcg"]
    assert all(r.correction_family_size == 2 and r.p_bh >= r.p_value for r in results)
    with pytest.raises(ValueError):
        run_all_comparisons({"hybrid": {"ndcg": BETTER}}, baseline_name="base")


# ------------------------------------------------------- propuestas de clase B
def test_identical_scores_give_p_one_not_nan():
    """H-02 (B2 aplicada el 2026-11-15): puntajes idénticos dan p = 1, no NaN."""
    out = paired_comparison(TIED, TIED)
    assert out["p_value"] == 1.0
    assert not out["is_significant"]


def test_one_tied_comparison_does_not_turn_the_whole_family_into_nan():
    """H-02 (B2 aplicada el 2026-11-15): un par empatado no vuelve NaN la familia BH/Holm."""
    results = run_all_comparisons(
        {"base": {"ndcg": BASE, "recall": TIED}, "hybrid": {"ndcg": BETTER, "recall": TIED}},
        baseline_name="base")
    by_metric = {r.metric_name: r for r in results}
    assert math.isfinite(by_metric["ndcg"].p_bh) and math.isfinite(by_metric["ndcg"].p_holm)
    assert by_metric["recall"].p_bh == 1.0 and by_metric["recall"].sig_bh is False


@pytest.mark.xfail(strict=True, reason="FABLE-B6: pendiente de decisión de Enzo")
def test_tail_latencies_are_not_trimmed_before_computing_the_tail_statistics():
    """H-06: hoy la consulta de 10 s desaparece antes de calcular p95, p99 y el máximo."""
    stats = compute_latency_stats([100.0] * 20 + [10_000.0])
    assert stats["count"] == 21
    assert stats["max"] == 10_000.0
    assert stats["p99"] > 100.0
