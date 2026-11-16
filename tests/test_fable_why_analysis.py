"""Auditoría Fable: funciones puras de `scripts/audit/fable_why_analysis.py`.

Fijan la aritmética en la que se apoya el informe del «por qué»: la dominancia por consenso de
RRF, las dos descomposiciones exactas de una diferencia de fidelidad y el cálculo de potencia.
Solo numpy y scipy. La prueba marcada `slow` recorre el análisis completo con los índices
reales (sin modelos) y se omite si los índices no están en disco.
"""

import importlib.util
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "fable_why_analysis", PROJECT_ROOT / "scripts" / "audit" / "fable_why_analysis.py")
why = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(why)

from src.embedding.index.hybrid_index import HybridIndex  # noqa: E402


def _random_legs(rng, depth=50, universe=400):
    ids = [f"c{i}" for i in range(universe)]
    return rng.sample(ids, depth), rng.sample(ids, depth)


# ------------------------------------------------------------------------- RRF
def test_consensus_dominance_holds_for_the_deployed_parameters_and_breaks_at_k_plus_two():
    assert why.consensus_dominates(60, 50)
    assert why.consensus_dominates(60, 61)
    assert not why.consensus_dominates(60, 62)      # 2/122 == 1/61: ya no es estricta
    for k in (1, 10, 60, 100):
        assert why.consensus_dominates(k, k + 1) and not why.consensus_dominates(k, k + 2)


def test_every_document_in_both_legs_outranks_every_single_leg_document():
    rng = random.Random(42)
    for _ in range(200):
        bm25, dense = _random_legs(rng)
        both = set(bm25) & set(dense)
        fused = [cid for cid, _ in why.rrf_fuse(bm25, dense)]
        assert set(fused[:len(both)]) == both


def test_the_first_dense_result_is_dropped_when_five_consensus_documents_exist():
    dense = ["top"] + [f"d{i}" for i in range(49)]
    bm25 = [f"b{i}" for i in range(45)] + ["d44", "d45", "d46", "d47", "d48"]   # consenso en el fondo
    top5 = [cid for cid, _ in why.rrf_fuse(bm25, dense)[:5]]
    assert "top" not in top5 and set(top5) == {"d44", "d45", "d46", "d47", "d48"}


def test_independent_rrf_matches_the_production_fusion_including_ties():
    rng = random.Random(7)
    for _ in range(100):
        bm25, dense = _random_legs(rng, depth=rng.randint(1, 50), universe=120)
        expected = HybridIndex._rrf_fusion(None, [(c, 0.0) for c in bm25], [(c, 0.0) for c in dense], 60, 50)
        assert why.rrf_fuse(bm25, dense)[:50] == expected


def test_rrf_score_uses_one_based_ranks_and_ignores_missing_legs():
    assert why.rrf_score((1, None)) == pytest.approx(1 / 61)
    assert why.rrf_score((50, 50)) == pytest.approx(2 / 110)
    assert why.rrf_score((None, None)) == 0.0


# ------------------------------------------------------------ descomposiciones
def test_selection_within_identity_by_hand():
    fa = {"q1": 1.0, "q2": 0.5, "q3": 0.0}           # A contesta q1..q3
    fb = {"q2": 0.25, "q3": 0.25, "q4": 1.0}         # B contesta q2..q4
    out = why.selection_within(fa, fb)
    assert (out["n_a"], out["n_b"], out["n_common"]) == (3, 3, 2)
    assert out["total"] == pytest.approx(0.5 - 0.5)
    assert out["within"] == pytest.approx(0.25 - 0.25)
    assert out["selection_a"] == pytest.approx(0.5 - 0.25)
    assert out["selection_b"] == pytest.approx(0.5 - 0.25)
    assert out["within"] + out["selection"] == pytest.approx(out["total"])


def test_selection_within_closes_on_random_unbalanced_panels():
    rng = random.Random(3)
    for _ in range(200):
        fa = {f"q{i}": rng.random() for i in rng.sample(range(40), rng.randint(5, 30))}
        fb = {f"q{i}": rng.random() for i in rng.sample(range(40), rng.randint(5, 30))}
        out = why.selection_within(fa, fb)
        if out["n_common"]:
            assert out["within"] + out["selection"] == pytest.approx(out["total"], abs=1e-12)


def test_ratio_shapley_splits_a_ratio_gap_exactly():
    # gemma: 1 apoyada de 4; granite: 4 apoyadas de 14. La brecha 0,25 - 0,2857 = -0,0357 sale de
    # un numerador mucho menor (-0,48) casi compensado por un denominador mucho menor (+0,45).
    numerator, denominator = why.ratio_shapley(1, 4, 4, 14)
    assert numerator == pytest.approx((1 - 4) * (1 / 4 + 1 / 14) / 2)
    assert denominator == pytest.approx((1 + 4) * (1 / 4 - 1 / 14) / 2)
    assert numerator < 0 < denominator
    assert numerator + denominator == pytest.approx(1 / 4 - 4 / 14)
    rng = random.Random(11)
    for _ in range(500):
        g_a, g_b = rng.randint(1, 40), rng.randint(1, 40)
        s_a, s_b = rng.randint(0, g_a), rng.randint(0, g_b)
        num, den = why.ratio_shapley(s_a, g_a, s_b, g_b)
        assert num + den == pytest.approx(s_a / g_a - s_b / g_b, abs=1e-12)


def test_ratio_shapley_is_antisymmetric():
    num_ab, den_ab = why.ratio_shapley(3, 9, 5, 12)
    num_ba, den_ba = why.ratio_shapley(5, 12, 3, 9)
    assert num_ab == pytest.approx(-num_ba) and den_ab == pytest.approx(-den_ba)


# ---------------------------------------------------------------------- potencia
def test_minimum_detectable_effect_and_required_sample_size():
    z_sum = 1.959963984540054 + 0.8416212335729143
    assert why.mde_dz(49) == pytest.approx(z_sum / 7)
    assert why.required_n(0.4) == math.ceil((z_sum / 0.4) ** 2) == 50
    assert why.required_n(-0.4) == 50
    assert math.isinf(why.required_n(0.0)) and math.isnan(why.mde_dz(0))
    assert why.mde_dz(200) < why.mde_dz(50)


# -------------------------------------------------------------- efectos fijos
def test_fixed_effects_ols_recovers_the_slope_and_ignores_group_levels():
    rng = np.random.default_rng(5)
    groups, x, y = [], [], []
    for group in range(60):
        level = rng.normal(scale=10.0)                # nivel propio de cada grupo: debe desaparecer
        for _ in range(4):
            value = rng.normal()
            groups.append(group); x.append(value); y.append(level - 0.3 * value)
    fit = why.fixed_effects_ols(y, np.array(x), groups, clusters=groups)
    assert fit["beta"][0] == pytest.approx(-0.3, abs=1e-9)
    assert int(fit["n_groups"]) == 60 and int(fit["n"]) == 240


def test_fixed_effects_ols_reports_a_null_slope_as_not_significant():
    rng = np.random.default_rng(9)
    groups = [g for g in range(80) for _ in range(3)]
    x = rng.normal(size=240)
    y = rng.normal(size=240)                          # sin relación con x
    fit = why.fixed_effects_ols(y, x, groups, clusters=groups)
    assert fit["p"][0] > 0.01 and fit["se"][0] > 0


# ------------------------------------------------------------- recorrido real
@pytest.mark.slow
@pytest.mark.skipif(not (PROJECT_ROOT / "data" / "indices" / "bm25_adaptive_500.pkl").exists(),
                    reason="índices no disponibles en este entorno")
def test_full_analysis_passes_every_check_on_the_signed_evidence():
    report = why.build_report()
    assert report.failures == []
    assert report.data["retrieval"]["lexical_reproduced"] == [194, 194]
    assert report.data["generation"]["cells_matching_signed_v4"] == [24, 24]
