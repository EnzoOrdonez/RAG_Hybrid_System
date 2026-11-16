"""Auditoría Fable: funciones puras de las sondas de concepto y del análisis de exp20.

Fijan la aritmética de `scripts/audit/fable_concept_probes.py` (tokens perdidos por truncado, NDCG
relativo al pool, correlación intraclase, efecto de diseño y contraste agrupado) y el contraste
pareado de `scripts/audit/exp20_rerank_arms.py`. La prueba `slow` recorre las sondas completas
sobre la corrida exp20 y se omite si esa corrida no está en disco. No se carga ningún modelo.
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / "audit" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probes = _load("fable_concept_probes")
exp20 = _load("exp20_rerank_arms")


# -------------------------------------------------------------------- truncado
def test_lost_tokens_counts_only_what_exceeds_the_budget():
    assert probes.lost_tokens(600, 509) == 91
    assert probes.lost_tokens(509, 509) == 0
    assert probes.lost_tokens(100, 509) == 0
    assert probes.lost_tokens(30, -5) == 30          # sin sitio: se pierde todo


# ------------------------------------------------------------------------ NDCG
def test_ndcg_matches_the_signed_definition_by_hand():
    pool = {"a": 2.0, "b": 1.0, "c": 3.0}
    expected = (2.0 + 1.0 / math.log2(3)) / (3.0 + 2.0 / math.log2(3))
    assert probes.ndcg_graded(["a", "b"], pool, k=2) == pytest.approx(expected)


def test_ndcg_is_identical_to_the_signed_metric_script_on_random_pools():
    spec = importlib.util.spec_from_file_location(
        "compute_retrieval_metrics", PROJECT_ROOT / "scripts" / "compute_retrieval_metrics.py")
    signed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signed)
    rng = np.random.default_rng(11)
    for _ in range(300):
        size = int(rng.integers(1, 15))
        pool = {f"c{i}": float(rng.normal(scale=3.0)) for i in range(size)}     # logits: hay negativos
        ranking = [f"c{i}" for i in rng.permutation(size)[:5]] + (["fuera"] if rng.random() < 0.3 else [])
        assert probes.ndcg_graded(ranking, pool, k=5) == pytest.approx(signed.ndcg_at_k_graded(ranking, pool, 5), abs=1e-12)


def test_the_same_ranking_scores_lower_when_another_system_adds_a_better_chunk():
    ranking = ["a", "b"]
    alone = probes.ndcg_graded(ranking, {"a": 2.0, "b": 1.0}, k=2)
    with_rival = probes.ndcg_graded(ranking, {"a": 2.0, "b": 1.0, "c": 3.0}, k=2)
    assert alone == pytest.approx(1.0)
    assert with_rival < alone


def test_ndcg_clamps_negative_scores_and_is_zero_when_nothing_is_positive():
    assert probes.ndcg_graded(["a"], {"a": -4.0, "b": 2.0}, k=1) == 0.0
    assert probes.ndcg_graded(["a", "b"], {"a": -1.0, "b": -2.0}, k=2) == 0.0
    assert probes.ndcg_graded(["x"], {"a": 1.0}, k=1) == 0.0     # fuera del pool: ganancia 0


# ------------------------------------------------- consultas agrupadas por plantilla
def test_icc_is_one_for_pure_group_effects_and_near_zero_for_noise():
    groups = [g for g in range(20) for _ in range(5)]
    assert probes.icc_oneway([float(g) for g in groups], groups) == pytest.approx(1.0)
    rng = np.random.default_rng(42)
    assert abs(probes.icc_oneway(list(rng.normal(size=100)), groups)) < 0.15
    assert math.isnan(probes.icc_oneway([1.0, 2.0, 3.0], ["g", "g", "g"]))


def test_icc_recovers_a_known_value_with_unbalanced_groups():
    rng = np.random.default_rng(5)
    estimates = []
    for _ in range(40):
        values, groups = [], []
        for group in range(74):
            level = rng.normal(scale=math.sqrt(0.4))                    # ICC real = 0,4
            for _ in range(int(rng.integers(1, 9))):
                values.append(level + rng.normal(scale=math.sqrt(0.6)))
                groups.append(group)
        estimates.append(probes.icc_oneway(values, groups))
    assert np.mean(estimates) == pytest.approx(0.4, abs=0.05)


def test_design_effect_follows_kish_for_equal_and_unequal_groups():
    assert probes.design_effect(0.0, [5, 5, 5]) == pytest.approx(1.0)
    assert probes.design_effect(1.0, [5, 5, 5]) == pytest.approx(5.0)
    assert probes.design_effect(0.5, [1, 3]) == pytest.approx(1 + (10 / 4 - 1) * 0.5)
    assert math.isnan(probes.design_effect(float("nan"), [2, 2]))


def test_cluster_test_equals_the_classic_one_when_every_observation_is_its_own_cluster():
    rng = np.random.default_rng(7)
    diff = list(rng.normal(loc=0.2, size=40))
    out = probes.cluster_mean_test(diff, list(range(40)))
    assert out["se_cluster"] == pytest.approx(out["se_iid"])
    assert out["p_cluster"] == pytest.approx(out["p_iid"])


def test_cluster_test_widens_the_error_when_observations_repeat_inside_clusters():
    rng = np.random.default_rng(3)
    base = list(rng.normal(loc=0.3, size=30))
    diff = [value for value in base for _ in range(4)]            # cada grupo repite su valor 4 veces
    clusters = [g for g in range(30) for _ in range(4)]
    out = probes.cluster_mean_test(diff, clusters)
    assert out["clusters"] == 30 and out["n"] == 120
    assert out["se_cluster"] > 1.8 * out["se_iid"]                # ~2: la información real es de 30, no de 120
    assert out["p_cluster"] > out["p_iid"]


def test_template_masks_the_provider_and_keeps_the_question_stem():
    a = probes.template_of("What are the limits and quotas for AWS EKS?")
    b = probes.template_of("What are the limits and quotas for Azure AKS?")
    assert a == b == "What are the limits"
    assert probes.template_of("How do I configure Google Cloud Storage?") == "How do I configure"
    assert probes.template_of("Compare AWS Lambda with Azure Functions") == "Compare <P> Lambda with"


# ------------------------------------------------------------- contraste de exp20
def test_exp20_paired_contrast_counts_wins_ties_and_losses():
    out = exp20._paired([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 4.0, 2.0])
    assert (out["wins"], out["ties"], out["losses"], out["n"]) == (2, 1, 1, 4)
    assert out["mean"] == pytest.approx(0.5)


def test_exp20_paired_contrast_of_identical_arms_is_not_significant():
    out = exp20._paired([0.5, 0.7, 0.9], [0.5, 0.7, 0.9])
    assert out["p"] == 1.0 and out["dz"] == 0.0 and out["ties"] == 3


def test_fair_hybrid_arm_fuses_the_filtered_pools_and_reorders_by_reranker_score():
    why = _load("fable_why_analysis")
    row = {"pools": {"lexico": ["a", "b", "c"], "denso": ["c", "d", "a"]},
           "reranker_scores": {"a": 0.1, "b": 5.0, "c": 2.0, "d": 9.0}}
    # depth=2: RRF deja solo el consenso (a y c); el reranker pone c delante.
    assert exp20.fair_hybrid_top5(row, why.rrf_fuse, depth=2, k=5) == ["c", "a"]
    # depth amplio: entran los cuatro y manda el puntaje del reranker.
    assert exp20.fair_hybrid_top5(row, why.rrf_fuse, depth=50, k=3) == ["d", "b", "c"]


# ------------------------------------------------------------ modo firmado de exp20
def test_signed_results_use_the_exp11_schema_so_the_metric_script_can_read_them():
    rows = [{"query_id": "q1", "question": "pregunta", "arms": {arm: [f"{arm}-{i}" for i in range(5)]
                                                              for arm in exp20.ARM_CONFIG_NAMES}}]
    configs = exp20.exp11_style_configs(rows)
    assert set(configs) == set(exp20.ARM_CONFIG_NAMES.values())
    assert {"RAG Lexico (BM25)", "RAG Semantico (Dense)", "RAG Hibrido Propuesto",
            "RAG Hibrido (pre-rerank RRF)"} <= set(configs)          # los cuatro nombres firmados de exp11
    block = configs["RAG Semantico (Dense) + reranker"]
    assert block["total_queries"] == 1 and block["errors"] == 0
    assert block["results"][0]["retrieved_ids"] == [f"denso_rerank-{i}" for i in range(5)]
    assert block["results"][0]["query_id"] == "q1" and block["results"][0]["question"] == "pregunta"


def test_signed_mode_writes_each_evidence_file_once_and_refuses_to_overwrite(tmp_path, monkeypatch):
    import src.utils.signed_evidence as signed_evidence
    monkeypatch.setattr(signed_evidence, "RESULTS", tmp_path)
    monkeypatch.setattr(exp20, "SIGNED", True)
    target = tmp_path / "exp99_prueba" / "results.json"
    exp20._write_once(target, "primera")
    assert target.read_text(encoding="utf-8") == "primera"
    with pytest.raises(SystemExit):
        exp20._write_once(target, "segunda")
    assert target.read_text(encoding="utf-8") == "primera"


def test_audit_mode_may_rewrite_its_own_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(exp20, "SIGNED", False)
    target = tmp_path / "analysis.md"
    exp20._write_once(target, "a")
    exp20._write_once(target, "b")
    assert target.read_text(encoding="utf-8") == "b"


# --------------------------------------------------------------- recorrido real
@pytest.mark.slow
@pytest.mark.skipif(not probes.EXP20.exists() or not (probes.MODELS / "nli-deberta-v3-small" / "tokenizer.json").exists(),
                    reason="corrida exp20 o vocabularios no disponibles en este entorno")
def test_probes_reproduce_the_signed_circular_ndcg_from_the_exp20_run():
    report = probes.build_report()
    assert report.failures == []
    assert report.data["ndcg_circular"]["max_dev_vs_signed"] < 5e-4
    assert report.data["embedder"]["chunks"] == 24481
