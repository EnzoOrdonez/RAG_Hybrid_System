"""
Unit + acceptance tests for the exp17 provider-balanced selection.

This rule is the ONLY positive lever the summer phase found, and it is the one change
the survey deployment makes over the measured pipeline. Two things therefore have to
hold, and the second is the one that actually matters:

  1. `balance()` obeys its contract (size, membership, coverage, order, edge cases).
  2. ACCEPTANCE: the 25 cross-cloud queries, selected through the SHIPPED code path
     (SURVEY_DEPLOY config -> RAGPipeline), reproduce the ids exp17 measured. If they
     do not, the deployed artifact is not the one the +0.081 was measured on.

Run: pytest tests/test_coverage_balancer.py -v
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.coverage_balancer import (  # noqa: E402
    balance, build_service_index, corpus_providers, resolve_wanted_providers)

EXP17_IDS = PROJECT_ROOT / "experiments/results/exp17_crosscloud_balanced/retrieval_ids.json"
SUBSET = PROJECT_ROOT / "data/evaluation/cross_cloud_subset.json"
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"


# A pool where provider order is deliberately adversarial: the top-5 by relevance is
# all-aws, so an unbalanced selection covers exactly one of the two wanted providers.
POOL = ["a1", "a2", "a3", "a4", "a5", "z1", "g1", "a6", "z2"]
PROV = {"a1": "aws", "a2": "aws", "a3": "aws", "a4": "aws", "a5": "aws",
        "z1": "azure", "g1": "gcp", "a6": "aws", "z2": "azure"}


def prov_of(cid):
    return PROV.get(cid)


# --------------------------------------------------------------------- contract
def test_returns_exactly_k():
    assert len(balance(POOL, prov_of, ["aws", "azure"], k=5)) == 5


def test_selection_is_a_subset_of_the_pool_without_duplicates():
    out = balance(POOL, prov_of, ["aws", "azure", "gcp"], k=5)
    assert set(out) <= set(POOL)
    assert len(set(out)) == len(out)


def test_covers_every_wanted_provider_present_in_the_pool():
    """The whole point: the unbalanced top-5 here is all-aws."""
    assert set(prov_of(c) for c in POOL[:5]) == {"aws"}
    out = balance(POOL, prov_of, ["aws", "azure", "gcp"], k=5)
    assert {"aws", "azure", "gcp"} <= set(prov_of(c) for c in out)


def test_preserves_reranked_order_within_a_provider():
    out = balance(POOL, prov_of, ["aws", "azure"], k=5)
    aws = [c for c in out if prov_of(c) == "aws"]
    assert aws == sorted(aws, key=POOL.index), "relevance order must survive the quota"


def test_is_idempotent_on_its_own_output():
    out = balance(POOL, prov_of, ["aws", "azure"], k=5)
    assert balance(out, prov_of, ["aws", "azure"], k=5) == out


def test_pool_smaller_than_k_returns_the_whole_pool():
    small = ["a1", "z1"]
    out = balance(small, prov_of, ["aws", "azure"], k=5)
    assert sorted(out) == sorted(small)


def test_single_wanted_provider_degenerates_to_plain_top_k():
    """quota == k, so the rule must not reorder anything."""
    assert balance(POOL, prov_of, ["aws"], k=5) == ["a1", "a2", "a3", "a4", "a5"]


def test_wanted_provider_absent_from_pool_is_simply_not_covered():
    """No evidence exists for it; the remaining slots must still be filled."""
    out = balance(POOL, prov_of, ["aws", "oracle"], k=5)
    assert len(out) == 5
    assert "oracle" not in {prov_of(c) for c in out}


def test_empty_wanted_falls_back_to_top_k():
    assert balance(POOL, prov_of, [], k=3) == POOL[:3]


def test_k_zero_returns_empty():
    assert balance(POOL, prov_of, ["aws", "azure"], k=0) == []


# ------------------------------------------------------- provider resolution
def test_resolver_recovers_providers_the_runtime_detector_misses():
    """GKE/Google are invisible to QueryProcessor.PROVIDER_KEYWORDS."""
    svc = {"eks": "aws", "aks": "azure", "gke": "gcp"}
    present = {"aws", "azure", "gcp"}
    got = resolve_wanted_providers(
        "How do EKS vs AKS vs GKE handle cluster upgrades?", ["k8s"], svc, present)
    assert got == ["aws", "azure", "gcp"]


def test_resolver_drops_providers_absent_from_the_corpus():
    """k8s/cncf keywords survive in QueryProcessor but their corpus was deleted."""
    got = resolve_wanted_providers("Compare Kubernetes on AWS and Azure",
                                   ["k8s", "aws", "azure"], {}, {"aws", "azure", "gcp"})
    assert "k8s" not in got and got == ["aws", "azure"]


def test_resolver_is_deterministic_and_sorted():
    svc = {"eks": "aws", "gke": "gcp"}
    a = resolve_wanted_providers("GKE vs EKS", [], svc, {"aws", "gcp"})
    b = resolve_wanted_providers("GKE vs EKS", [], svc, {"gcp", "aws"})
    assert a == b == ["aws", "gcp"], "quota assignment must not depend on set ordering"


def test_resolver_matches_word_boundaries_not_substrings():
    """'gkeeper' must not resolve to gcp."""
    got = resolve_wanted_providers("What is a gkeeper daemon?", [],
                                   {"gke": "gcp"}, {"gcp", "aws"})
    assert got == []


# ------------------------------------------------------------------ acceptance
@pytest.mark.needs_artifacts
def test_resolver_reproduces_exp17_provider_labels():
    """All 25 exp17 queries must resolve to the hand-labelled provider sets.

    The raw QueryProcessor manages only 20/25 (it misses GCP whenever the query names a
    Google SERVICE rather than the string 'google cloud'), which would have made the
    deployed selection differ from the measured one on 5 of the pilot's own queries.
    """
    for p in (EXP17_IDS, SUBSET, CHUNK_MAP):
        if not p.exists():
            pytest.skip(f"missing artifact: {p}")
    from src.retrieval.query_processor import QueryProcessor

    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    svc, present = build_service_index(chunk_map), corpus_providers(chunk_map)
    qp = QueryProcessor()
    for item in json.loads(SUBSET.read_text(encoding="utf-8")):
        got = resolve_wanted_providers(
            item["question"], qp.process(item["question"]).detected_providers, svc, present)
        assert got == sorted(set(item["cloud_providers"]) & present), item["query_id"]


@pytest.mark.needs_artifacts
def test_balance_reproduces_exp17_balanced_ids_from_the_stored_pool():
    """Replay the rule over exp17's own pool and demand the exact stored ids.

    Uses the persisted baseline/balanced id lists, so it needs no GPU and no index: it
    pins the SELECTION RULE. The end-to-end RAGPipeline check (which also re-runs
    retrieval and reranking) is test_survey_pipeline_matches_exp17, marked slow.
    """
    if not EXP17_IDS.exists() or not CHUNK_MAP.exists():
        pytest.skip("exp17 artifacts or chunk map missing")
    doc = json.loads(EXP17_IDS.read_text(encoding="utf-8"))
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))

    def prov(cid):
        return (chunk_map.get(cid) or {}).get("cloud_provider")

    checked = 0
    for qid, rec in doc["ids"].items():
        # The stored balanced ids came from the k=50 reranked pool; the union of the two
        # stored lists is the part of that pool we can replay offline. Balancing over it
        # must not change the answer, because every id the rule would pick from the full
        # pool that beats a stored one would already be in the stored baseline prefix.
        pool = list(dict.fromkeys(rec["baseline_ids"] + rec["balanced_ids"]))
        got = balance(pool, prov, rec["wanted_providers"], k=len(rec["balanced_ids"]))
        assert set(got) == set(rec["balanced_ids"]), f"{qid}: {got} != {rec['balanced_ids']}"
        checked += 1
    assert checked == 25, f"expected the 25 cross-cloud queries, replayed {checked}"


@pytest.mark.needs_artifacts
def test_survey_config_flips_exactly_two_knobs():
    """SURVEY_DEPLOY must differ from the measured system in nothing else."""
    from src.pipeline.pipeline_config import PROPOSED_HYBRID, SURVEY_DEPLOY, get_config

    a, b = PROPOSED_HYBRID.model_dump(), SURVEY_DEPLOY.model_dump()
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"name", "prompt_routing", "balance_cross_cloud_providers"}, (
        f"survey config drifted from the measured pipeline: {differing}")
    assert b["prompt_routing"] is True and b["balance_cross_cloud_providers"] is True
    # and it must not leak into the experiment configs
    assert get_config("hybrid").balance_cross_cloud_providers is False
    assert get_config("hybrid").prompt_routing is False


def test_legacy_configs_keep_both_knobs_off():
    """Every signed config must reproduce byte-for-byte; exp8 rode the legacy path."""
    from src.pipeline.pipeline_config import PIPELINE_CONFIGS

    for name, cfg in PIPELINE_CONFIGS.items():
        assert cfg.prompt_routing is False, name
        assert cfg.balance_cross_cloud_providers is False, name
