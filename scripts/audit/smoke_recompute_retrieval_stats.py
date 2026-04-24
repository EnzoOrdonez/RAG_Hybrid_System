"""
Smoke test for Phase 2.5 - recompute retrieval stats.

Executes scripts/recompute_retrieval_stats.py over exp8 and exp8b,
then asserts the regenerated retrieval_metrics.json satisfies:

  1. Every leaf of statistical_tests carries p_raw, p_bh, sig_bh,
     p_holm, sig_holm, correction_family_size.
  2. correction_family_size == 12 across every leaf (4 metrics x 3
     pairs) AND at top level.
  3. p_bh >= p_raw per leaf (BH adjustment never lowers a raw p-value).
  4. sig_holm => sig_bh per leaf (Holm is at least as strict as BH;
     equivalently, BH is at least as permissive as Holm - a comparison
     flagged significant under Holm must also survive BH).
  5. System ordering "Hibrido > Dense > BM25" on every metric mean in
     `systems`, consistent with audit §11 and the existing March
     numbers (ndcg@5, precision@5, mrr, avg_score@5).

Run with:
    python scripts/audit/smoke_recompute_retrieval_stats.py

Requires the cross-encoder model to be locally available (or network to
huggingface.co for the first run) and ~200-300 seconds on CPU per
experiment. Skip with `SMOKE_SKIP_RECOMPUTE=1` in the env to re-run the
assertions against the already-written JSON.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.recompute_retrieval_stats import (  # noqa: E402
    SUPPORTED_EXPERIMENTS,
    recompute_experiment,
)

EXPECTED_FAMILY_SIZE = 12  # 4 metrics x C(3,2) = 4 * 3 = 12

# Ordering reference from audit §11 (empirical, preserved under
# deterministic retrieval). Every metric in systems[].<metric>_mean
# should satisfy Hibrido > Dense > BM25.
SYSTEM_ORDER = {
    "bm25": "RAG Lexico (BM25)",
    "dense": "RAG Semantico (Dense)",
    "hybrid": "RAG Hibrido Propuesto",
}
ORDERED_METRIC_MEANS = [
    "precision@1_mean",
    "precision@3_mean",
    "precision@5_mean",
    "mrr_mean",
    "ndcg@5_mean",
    "avg_score@5_mean",
]
REQUIRED_LEAF_KEYS = {
    "p_raw", "p_bh", "sig_bh", "p_holm", "sig_holm",
    "correction_family_size", "cohens_d_z",
    "metric", "system_a", "system_b",
}


def _run_or_read(exp_id: str) -> dict:
    if os.environ.get("SMOKE_SKIP_RECOMPUTE") == "1":
        path = REPO_ROOT / "experiments" / "results" / exp_id / "retrieval_metrics.json"
        return json.loads(path.read_text(encoding="utf-8"))
    return recompute_experiment(exp_id)


def _assert_leaf_keys(exp_id: str, doc: dict) -> None:
    stats = doc["statistical_tests"]
    total_leaves = 0
    for metric, by_pair in stats.items():
        for pair, leaf in by_pair.items():
            total_leaves += 1
            missing = REQUIRED_LEAF_KEYS - set(leaf.keys())
            assert not missing, (
                f"{exp_id}: {metric}/{pair} missing required keys: "
                f"{sorted(missing)}"
            )
    assert total_leaves == EXPECTED_FAMILY_SIZE, (
        f"{exp_id}: total leaves={total_leaves} expected={EXPECTED_FAMILY_SIZE}"
    )
    print(f"PASS: {exp_id} - all {total_leaves} leaves carry required keys")


def _assert_family_size(exp_id: str, doc: dict) -> None:
    # Top-level.
    assert doc.get("correction_family_size") == EXPECTED_FAMILY_SIZE, (
        f"{exp_id}: top-level correction_family_size="
        f"{doc.get('correction_family_size')}, expected {EXPECTED_FAMILY_SIZE}"
    )
    # Leaf-level.
    for metric, by_pair in doc["statistical_tests"].items():
        for pair, leaf in by_pair.items():
            assert leaf["correction_family_size"] == EXPECTED_FAMILY_SIZE, (
                f"{exp_id}: leaf {metric}/{pair} family_size="
                f"{leaf['correction_family_size']}"
            )
    print(f"PASS: {exp_id} - correction_family_size == {EXPECTED_FAMILY_SIZE} "
          f"top-level and per-leaf")


def _assert_bh_monotone(exp_id: str, doc: dict) -> None:
    for metric, by_pair in doc["statistical_tests"].items():
        for pair, leaf in by_pair.items():
            assert leaf["p_bh"] >= leaf["p_raw"] - 1e-12, (
                f"{exp_id}: {metric}/{pair} p_bh={leaf['p_bh']:.6e} < "
                f"p_raw={leaf['p_raw']:.6e} (BH must never lower raw p)"
            )
    print(f"PASS: {exp_id} - p_bh >= p_raw on every leaf")


def _assert_holm_at_least_as_strict(exp_id: str, doc: dict) -> None:
    """sig_holm => sig_bh. Equivalent: never (sig_holm AND NOT sig_bh)."""
    for metric, by_pair in doc["statistical_tests"].items():
        for pair, leaf in by_pair.items():
            if leaf["sig_holm"]:
                assert leaf["sig_bh"], (
                    f"{exp_id}: {metric}/{pair} sig_holm=True but "
                    f"sig_bh=False - Holm is stricter, so anything Holm "
                    f"accepts, BH must also accept."
                )
    print(f"PASS: {exp_id} - sig_holm implies sig_bh on every leaf")


def _assert_system_ordering(exp_id: str, doc: dict) -> None:
    systems = doc["systems"]
    bm25 = systems[SYSTEM_ORDER["bm25"]]
    dense = systems[SYSTEM_ORDER["dense"]]
    hybrid = systems[SYSTEM_ORDER["hybrid"]]
    for metric_key in ORDERED_METRIC_MEANS:
        b = bm25[metric_key]
        d = dense[metric_key]
        h = hybrid[metric_key]
        assert b < d, (
            f"{exp_id}: {metric_key} BM25={b:.4f} not < Dense={d:.4f}"
        )
        assert d < h, (
            f"{exp_id}: {metric_key} Dense={d:.4f} not < Hibrido={h:.4f}"
        )
    print(f"PASS: {exp_id} - Hibrido > Dense > BM25 across "
          f"{len(ORDERED_METRIC_MEANS)} metrics")


def main() -> None:
    for exp_id in SUPPORTED_EXPERIMENTS:
        print(f"\n=== {exp_id} ===")
        doc = _run_or_read(exp_id)
        _assert_leaf_keys(exp_id, doc)
        _assert_family_size(exp_id, doc)
        _assert_bh_monotone(exp_id, doc)
        _assert_holm_at_least_as_strict(exp_id, doc)
        _assert_system_ordering(exp_id, doc)

    print("\nAll Phase 2.5 recompute smoke assertions passed.")


if __name__ == "__main__":
    main()
