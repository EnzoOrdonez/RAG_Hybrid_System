"""
Smoke test for Flag 108-full refactor in scripts/compute_retrieval_metrics.py.

Verifies that run_pairwise_tests_with_corrections:
  1. returns a nested dict {metric -> {pair -> dict}} matching the legacy
     shape consumed by generate_latex_table and the CLI print loop.
  2. every leaf carries BH and Holm corrections: p_bh, sig_bh, p_holm,
     sig_holm, correction_family_size.
  3. every leaf is LABELED with its metric, system_a, system_b (no
     parallel unlabeled arrays).
  4. the global family size reflects all (metric x pair) comparisons
     combined, not per-metric.

Fixture: four metrics x three synthetic systems x 30 queries each,
constructed so the 12 raw p-values span the range where BH might or
might not reject.

Run with:
    python scripts/audit/smoke_compute_retrieval_stats.py
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import from the script under test.
import importlib.util  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "compute_retrieval_metrics.py"
spec = importlib.util.spec_from_file_location("compute_retrieval_metrics", SCRIPT)
crm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crm)


def build_fixture(n: int = 30, seed: int = 42):
    """Three systems with increasing-quality metrics, paired by query."""
    rng = np.random.default_rng(seed)
    systems = ["RAG Lexico (BM25)", "RAG Semantico (Dense)", "RAG Hibrido Propuesto"]
    metrics = ["ndcg@5", "avg_score@5", "precision@5", "mrr"]
    base_means = {"ndcg@5": 0.55, "avg_score@5": 2.0, "precision@5": 0.72, "mrr": 0.83}
    deltas = {  # per-system additive shift
        systems[0]: 0.0,
        systems[1]: 0.05,
        systems[2]: 0.12,
    }
    per_q = {s: {} for s in systems}
    # Generate a shared latent per-query difficulty so comparisons are paired.
    latent = rng.normal(0, 1, size=n)
    for s in systems:
        for m in metrics:
            noise = rng.normal(0, 0.2, size=n)
            vals = base_means[m] + deltas[s] + 0.1 * latent + noise
            per_q[s][m] = vals.tolist()
    return per_q, systems, metrics


def main() -> None:
    per_q, systems, metrics = build_fixture()
    nested = crm.run_pairwise_tests_with_corrections(per_q, systems, metrics)

    # 1. Shape: nested[metric][pair] -> dict
    assert set(nested.keys()) == set(metrics), (
        f"metrics missing: expected {metrics}, got {list(nested.keys())}"
    )
    expected_pairs = {
        f"{systems[0]} vs {systems[1]}",
        f"{systems[0]} vs {systems[2]}",
        f"{systems[1]} vs {systems[2]}",
    }
    for m in metrics:
        assert set(nested[m].keys()) == expected_pairs, (
            f"metric '{m}': pair set mismatch {nested[m].keys()}"
        )
    total_tests = sum(len(v) for v in nested.values())
    assert total_tests == 12, f"expected 12 leaf results, got {total_tests}"
    print(f"PASS: nested shape ({total_tests} labeled comparisons)")

    # 2. Every leaf carries the correction keys AND its identifying labels.
    required_keys = {
        "metric",
        "system_a",
        "system_b",
        "p_value",
        "p_raw",
        "p_bh",
        "sig_bh",
        "p_holm",
        "sig_holm",
        "correction_family_size",
        # Legacy aliases kept for downstream callers:
        "cohens_d",
        "mean_1",
        "mean_2",
        "diff",
        "significant",
        "meets_thesis_threshold",
    }
    for m, by_pair in nested.items():
        for pair, result in by_pair.items():
            missing = required_keys - set(result.keys())
            assert not missing, (
                f"leaf {m}/{pair} missing keys: {missing}. "
                f"Available: {sorted(result.keys())}"
            )
    print(f"PASS: every leaf carries p_bh, sig_bh, p_holm, sig_holm + labels")

    # 3. Identifying labels in the leaf match the nested path.
    for m, by_pair in nested.items():
        for pair, result in by_pair.items():
            assert result["metric"] == m, (
                f"leaf metric '{result['metric']}' != nested path '{m}'"
            )
            expected_pair = f"{result['system_a']} vs {result['system_b']}"
            assert expected_pair == pair, (
                f"leaf pair '{expected_pair}' != nested key '{pair}'"
            )
    print("PASS: leaf labels consistent with nested path")

    # 4. Family size = 12 everywhere.
    for m, by_pair in nested.items():
        for pair, result in by_pair.items():
            assert result["correction_family_size"] == 12, (
                f"leaf {m}/{pair} family_size={result['correction_family_size']} "
                f"(expected 12)"
            )
    print("PASS: correction_family_size == 12 across all leaves")

    # 5. BH is monotonic: sort by p_raw, p_bh should be non-decreasing.
    flat = []
    for m, by_pair in nested.items():
        for pair, result in by_pair.items():
            flat.append((result["p_raw"], result["p_bh"]))
    flat.sort()
    p_bh_sorted = [pb for _, pb in flat]
    assert p_bh_sorted == sorted(p_bh_sorted), (
        "BH-adjusted p-values are not monotone-non-decreasing in p_raw"
    )
    print("PASS: BH-adjusted p-values monotone in p_raw")

    # 6. Sig flags are boolean.
    for m, by_pair in nested.items():
        for pair, result in by_pair.items():
            assert isinstance(result["sig_bh"], bool), (
                f"sig_bh not bool: {type(result['sig_bh'])}"
            )
            assert isinstance(result["sig_holm"], bool), (
                f"sig_holm not bool: {type(result['sig_holm'])}"
            )
    print("PASS: sig_bh and sig_holm are booleans")

    print("\nAll smoke tests passed for run_pairwise_tests_with_corrections.")


if __name__ == "__main__":
    main()
