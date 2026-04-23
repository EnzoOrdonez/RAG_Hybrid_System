"""
Smoke test for Flag 137+140 fix in src/evaluation/benchmark_runner.py.

Validates that _save_aggregated_metrics excludes queries whose
hallucination detector returned method="none" or "error" from the
hall_faithfulness_mean / hall_hallucination_rate_mean aggregates, and
emits hall_n_total / hall_n_effective / hall_n_excluded_none_error /
hall_method_counts so downstream consumers know the denominator.

Numerical check: reproduces audit §19.3 table corrections over exp8
(200 queries per system, 13 queries with method="none" per system).

Runs WITHOUT network, no GPU: builds 200 synthetic QueryResult objects
per config, writes the aggregated JSON to a temp directory, and
compares to the expected values derived from the audit table.

Run with:
    python scripts/audit/smoke_hall_n_effective.py
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.benchmark_runner import BenchmarkRunner, QueryResult  # noqa: E402


def make_config_results(config_name: str, mean_faith_real: float, n_total: int = 200,
                        n_none: int = 13, n_error: int = 0):
    """
    Build a list of n_total QueryResult for one config.

    - (n_total - n_none - n_error) real NLI results with a mean
      faithfulness near `mean_faith_real`, linearly varying across
      queries so the realized mean exactly equals mean_faith_real.
    - n_none results with method="none" and faithfulness=1.0
      (the synthetic value produced by the detector when no claims
      are extracted).
    - n_error results with method="error" and faithfulness=0.0.
    """
    n_real = n_total - n_none - n_error
    # Produce a symmetric ramp around mean_faith_real so the mean is exact.
    if n_real == 0:
        real_values = []
    elif n_real == 1:
        real_values = [mean_faith_real]
    else:
        step = 0.10
        offsets = [
            (i - (n_real - 1) / 2) * step / ((n_real - 1) / 2)
            for i in range(n_real)
        ]
        real_values = [
            max(0.0, min(1.0, mean_faith_real + o * 0.1))
            for o in offsets
        ]
        # Re-center to make the mean exact.
        delta = mean_faith_real - sum(real_values) / n_real
        real_values = [max(0.0, min(1.0, v + delta)) for v in real_values]

    results = []
    qid = 0
    for v in real_values:
        results.append(
            QueryResult(
                query_id=f"q{qid}",
                config_name=config_name,
                question=f"synthetic {qid}",
                hallucination_metrics={
                    "faithfulness": v,
                    "hallucination_rate": 1 - v,
                    "total_claims": 3,
                    "supported_claims": round(3 * v),
                    "contradicted_claims": 0,
                    "unsupported_claims": 3 - round(3 * v),
                    "method": "nli",
                },
            )
        )
        qid += 1
    for _ in range(n_none):
        results.append(
            QueryResult(
                query_id=f"q{qid}",
                config_name=config_name,
                question=f"synthetic {qid}",
                hallucination_metrics={
                    "faithfulness": 1.0,   # synthetic
                    "hallucination_rate": 0.0,
                    "total_claims": 0,
                    "supported_claims": 0,
                    "contradicted_claims": 0,
                    "unsupported_claims": 0,
                    "method": "none",
                },
            )
        )
        qid += 1
    for _ in range(n_error):
        results.append(
            QueryResult(
                query_id=f"q{qid}",
                config_name=config_name,
                question=f"synthetic {qid}",
                hallucination_metrics={
                    "faithfulness": 0.0,   # synthetic
                    "hallucination_rate": 1.0,
                    "total_claims": 0,
                    "supported_claims": 0,
                    "contradicted_claims": 0,
                    "unsupported_claims": 0,
                    "method": "error",
                },
            )
        )
        qid += 1
    return results


def main() -> None:
    # Reference values from audit §19.3 Flag 137 table (post-fix means):
    # BM25     0.331, Semantic 0.352, Hibrido 0.325 (excluding 13 none per system).
    # Use these as the "mean_faith_real" targets.
    configs = {
        "BM25":     {"mean_real": 0.331, "n_none": 13},
        "Semantic": {"mean_real": 0.352, "n_none": 14},
        "Hibrido":  {"mean_real": 0.325, "n_none": 13},
    }

    with tempfile.TemporaryDirectory() as tmp:
        runner = BenchmarkRunner(results_dir=tmp)
        exp_id = "smoke_exp"
        (Path(tmp) / exp_id).mkdir(parents=True, exist_ok=True)

        results_by_config = {
            name: make_config_results(name, cfg["mean_real"], 200, cfg["n_none"], 0)
            for name, cfg in configs.items()
        }

        runner._save_aggregated_metrics(exp_id, results_by_config)

        agg_path = Path(tmp) / exp_id / "aggregated_metrics.json"
        agg = json.loads(agg_path.read_text(encoding="utf-8"))

        for name, cfg in configs.items():
            a = agg[name]

            # n fields present + correct
            assert a["hall_n_total"] == 200, (
                f"{name}: hall_n_total={a['hall_n_total']} (expected 200)"
            )
            assert a["hall_n_effective"] == 200 - cfg["n_none"], (
                f"{name}: hall_n_effective={a['hall_n_effective']} "
                f"(expected {200 - cfg['n_none']})"
            )
            assert a["hall_n_excluded_none_error"] == cfg["n_none"], (
                f"{name}: n_excluded={a['hall_n_excluded_none_error']} "
                f"(expected {cfg['n_none']})"
            )
            assert a["hall_method_counts"].get("nli", 0) == 200 - cfg["n_none"]
            assert a["hall_method_counts"].get("none", 0) == cfg["n_none"]

            # Mean is over effective results only — matches audit target
            # to within rounding error of the ramp construction.
            mean = a["hall_faithfulness_mean"]
            assert abs(mean - cfg["mean_real"]) < 1e-3, (
                f"{name}: hall_faithfulness_mean={mean:.4f} "
                f"(expected ≈ {cfg['mean_real']:.4f})"
            )

            # hall_faithfulness_n matches n_effective
            assert a["hall_faithfulness_n"] == 200 - cfg["n_none"], (
                f"{name}: hall_faithfulness_n={a['hall_faithfulness_n']} "
                f"(expected {200 - cfg['n_none']})"
            )

            print(
                f"PASS: {name} — faith_mean={mean:.4f} (target {cfg['mean_real']:.4f}), "
                f"n_eff={a['hall_n_effective']}/{a['hall_n_total']}, "
                f"excluded={a['hall_n_excluded_none_error']}"
            )

        # Sanity: the naive mean that INCLUDES method=none would be higher
        # (because none entries carry synthetic 1.0). Verify our filter
        # actually changed the number.
        for name, cfg in configs.items():
            naive = (cfg["mean_real"] * (200 - cfg["n_none"]) + 1.0 * cfg["n_none"]) / 200
            assert agg[name]["hall_faithfulness_mean"] != naive, (
                f"{name}: filter did not change mean; "
                f"filtered={agg[name]['hall_faithfulness_mean']} "
                f"vs naive={naive}"
            )
            print(
                f"PASS: {name} — filter changed mean from naive={naive:.4f} "
                f"to filtered={agg[name]['hall_faithfulness_mean']:.4f} "
                f"(delta={naive - agg[name]['hall_faithfulness_mean']:+.4f})"
            )

    print("\nAll smoke cases passed for hall_n_effective / method filter.")


if __name__ == "__main__":
    main()
