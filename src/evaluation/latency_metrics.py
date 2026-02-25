"""
Latency Metrics for thesis evaluation.

Measures per-stage pipeline latency:
  query_processing_ms, retrieval_ms, reranking_ms,
  generation_ms, hallucination_check_ms, total_ms

Statistics: p50, p95, p99, mean, std, min, max.

Thesis criteria:
  - p95 <= 2x fastest baseline
  - Mean <= 3s for medium queries
  - Throughput: queries per minute
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

STAGES = [
    "query_processing_ms",
    "retrieval_ms",
    "reranking_ms",
    "generation_ms",
    "hallucination_check_ms",
    "total_ms",
]


def compute_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Compute p50, p95, p99, mean, std, min, max for a list of latencies."""
    valid = [x for x in latencies if x is not None and not math.isnan(x)]
    if not valid:
        return {
            "p50": 0.0, "p95": 0.0, "p99": 0.0,
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "count": 0,
        }

    arr = np.array(valid)

    # Filter outliers (>3 std dev from mean)
    if len(arr) > 10:
        mean, std = arr.mean(), arr.std()
        if std > 0:
            mask = np.abs(arr - mean) <= 3 * std
            arr = arr[mask]

    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(arr),
    }


def compute_all_latency_metrics(
    latency_records: List[dict],
) -> Dict[str, Dict[str, float]]:
    """
    Compute latency statistics for each pipeline stage.

    Args:
        latency_records: List of dicts, each with keys from STAGES.

    Returns:
        Dict mapping stage name to stats dict.
    """
    results = {}
    for stage in STAGES:
        values = []
        for rec in latency_records:
            v = rec.get(stage)
            if v is not None:
                values.append(v)
        results[stage] = compute_latency_stats(values)

    # Throughput
    total_times = [r.get("total_ms", 0) for r in latency_records if r.get("total_ms")]
    if total_times:
        avg_time_s = np.mean(total_times) / 1000
        results["throughput_qpm"] = 60.0 / avg_time_s if avg_time_s > 0 else 0.0
    else:
        results["throughput_qpm"] = 0.0

    return results


def format_latency_table(
    latency_stats: Dict[str, Dict[str, float]],
) -> str:
    """Format latency stats as a readable table string."""
    lines = [
        f"{'Stage':<25} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8} {'std':>8}",
        "-" * 73,
    ]
    for stage in STAGES:
        s = latency_stats.get(stage, {})
        lines.append(
            f"{stage:<25} {s.get('p50',0):>8.1f} {s.get('p95',0):>8.1f} "
            f"{s.get('p99',0):>8.1f} {s.get('mean',0):>8.1f} {s.get('std',0):>8.1f}"
        )
    tput = latency_stats.get("throughput_qpm", 0)
    if isinstance(tput, dict):
        tput = 0
    lines.append(f"\nThroughput: {tput:.1f} queries/min")
    return "\n".join(lines)
