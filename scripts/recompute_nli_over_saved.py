"""
Phase 3.5 — recompute NLI-based hallucination metrics over SAVED answers.

Motivation
----------
Phase 2 (commits c4d6a55 / bf71a94 / b29c3a2) fixed three NLI-layer
bugs:
  - Flag 135  predict() missed apply_softmax=True
  - Flag 138  _nli_matching short-circuited on supported before
              inspecting contradiction evidence
  - Flag 137  method="none" rows synthetic faithfulness=1.0 polluted
              hall_faithfulness_mean aggregates; hall_n_effective added
The Phase 2 rerun only got through exp5. exp6, exp8, exp8b still carry
PRE-fix aggregated_metrics.json (same shape as before 2026-04-23:
hall_faithfulness_mean / hall_hallucination_rate_mean / hall_*_std only,
no hall_n_effective, no method filter).

Because results.json persists the LLM `answer` plus `retrieved_ids` per
query, we do NOT need to re-run Ollama / the LLM to recompute
hallucination metrics. We only need to re-run the NLI-based
HallucinationDetector.check() over each saved (answer, chunks) pair,
then re-aggregate with the Phase 2 method filter.

Scope (approved 2026-04-23): exp6, exp8, exp8b. Explicitly NOT exp5
(already post-fix) and NOT exp7 (Phase 4 decision pending on +16.8%).

Output per experiment:
  experiments/results/{exp}/aggregated_metrics.json              # updated in place
  experiments/results/{exp}/aggregated_metrics_pre_nli_fix.json  # backup

Does NOT touch results.json. Per-query rows keep their original
hallucination_metrics dict (PRE-fix) for traceability. Only the
aggregate is refreshed.

Usage
-----
    python scripts/recompute_nli_over_saved.py --experiment exp8
    python scripts/recompute_nli_over_saved.py --all   # exp6, exp8, exp8b
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from scripts.compute_retrieval_metrics import load_chunk_texts  # noqa: E402
from src.evaluation.hallucination_metrics import (  # noqa: E402
    compute_hallucination_metrics,
)
from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402

EXP_DIR = PROJECT_ROOT / "experiments" / "results"
SUPPORTED_EXPERIMENTS = ("exp6", "exp8", "exp8b")
EXCLUDED_METHODS = {"none", "error"}

COUNT_KEYS = [
    "total_claims",
    "supported_claims",
    "contradicted_claims",
    "unsupported_claims",
]


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_chunks_list(retrieved_ids: List[str], texts: Dict[str, str]) -> List[dict]:
    chunks = []
    for cid in retrieved_ids:
        t = texts.get(cid, "")
        if t:
            chunks.append({"chunk_id": cid, "text": t})
    return chunks


def recompute_experiment(
    exp_id: str,
    detector: HallucinationDetector,
) -> Dict:
    if exp_id not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            f"{exp_id!r} out of Phase 3.5 scope. Supported: {SUPPORTED_EXPERIMENTS}"
        )

    exp_dir = EXP_DIR / exp_id
    results_path = exp_dir / "results.json"
    agg_path = exp_dir / "aggregated_metrics.json"
    backup_path = exp_dir / "aggregated_metrics_pre_nli_fix.json"

    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not agg_path.exists():
        raise FileNotFoundError(agg_path)

    # Backup the pre-fix aggregated_metrics.json once (idempotent).
    if not backup_path.exists():
        shutil.copy2(agg_path, backup_path)
        logger.info("Backup: %s -> %s", agg_path.name, backup_path.name)
    else:
        logger.info("Backup already exists: %s", backup_path.name)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    configs = data["configs"]

    # Pool every retrieved chunk id across every config so the chunk-text
    # load happens once per experiment.
    all_ids: set = set()
    for cfg_name, cfg_data in configs.items():
        for r in cfg_data.get("results", []):
            all_ids.update(r.get("retrieved_ids", []) or [])

    logger.info(
        "%s: pooling %d unique chunk ids across %d configs",
        exp_id, len(all_ids), len(configs),
    )
    chunk_texts = load_chunk_texts(all_ids)

    # Load the existing aggregated_metrics.json so we preserve
    # retrieval / generation / latency aggregates untouched.
    aggregated_pre = json.loads(agg_path.read_text(encoding="utf-8"))

    new_aggregated: Dict = {}
    per_config_report: Dict[str, Dict] = {}

    for cfg_name, cfg_data in configs.items():
        query_results = cfg_data.get("results", [])
        new_hall_per_query: List[Dict] = []
        for r in query_results:
            if r.get("error"):
                # Error rows stay errors; they carry no hallucination_metrics
                # to recompute.
                continue
            answer = r.get("answer", "") or ""
            retrieved_ids = r.get("retrieved_ids", []) or []
            chunks = _build_chunks_list(retrieved_ids, chunk_texts)
            hm = compute_hallucination_metrics(answer, chunks, detector=detector)
            new_hall_per_query.append(hm)

        n_total_hall = len(new_hall_per_query)
        effective = [
            h for h in new_hall_per_query
            if h.get("method") not in EXCLUDED_METHODS
        ]
        n_effective_hall = len(effective)
        n_excluded = n_total_hall - n_effective_hall
        method_counts = dict(Counter(h.get("method", "unknown") for h in new_hall_per_query))

        pre_cfg = aggregated_pre.get(cfg_name, {})
        new_cfg = dict(pre_cfg)

        # Drop any pre-fix hall_* aggregates before inserting new ones,
        # so we don't leave stale means lying around.
        for k in list(new_cfg.keys()):
            if k.startswith("hall_"):
                new_cfg.pop(k, None)

        new_cfg["hall_n_total"] = n_total_hall
        new_cfg["hall_n_effective"] = n_effective_hall
        new_cfg["hall_n_excluded_none_error"] = n_excluded
        new_cfg["hall_method_counts"] = method_counts

        for key in ("faithfulness", "hallucination_rate"):
            values = [h[key] for h in effective if key in h]
            if values:
                new_cfg[f"hall_{key}_mean"] = float(np.mean(values))
                new_cfg[f"hall_{key}_std"] = float(np.std(values))
                new_cfg[f"hall_{key}_n"] = len(values)

        for key in COUNT_KEYS:
            values = [h[key] for h in new_hall_per_query if key in h]
            if values:
                new_cfg[f"hall_{key}_mean"] = float(np.mean(values))
                new_cfg[f"hall_{key}_sum"] = int(sum(values))

        new_aggregated[cfg_name] = new_cfg

        # Diff report payload.
        per_config_report[cfg_name] = {
            "pre_hall_faithfulness_mean": pre_cfg.get("hall_faithfulness_mean"),
            "post_hall_faithfulness_mean": new_cfg.get("hall_faithfulness_mean"),
            "hall_n_total": n_total_hall,
            "hall_n_effective": n_effective_hall,
            "hall_n_excluded_none_error": n_excluded,
            "method_counts": method_counts,
        }

    # Write (same path, overwriting the pre-fix file; backup taken above).
    agg_path.write_text(
        json.dumps(new_aggregated, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.info("Wrote post-NLI-fix aggregated metrics: %s", agg_path)
    return per_config_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--experiment",
        choices=list(SUPPORTED_EXPERIMENTS),
        help="Single experiment to recompute.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Recompute exp6, exp8, exp8b in that order.",
    )
    args = parser.parse_args()

    if args.all:
        targets = list(SUPPORTED_EXPERIMENTS)
    elif args.experiment:
        targets = [args.experiment]
    else:
        parser.error("Specify --experiment or --all")

    # Share the detector across experiments so the NLI model loads once.
    logger.info("Instantiating HallucinationDetector (loads nli-deberta-v3-small)")
    detector = HallucinationDetector(use_nli=True)
    # Trigger eager load so the model download happens up-front.
    _ = detector.nli_model

    combined_report: Dict[str, Dict] = {}
    for exp in targets:
        logger.info("=" * 60)
        logger.info("Recomputing NLI for %s", exp)
        logger.info("=" * 60)
        combined_report[exp] = recompute_experiment(exp, detector)

    # Compact human-readable summary of the pre/post faithfulness shift.
    print("\n=== Phase 3.5 summary ===")
    for exp, by_cfg in combined_report.items():
        print(f"\n{exp}:")
        for cfg, info in by_cfg.items():
            pre = info["pre_hall_faithfulness_mean"]
            post = info["post_hall_faithfulness_mean"]
            pre_str = f"{pre:.4f}" if pre is not None else "--"
            post_str = f"{post:.4f}" if post is not None else "--"
            delta_str = (
                f"{post - pre:+.4f}"
                if (pre is not None and post is not None) else "n/a"
            )
            print(
                f"  {cfg}: faithfulness pre={pre_str}  post={post_str}  "
                f"delta={delta_str}  (n_eff={info['hall_n_effective']}/"
                f"{info['hall_n_total']}, excluded={info['hall_n_excluded_none_error']})"
            )


if __name__ == "__main__":
    main()
