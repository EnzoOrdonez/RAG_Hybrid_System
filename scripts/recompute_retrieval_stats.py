"""
Recompute retrieval_metrics.json for exp8 / exp8b WITHOUT re-running the
benchmark.

Motivation
----------
Phase 1 (commits 44af679 / b05d7a5 on main) added Benjamini-Hochberg FDR
and Holm-Bonferroni corrections to pairwise comparisons and replaced the
old pooled-SD independent-samples Cohen's d in
scripts/compute_retrieval_metrics.py with paired Cohen's d_z, but the
retrieval_metrics.json files under experiments/results/exp8 and exp8b
were last written in March (pre-fix). Phase 2's re-run (Ollama NLI)
only regenerated exp5.

Retrieval is deterministic under the fixed seed (BGE-large FlatIP +
BM25 rank_bm25, no dropout, no sampling) so the retrieved_ids in
results.json are canonical. The cross-encoder oracle
(ms-marco-MiniLM-L-12-v2) is also deterministic for a fixed (query,
chunk) input — its scores over the same pairs would reproduce exactly.
Therefore the per-query metric values are deterministic, and the only
substantive change vs the March JSON is the stats section: p_raw,
p_bh, sig_bh, p_holm, sig_holm, correction_family_size, cohens_d_z.

This script invokes the Phase 1 refactored
run_pairwise_tests_with_corrections over the same cross-encoder oracle
and writes the regenerated retrieval_metrics.json to the same path,
preserving the pre-fix JSON as retrieval_metrics_pre_fix.json.

Scope
-----
exp8 and exp8b only. exp3 / exp4 / exp6 / exp7 are NOT processed here:
  - exp3 / exp4: no retrieval_metrics.json exists (audit §10 Flag 69);
    re-running those needs the benchmark, not just stats recompute.
  - exp6: ablation on cost/latency, no NLI, stats recompute not applicable.
  - exp7: Phase 4 decision pending on whether to keep the +16.8% claim.

Usage
-----
    python scripts/recompute_retrieval_stats.py --experiment exp8
    python scripts/recompute_retrieval_stats.py --experiment exp8b
    python scripts/recompute_retrieval_stats.py --all   # exp8 + exp8b

Output
------
experiments/results/{exp}/retrieval_metrics.json          # regenerated
experiments/results/{exp}/retrieval_metrics_pre_fix.json  # backup of old
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def _json_default(obj):
    """Convert numpy scalar / bool types to plain Python so json.dumps
    survives Python 3.14's stricter encoder (which no longer coerces
    np.bool_ via the builtin bool() fallback)."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from scripts.compute_retrieval_metrics import (  # noqa: E402
    CROSS_ENCODER_MODEL,
    RELEVANCE_THRESHOLD,
    K_VALUES,
    avg_relevance_score,
    load_chunk_texts,
    mrr_score,
    ndcg_at_k_graded,
    precision_at_k,
    recall_at_k,
    run_pairwise_tests_with_corrections,
    score_chunks_with_cross_encoder,
)

METRIC_FAMILY = ["ndcg@5", "avg_score@5", "precision@5", "mrr"]

# Supported experiments — do NOT expand without user approval per
# plan Phase 2.5 scope (exp3/4/6/7 deliberately excluded).
SUPPORTED_EXPERIMENTS = ("exp8", "exp8b")


def _annotate_leaf_with_cohens_d_z(stats: Dict) -> None:
    """
    Add an explicit cohens_d_z key to every leaf in the nested
    statistical_tests dict, mirroring the effect_size / cohens_d value
    produced by Phase 1's compare_systems (which is paired d_z by
    construction). Leaves cohens_d alone for backward compatibility
    with generate_latex_table.
    """
    for _metric, by_pair in stats.items():
        for _pair, leaf in by_pair.items():
            # Phase 1 emits effect_size under the "effect_size" key and
            # aliases it to "cohens_d" for legacy callers. Surface it
            # under the explicit name the user requested in the
            # Phase 2.5 brief.
            leaf["cohens_d_z"] = float(
                leaf.get("effect_size", leaf.get("cohens_d", 0.0))
            )


def _compute_per_query_and_aggregates(
    configs: Dict,
    query_chunks_map: Dict[str, Dict[str, str]],
    relevance_scores: Dict,
):
    """
    Reproduce the per-query metric loop from compute_retrieval_metrics.main
    (lines 506-551 in that file) without the side-effects (CSV, figure,
    LaTeX). Returns (aggregated_per_system, per_query_per_system).

    The relevance pool per query is the UNION of chunks retrieved by any
    system (query_chunks_map[qid]), not just this system's top-k — this
    matches compute_retrieval_metrics:L513-521 so the IDCG normalizer in
    NDCG@5 is computed over the pooled set, preserving the numerical
    meaning of the metric across systems.
    """
    aggregated: Dict[str, Dict] = {}
    all_per_query: Dict[str, Dict[str, List[float]]] = {}
    system_names = list(configs.keys())

    for sname in system_names:
        per_query: Dict[str, List[float]] = defaultdict(list)

        for result in configs[sname]["results"]:
            qid = result["query_id"]
            retrieved_ids = result.get("retrieved_ids", [])

            # Build the POOLED relevance set for this query from the
            # union of chunks seen by any system. Matches
            # compute_retrieval_metrics.py:L514-521.
            relevant_set = set()
            query_rel_scores: Dict[str, float] = {}
            for cid in query_chunks_map.get(qid, {}):
                sc = relevance_scores.get((qid, cid), 0.0)
                query_rel_scores[cid] = sc
                if sc >= RELEVANCE_THRESHOLD:
                    relevant_set.add(cid)

            for k in K_VALUES:
                per_query[f"precision@{k}"].append(
                    precision_at_k(retrieved_ids, relevant_set, k)
                )
                per_query[f"recall@{k}"].append(
                    recall_at_k(retrieved_ids, relevant_set, k)
                )
            per_query["mrr"].append(mrr_score(retrieved_ids, relevant_set))
            per_query["ndcg@5"].append(
                ndcg_at_k_graded(retrieved_ids, query_rel_scores, 5)
            )
            per_query["avg_score@5"].append(
                avg_relevance_score(retrieved_ids, query_rel_scores, 5)
            )

        agg: Dict = {}
        for metric, values in per_query.items():
            agg[f"{metric}_mean"] = float(np.mean(values)) if values else 0.0
            agg[f"{metric}_std"] = float(np.std(values)) if values else 0.0

        aggregated[sname] = agg
        all_per_query[sname] = dict(per_query)

    return aggregated, all_per_query


def recompute_experiment(exp_id: str) -> Dict:
    """
    Re-score retrieval metrics for one experiment and write the new
    retrieval_metrics.json. Returns the JSON dict that was written (so
    the smoke test can assert over it without a second disk read).
    """
    if exp_id not in SUPPORTED_EXPERIMENTS:
        raise ValueError(
            f"Experiment '{exp_id}' is out of Phase 2.5 scope. "
            f"Supported: {SUPPORTED_EXPERIMENTS}."
        )

    exp_dir = PROJECT_ROOT / "experiments" / "results" / exp_id
    results_path = exp_dir / "results.json"
    metrics_path = exp_dir / "retrieval_metrics.json"
    backup_path = exp_dir / "retrieval_metrics_pre_fix.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")

    # 1. Backup existing retrieval_metrics.json (if any) BEFORE writing.
    if metrics_path.exists():
        if backup_path.exists():
            logger.info(
                "Backup already exists at %s — leaving untouched and "
                "overwriting retrieval_metrics.json anyway",
                backup_path,
            )
        else:
            shutil.copy2(metrics_path, backup_path)
            logger.info("Backed up pre-fix JSON -> %s", backup_path.name)

    # 2. Load results.json.
    data = json.loads(results_path.read_text(encoding="utf-8"))
    configs = data["configs"]
    system_names = list(configs.keys())
    total_queries = max(configs[s]["total_queries"] for s in system_names)
    logger.info(
        "Loaded %s: systems=%s, total_queries=%d",
        exp_id, system_names, total_queries,
    )

    # 3. Build (query_id, question, chunk_id, chunk_text) tuples pooled
    #    across systems, matching compute_retrieval_metrics.py:464-479.
    query_chunks_map: Dict[str, Dict[str, str]] = defaultdict(dict)
    query_questions: Dict[str, str] = {}
    all_chunk_ids: set = set()
    for sname in system_names:
        for r in configs[sname]["results"]:
            qid = r["query_id"]
            query_questions[qid] = r["question"]
            for cid in r.get("retrieved_ids", []):
                all_chunk_ids.add(cid)
                if cid not in query_chunks_map[qid]:
                    query_chunks_map[qid][cid] = ""
    logger.info("Pooled %d unique chunks across %d queries",
                len(all_chunk_ids), len(query_questions))

    # 4. Load chunk texts.
    chunk_texts = load_chunk_texts(all_chunk_ids)

    # 5. Fill the query_chunks_map with the loaded texts.
    for qid in query_chunks_map:
        for cid in query_chunks_map[qid]:
            query_chunks_map[qid][cid] = chunk_texts.get(cid, "")

    # 6. Build flat pairs list and score with the cross-encoder oracle.
    pairs_to_score = []
    for qid, chunks in query_chunks_map.items():
        question = query_questions[qid]
        for cid, text in chunks.items():
            pairs_to_score.append((qid, question, cid, text))
    logger.info("Scoring %d (query, chunk) pairs with %s...",
                len(pairs_to_score), CROSS_ENCODER_MODEL)
    relevance_scores = score_chunks_with_cross_encoder(pairs_to_score)

    # 7. Per-query metrics + aggregates. Pass the POOLED query_chunks_map
    #    so NDCG's IDCG normalizer uses the pooled chunk set per query,
    #    not this system's top-5 only (otherwise NDCG inflates because
    #    the normalizer considers only 5 chunks instead of ~12-15).
    aggregated, all_per_query = _compute_per_query_and_aggregates(
        configs, query_chunks_map, relevance_scores,
    )

    # 8. Pairwise stats with BH / Holm / d_z.
    stats = run_pairwise_tests_with_corrections(
        all_per_query, system_names, METRIC_FAMILY,
    )
    _annotate_leaf_with_cohens_d_z(stats)

    # 9. Assemble new JSON with the same top-level shape as the pre-fix
    #    file (experiment, model, relevance_threshold, total_queries,
    #    total_chunks_scored, score_distribution, systems,
    #    statistical_tests) + two new top-level fields for traceability.
    all_score_vals = list(relevance_scores.values())
    new_doc = {
        "experiment": exp_id,
        "model": CROSS_ENCODER_MODEL,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "total_queries": total_queries,
        "total_chunks_scored": len(relevance_scores),
        "score_distribution": {
            "min": float(np.min(all_score_vals)),
            "max": float(np.max(all_score_vals)),
            "mean": float(np.mean(all_score_vals)),
            "median": float(np.median(all_score_vals)),
            "std": float(np.std(all_score_vals)),
        },
        "systems": {
            sname: {k: v for k, v in aggregated[sname].items()
                    if not k.startswith("_")}
            for sname in system_names
        },
        "statistical_tests": stats,
        # New traceability fields (audit §21.5 T0.8):
        "correction_family_size": sum(len(v) for v in stats.values()),
        "source": {
            "regenerated_by": "scripts/recompute_retrieval_stats.py",
            "reason": "Phase 2.5 — Phase 1 BH/Holm/d_z applied to pre-fix retrieval runs",
            "benchmark_rerun": False,
        },
    }

    metrics_path.write_text(
        json.dumps(new_doc, indent=2, ensure_ascii=False,
                   default=_json_default),
        encoding="utf-8",
    )
    logger.info("Wrote regenerated %s", metrics_path)
    return new_doc


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument(
        "--experiment",
        choices=list(SUPPORTED_EXPERIMENTS),
        help="Experiment ID to recompute.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Recompute both exp8 and exp8b.",
    )
    args = parser.parse_args()

    if args.all:
        targets = list(SUPPORTED_EXPERIMENTS)
    elif args.experiment:
        targets = [args.experiment]
    else:
        parser.error("Specify --experiment or --all")

    for exp in targets:
        logger.info("=" * 60)
        logger.info("Recomputing %s", exp)
        logger.info("=" * 60)
        recompute_experiment(exp)

    logger.info("Done.")


if __name__ == "__main__":
    main()
