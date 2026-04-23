"""
Cross-encoder based retrieval quality metrics.

Since test queries lack ground-truth relevant_chunk_ids, we use the
cross-encoder (ms-marco-MiniLM-L-12-v2) as a relevance oracle:

1. For each query, pool ALL unique chunks retrieved by the 3 systems.
2. Score every (query, chunk_text) pair with the cross-encoder.
3. Chunks with score >= RELEVANCE_THRESHOLD are "relevant".
4. Compute Recall@K, Precision@K, MRR, NDCG@K per system.
5. Aggregate across queries and run Wilcoxon signed-rank tests.

Usage:
  python scripts/compute_retrieval_metrics.py                  # default: exp8
  python scripts/compute_retrieval_metrics.py --experiment exp8b
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

RELEVANCE_THRESHOLD = 0.0  # ms-marco cross-encoders output logits; >0 = relevant
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
CHUNK_DIR = PROJECT_ROOT / "data" / "chunks" / "adaptive" / "size_500"
OUTPUT_DIR = PROJECT_ROOT / "output"

K_VALUES = [1, 3, 5]  # Practical K values given 5 retrieved chunks per system


# ============================================================
# Chunk loader
# ============================================================

def load_chunk_texts(chunk_ids: set) -> dict:
    """Load chunk texts from individual JSON files."""
    texts = {}
    missing = 0
    for cid in chunk_ids:
        path = CHUNK_DIR / f"{cid}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            texts[cid] = data.get("text", "")
        else:
            missing += 1
            texts[cid] = ""
    if missing:
        logger.warning("Missing %d chunk files out of %d", missing, len(chunk_ids))
    return texts


# ============================================================
# Cross-encoder scoring
# ============================================================

def score_chunks_with_cross_encoder(
    queries_chunks: list,
    model_name: str = CROSS_ENCODER_MODEL,
    batch_size: int = 64,
) -> dict:
    """
    Score (query, chunk_text) pairs with a cross-encoder.

    Args:
        queries_chunks: list of (query_id, question, chunk_id, chunk_text)
        model_name: HuggingFace cross-encoder model name

    Returns:
        dict: (query_id, chunk_id) -> relevance score
    """
    from sentence_transformers import CrossEncoder
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading cross-encoder: %s on %s", model_name, device)
    model = CrossEncoder(model_name, device=device)

    # Build pairs
    pairs = [(item[1], item[3]) for item in queries_chunks]

    logger.info("Scoring %d (query, chunk) pairs...", len(pairs))
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)

    # Map back to (query_id, chunk_id) -> score
    result = {}
    for i, item in enumerate(queries_chunks):
        qid, _, cid, _ = item
        result[(qid, cid)] = float(scores[i])

    return result


# ============================================================
# Retrieval metrics (with graded and binary relevance)
# ============================================================

def precision_at_k(retrieved_ids, relevant_set, k):
    """Fraction of top-k that are relevant."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for cid in top_k if cid in relevant_set) / len(top_k)


def recall_at_k(retrieved_ids, relevant_set, k):
    """Fraction of relevant docs found in top-k."""
    if not relevant_set:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for cid in top_k if cid in relevant_set) / len(relevant_set)


def mrr_score(retrieved_ids, relevant_set):
    """Reciprocal rank of first relevant result."""
    for i, cid in enumerate(retrieved_ids):
        if cid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k_graded(retrieved_ids, relevance_scores, k):
    """
    NDCG@K with graded relevance from cross-encoder scores.

    Args:
        retrieved_ids: ordered list of chunk IDs
        relevance_scores: dict chunk_id -> cross-encoder score
        k: cutoff
    """
    top_k = retrieved_ids[:k]

    # DCG with graded relevance (use max(0, score) to avoid negative gains)
    dcg = 0.0
    for i, cid in enumerate(top_k):
        rel = max(0.0, relevance_scores.get(cid, 0.0))
        dcg += rel / math.log2(i + 2)

    # Ideal DCG: sort all chunks by score descending, take top-k
    all_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for i in range(min(k, len(all_scores))):
        rel = max(0.0, all_scores[i])
        idcg += rel / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def avg_relevance_score(retrieved_ids, relevance_scores, k):
    """Average cross-encoder score of top-k retrieved chunks."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    scores = [relevance_scores.get(cid, 0.0) for cid in top_k]
    return float(np.mean(scores))


# ============================================================
# Statistical tests
# ============================================================

def run_pairwise_tests_with_corrections(
    all_per_query,
    system_names,
    metric_names,
):
    """
    Run all-pairwise paired tests across every (metric, system-pair) and
    apply Benjamini-Hochberg FDR + Holm-Bonferroni corrections GLOBALLY
    across the whole family of comparisons.

    Delegates the actual statistics (Wilcoxon / Cohen's d_z / bootstrap CI)
    to src.evaluation.statistical_analysis.compare_systems, closing audit
    §15.2 Flag 108 (no multiple-comparison correction anywhere) and §3.5
    Flag 21 (wrong Cohen's d formula: the previous implementation here
    used pooled-SD for independent samples while the data is paired).

    Args:
        all_per_query: {system_name: {metric_name: [per-query scores]}}
        system_names: ordered list of system names; pairs are generated
            with i<j to avoid duplicates.
        metric_names: metrics to compare (e.g., ndcg@5, mrr, precision@5,
            avg_score@5).

    Returns:
        Nested dict {metric -> {"systemA vs systemB" -> test_result_dict}}
        where each test_result_dict contains both the legacy keys
        (p_value, cohens_d, mean_1, mean_2, diff, significant,
        meets_thesis_threshold) kept for backward compatibility with
        generate_latex_table and the CLI print loop, AND the new
        correction keys (p_raw, p_bh, sig_bh, p_holm, sig_holm,
        correction_family_size).
    """
    from src.evaluation.statistical_analysis import (
        apply_corrections_to_results,
        compare_systems,
    )

    flat_results = []
    index_keys = []  # parallel to flat_results: (metric, pair_str)

    for metric in metric_names:
        for i, s1 in enumerate(system_names):
            for s2 in system_names[i + 1:]:
                scores_a = all_per_query[s1][metric]
                scores_b = all_per_query[s2][metric]
                sr = compare_systems(
                    metric_name=metric,
                    system_a_name=s1,
                    system_b_name=s2,
                    scores_a=scores_a,
                    scores_b=scores_b,
                )
                flat_results.append(sr)
                index_keys.append((metric, f"{s1} vs {s2}"))

    # Apply BH-FDR and Holm-Bonferroni globally across ALL comparisons in
    # the family (not per-metric), matching the m16 recompute in
    # paper/audit_outputs/exp8_stats_corrected.csv.
    apply_corrections_to_results(flat_results)

    nested = {}
    for sr, (metric, pair_str) in zip(flat_results, index_keys):
        d = sr.to_dict()
        # Backward-compatible aliases for existing downstream consumers
        # (generate_latex_table reads cohens_d and significant; the CLI
        # print loop at the bottom of main() reads cohens_d, p_value,
        # meets_thesis_threshold).
        d["cohens_d"] = d.get("effect_size", 0.0)
        d["mean_1"] = d["mean_a"]
        d["mean_2"] = d["mean_b"]
        d["diff"] = float(d["mean_a"] - d["mean_b"])
        nested.setdefault(metric, {})[pair_str] = d

    return nested


def run_statistical_tests(per_query_metrics, system_names, metric_name):
    """
    DEPRECATED: retained only for callers that invoke one metric at a
    time. Internally delegates to run_pairwise_tests_with_corrections;
    the returned dict for a single metric is missing the BH/Holm
    correction because corrections require the full family of tests to
    be known in advance.

    New callers should use run_pairwise_tests_with_corrections with the
    complete list of metrics. Audit §15.2 Flag 108 + Addenda A4.
    """
    logger.warning(
        "run_statistical_tests is deprecated; per-metric invocation cannot "
        "apply family-wide BH/Holm corrections. Use "
        "run_pairwise_tests_with_corrections with all metrics at once."
    )
    nested = run_pairwise_tests_with_corrections(
        {s: {metric_name: per_query_metrics[s]} for s in system_names},
        system_names,
        [metric_name],
    )
    return nested.get(metric_name, {})


# ============================================================
# LaTeX table generation
# ============================================================

def generate_latex_table(aggregated, stats):
    """Generate LaTeX table for retrieval metrics."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-encoder based retrieval quality metrics (exp8, 200 queries)}",
        r"\label{tab:retrieval-metrics}",
        r"\begin{tabular}{l" + "c" * 6 + "}",
        r"\toprule",
        r"Sistema & Prec@1 & Prec@3 & Prec@5 & MRR & NDCG@5 & Avg.~Score \\",
        r"\midrule",
    ]

    for sys_name, metrics in aggregated.items():
        short = sys_name.split("(")[0].strip() if "(" in sys_name else sys_name
        p1 = metrics.get("precision@1_mean", 0)
        p3 = metrics.get("precision@3_mean", 0)
        p5 = metrics.get("precision@5_mean", 0)
        m = metrics.get("mrr_mean", 0)
        n5 = metrics.get("ndcg@5_mean", 0)
        avg = metrics.get("avg_score@5_mean", 0)

        lines.append(
            f"{short} & {p1:.3f} & {p3:.3f} & {p5:.3f} & {m:.3f} & {n5:.3f} & {avg:.3f} \\\\"
        )

    lines.append(r"\midrule")

    # Add statistical significance
    for pair, result in stats.items():
        sig = r"$p<0.05$" if result["significant"] else r"$p\geq0.05$"
        d = result["cohens_d"]
        effect = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
        lines.append(
            f"\\multicolumn{{7}}{{l}}"
            f"{{\\footnotesize {pair}: $d={d:.3f}$ ({effect}), {sig}}} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ============================================================
# Figure generation
# ============================================================

def generate_figure(aggregated, output_path):
    """Generate bar chart comparing retrieval metrics across systems."""
    import matplotlib.pyplot as plt

    SYSTEM_COLORS = {
        "RAG Lexico (BM25)": "#E67E22",
        "RAG Semantico (Dense)": "#3498DB",
        "RAG Hibrido Propuesto": "#27AE60",
    }

    systems = list(aggregated.keys())
    short_names = []
    for s in systems:
        if "Lexico" in s:
            short_names.append("Lexical\n(BM25)")
        elif "Semantico" in s:
            short_names.append("Semantic\n(Dense)")
        else:
            short_names.append("Hybrid\n(Proposed)")

    metrics = ["precision@1", "precision@3", "precision@5", "mrr", "ndcg@5", "avg_score@5"]
    metric_labels = ["Prec@1", "Prec@3", "Prec@5", "MRR", "NDCG@5", "Avg Score"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: Bar chart of key metrics ---
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.25

    for i, sys_name in enumerate(systems):
        means = [aggregated[sys_name].get(f"{m}_mean", 0) for m in metrics]
        stds = [aggregated[sys_name].get(f"{m}_std", 0) for m in metrics]
        color = SYSTEM_COLORS.get(sys_name, f"C{i}")
        ax1.bar(
            x + i * width, means, width,
            yerr=stds, label=short_names[i].replace("\n", " "),
            color=color, alpha=0.85, capsize=3,
        )

    ax1.set_xlabel("Metric")
    ax1.set_ylabel("Score")
    ax1.set_title("Retrieval Quality (Cross-Encoder Evaluation)")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metric_labels, fontsize=9)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    # --- Subplot 2: Score distribution (box plot) ---
    ax2 = axes[1]
    box_data = []
    box_labels = []
    box_colors = []
    for sys_name in systems:
        scores = aggregated[sys_name].get("_per_query_avg_score@5", [])
        if scores:
            box_data.append(scores)
            short = short_names[systems.index(sys_name)].replace("\n", " ")
            box_labels.append(short)
            box_colors.append(SYSTEM_COLORS.get(sys_name, "gray"))

    if box_data:
        bp = ax2.boxplot(box_data, tick_labels=box_labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax2.set_ylabel("Avg Cross-Encoder Score")
        ax2.set_title("Relevance Score Distribution")
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved: %s", output_path)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Cross-encoder retrieval metrics")
    parser.add_argument(
        "--experiment", default="exp8",
        help="Experiment ID to process (default: exp8)",
    )
    args = parser.parse_args()
    exp_id = args.experiment

    results_path = PROJECT_ROOT / "experiments" / "results" / exp_id / "results.json"
    results_out = PROJECT_ROOT / "experiments" / "results" / exp_id

    logger.info("=" * 60)
    logger.info("Computing cross-encoder retrieval metrics for %s", exp_id)
    logger.info("=" * 60)

    # 1. Load experiment results
    logger.info("Loading %s results from %s", exp_id, results_path)
    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        sys.exit(1)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    configs = data["configs"]
    system_names = list(configs.keys())

    logger.info("Systems: %s", system_names)
    for sname in system_names:
        logger.info("  %s: %d queries", sname, configs[sname]["total_queries"])

    # 2. Collect all unique chunk IDs across all systems
    all_chunk_ids = set()
    for sname, sdata in configs.items():
        for result in sdata["results"]:
            all_chunk_ids.update(result.get("retrieved_ids", []))

    logger.info("Total unique chunks to score: %d", len(all_chunk_ids))

    # 3. Load chunk texts
    logger.info("Loading chunk texts from %s", CHUNK_DIR)
    chunk_texts = load_chunk_texts(all_chunk_ids)
    non_empty = sum(1 for t in chunk_texts.values() if t)
    logger.info("Loaded %d chunk texts (%d non-empty)", len(chunk_texts), non_empty)

    # 4. Build (query, chunk) pairs for cross-encoder
    #    For each query, pool chunks from ALL systems
    query_chunks_map = defaultdict(dict)  # query_id -> {chunk_id: text}
    query_questions = {}  # query_id -> question

    for sname, sdata in configs.items():
        for result in sdata["results"]:
            qid = result["query_id"]
            question = result["question"]
            query_questions[qid] = question
            for cid in result.get("retrieved_ids", []):
                if cid not in query_chunks_map[qid]:
                    query_chunks_map[qid][cid] = chunk_texts.get(cid, "")

    # Build flat list for scoring
    pairs_to_score = []
    for qid, chunks in query_chunks_map.items():
        question = query_questions[qid]
        for cid, text in chunks.items():
            pairs_to_score.append((qid, question, cid, text))

    logger.info("Total (query, chunk) pairs to score: %d", len(pairs_to_score))
    logger.info("  Avg chunks per query: %.1f", len(pairs_to_score) / len(query_questions))

    # 5. Score with cross-encoder
    relevance_scores = score_chunks_with_cross_encoder(pairs_to_score)

    # Log score distribution
    all_scores = list(relevance_scores.values())
    logger.info(
        "Score distribution: min=%.3f, max=%.3f, mean=%.3f, median=%.3f",
        np.min(all_scores), np.max(all_scores),
        np.mean(all_scores), np.median(all_scores),
    )
    logger.info(
        "Chunks above threshold (%.1f): %d / %d (%.1f%%)",
        RELEVANCE_THRESHOLD,
        sum(1 for s in all_scores if s >= RELEVANCE_THRESHOLD),
        len(all_scores),
        100 * sum(1 for s in all_scores if s >= RELEVANCE_THRESHOLD) / len(all_scores),
    )

    # 6. Compute per-query metrics for each system
    aggregated = {}
    all_per_query = {}  # system_name -> metric_name -> [per-query values]

    for sname in system_names:
        per_query = defaultdict(list)

        for result in configs[sname]["results"]:
            qid = result["query_id"]
            retrieved_ids = result.get("retrieved_ids", [])

            # Build relevance set for this query (chunks above threshold from pool)
            relevant_set = set()
            query_rel_scores = {}
            for cid in query_chunks_map[qid]:
                score = relevance_scores.get((qid, cid), 0.0)
                query_rel_scores[cid] = score
                if score >= RELEVANCE_THRESHOLD:
                    relevant_set.add(cid)

            # Compute metrics
            for k in K_VALUES:
                per_query[f"precision@{k}"].append(
                    precision_at_k(retrieved_ids, relevant_set, k)
                )
                per_query[f"recall@{k}"].append(
                    recall_at_k(retrieved_ids, relevant_set, k)
                )

            per_query["mrr"].append(mrr_score(retrieved_ids, relevant_set))
            per_query["ndcg@5"].append(
                ndcg_at_k_graded(retrieved_ids, query_rel_scores, k=5)
            )
            per_query["avg_score@5"].append(
                avg_relevance_score(retrieved_ids, query_rel_scores, k=5)
            )

        # Aggregate
        agg = {}
        for metric, values in per_query.items():
            agg[f"{metric}_mean"] = float(np.mean(values))
            agg[f"{metric}_std"] = float(np.std(values))

        # Store per-query values for box plots and stats
        agg["_per_query_avg_score@5"] = per_query["avg_score@5"]
        agg["_per_query_ndcg@5"] = per_query["ndcg@5"]
        agg["_per_query_precision@5"] = per_query["precision@5"]

        aggregated[sname] = agg
        all_per_query[sname] = per_query

    # 7. Print results
    print("\n" + "=" * 80)
    print(f"CROSS-ENCODER RETRIEVAL METRICS ({exp_id}, {len(query_questions)} queries)")
    print(f"Model: {CROSS_ENCODER_MODEL}")
    print(f"Relevance threshold: {RELEVANCE_THRESHOLD}")
    print("=" * 80)

    header = f"{'System':<30} {'Prec@1':>7} {'Prec@3':>7} {'Prec@5':>7} {'Rec@5':>7} {'MRR':>7} {'NDCG@5':>7} {'AvgScr':>7}"
    print(header)
    print("-" * len(header))

    for sname in system_names:
        a = aggregated[sname]
        print(
            f"{sname:<30} "
            f"{a.get('precision@1_mean',0):>7.3f} "
            f"{a.get('precision@3_mean',0):>7.3f} "
            f"{a.get('precision@5_mean',0):>7.3f} "
            f"{a.get('recall@5_mean',0):>7.3f} "
            f"{a.get('mrr_mean',0):>7.3f} "
            f"{a.get('ndcg@5_mean',0):>7.3f} "
            f"{a.get('avg_score@5_mean',0):>7.3f}"
        )

    # 8. Statistical tests
    print("\n" + "-" * 80)
    print("STATISTICAL TESTS (Wilcoxon signed-rank)")
    print("-" * 80)

    # Single call across the whole family (4 metrics x 3 pairs = 12 tests)
    # so Benjamini-Hochberg FDR + Holm corrections are applied globally.
    metric_family = ["ndcg@5", "avg_score@5", "precision@5", "mrr"]
    all_stats = run_pairwise_tests_with_corrections(
        all_per_query, system_names, metric_family
    )

    for metric in metric_family:
        for pair, result in all_stats[metric].items():
            p_raw = result["p_value"]
            p_bh = result.get("p_bh", p_raw)
            sig = "***" if p_raw < 0.001 else "**" if p_raw < 0.01 else "*" if p_raw < 0.05 else "n.s."
            sig_bh_marker = "+" if result.get("sig_bh") else " "
            effect = "large" if abs(result["cohens_d"]) >= 0.8 else "medium" if abs(result["cohens_d"]) >= 0.5 else "small"
            thesis = "MET" if result["meets_thesis_threshold"] else "NOT MET"
            print(
                f"  {metric:<15} {pair:<50} "
                f"d={result['cohens_d']:+.3f} ({effect}) "
                f"p={p_raw:.4f} {sig}  "
                f"p_bh={p_bh:.4f} [{sig_bh_marker}]  | thesis={thesis}"
            )

    # 9. Save outputs
    # Create output dirs
    (OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "csv").mkdir(parents=True, exist_ok=True)
    results_out.mkdir(parents=True, exist_ok=True)

    # LaTeX table
    latex = generate_latex_table(aggregated, all_stats.get("ndcg@5", {}))
    latex_path = OUTPUT_DIR / "tables" / f"table_retrieval_metrics_{exp_id}.tex"
    latex_path.write_text(latex, encoding="utf-8")
    logger.info("LaTeX table saved: %s", latex_path)

    # Figure
    fig_path = OUTPUT_DIR / "figures" / f"fig_retrieval_metrics_{exp_id}.png"
    generate_figure(aggregated, fig_path)

    # CSV
    csv_path = OUTPUT_DIR / "csv" / f"{exp_id}_retrieval_metrics.csv"
    csv_lines = ["system,precision@1,precision@3,precision@5,recall@5,mrr,ndcg@5,avg_score@5"]
    for sname in system_names:
        a = aggregated[sname]
        csv_lines.append(
            f"{sname},"
            f"{a.get('precision@1_mean',0):.4f},"
            f"{a.get('precision@3_mean',0):.4f},"
            f"{a.get('precision@5_mean',0):.4f},"
            f"{a.get('recall@5_mean',0):.4f},"
            f"{a.get('mrr_mean',0):.4f},"
            f"{a.get('ndcg@5_mean',0):.4f},"
            f"{a.get('avg_score@5_mean',0):.4f}"
        )
    csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    logger.info("CSV saved: %s", csv_path)

    # JSON with full details (excluding internal per-query lists)
    json_output = {
        "experiment": exp_id,
        "model": CROSS_ENCODER_MODEL,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "total_queries": len(query_questions),
        "total_chunks_scored": len(relevance_scores),
        "score_distribution": {
            "min": float(np.min(all_scores)),
            "max": float(np.max(all_scores)),
            "mean": float(np.mean(all_scores)),
            "median": float(np.median(all_scores)),
            "std": float(np.std(all_scores)),
        },
        "systems": {},
        "statistical_tests": all_stats,
    }
    for sname in system_names:
        a = {k: v for k, v in aggregated[sname].items() if not k.startswith("_")}
        json_output["systems"][sname] = a

    json_path = results_out / "retrieval_metrics.json"
    json_path.write_text(
        json.dumps(json_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("JSON saved: %s", json_path)

    print("\n" + "=" * 80)
    print("OUTPUTS:")
    print(f"  LaTeX:  {latex_path}")
    print(f"  Figure: {fig_path}")
    print(f"  CSV:    {csv_path}")
    print(f"  JSON:   {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
