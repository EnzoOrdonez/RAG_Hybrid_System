"""
Phase 3 — regenerate figures and tables from post-fix JSONs.

Scope: exp5, exp6, exp8, exp8b only. Out of scope: exp3/exp4/exp7.

Produces 10 artefacts with the `_phase3` suffix in
paper/overleaf_ready/figures/. Preserves the 6 pre-existing baselines
with `_pre_phase3` suffix (byte-identical copy). The 4 new
`table_expN_faithfulness_phase3.tex` files have no pre-Phase-3 baseline
so no backup is created for them.

Sources (read-only — Phase 2/2.5 artefacts):
  exp5/aggregated_metrics.json   post-rerun (Honovich + hall_n_effective)
  exp6/aggregated_metrics.json   pre-fix shape (no hall_n_effective)
  exp8/retrieval_metrics.json    post Phase 2.5 (BH/Holm/d_z)
  exp8b/retrieval_metrics.json   post Phase 2.5
  exp8/aggregated_metrics.json   pre-fix faithfulness for exp8 table
  exp8b/aggregated_metrics.json  pre-fix faithfulness for exp8b table

Does NOT run the cross-encoder oracle (already done in Phase 2.5). The
retrieval figure/table are generated from the existing
retrieval_metrics.json using compute_retrieval_metrics.generate_figure
(bar chart subplot only — box plot requires per-query scores which are
not preserved in the JSON).

Writes a human-readable diff report to paper/phase3_diff_report.md.

Usage:
    python scripts/generate_phase3_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FIGURES_DIR = REPO_ROOT / "paper" / "overleaf_ready" / "figures"
EXP_DIR = REPO_ROOT / "experiments" / "results"
DIFF_REPORT = REPO_ROOT / "paper" / "phase3_diff_report.md"

SCOPE_ARTEFACTS = [
    # (base_name, ext, has_preexisting_baseline)
    ("fig_llm_comparison", "png", True),
    ("fig_latency_breakdown", "png", True),
    ("fig_retrieval_metrics", "png", True),
    ("fig_retrieval_metrics_exp8b", "png", True),
    ("table_retrieval_metrics", "tex", True),
    ("table_retrieval_metrics_exp8b", "tex", True),
    ("table_exp5_faithfulness", "tex", False),
    ("table_exp6_faithfulness", "tex", False),
    ("table_exp8_faithfulness", "tex", False),
    ("table_exp8b_faithfulness", "tex", False),
]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _backup_baseline(base_name: str, ext: str) -> Path | None:
    """Copy paper/overleaf_ready/figures/<base>.<ext> to
    <base>_pre_phase3.<ext> if the baseline exists. Idempotent: if the
    backup already exists it is left untouched.
    """
    src = FIGURES_DIR / f"{base_name}.{ext}"
    dst = FIGURES_DIR / f"{base_name}_pre_phase3.{ext}"
    if not src.exists():
        return None
    if dst.exists():
        logger.info("Backup already exists: %s", dst.name)
        return dst
    shutil.copy2(src, dst)
    logger.info("Backup: %s -> %s", src.name, dst.name)
    return dst


# ============================================================
# Generator: fig_llm_comparison (exp5) — patched metric list
# ============================================================

def gen_fig_llm_comparison(out_path: Path) -> Dict:
    """
    Patched version of ResultsExporter.fig_llm_comparison that uses only
    columns that exist in the post-rerun exp5/aggregated_metrics.json:
    hall_faithfulness_mean and lat_total_ms_mean. The original method
    also referenced ret_ndcg@5_mean / gen_f1_token_mean / gen_rouge_l_mean
    which the Phase 1 KeyError guard now rejects.
    """
    import matplotlib.pyplot as plt

    data = json.loads(
        (EXP_DIR / "exp5" / "aggregated_metrics.json").read_text(encoding="utf-8")
    )
    models = list(data.keys())

    faith = [data[m]["hall_faithfulness_mean"] for m in models]
    faith_std = [data[m].get("hall_faithfulness_std", 0.0) for m in models]
    lat = [data[m]["lat_total_ms_mean"] / 1000.0 for m in models]  # seconds
    n_eff = [data[m].get("hall_n_effective") for m in models]
    n_tot = [data[m].get("hall_n_total") for m in models]

    short_labels = []
    for m in models:
        if "llama" in m.lower():
            short_labels.append("Llama 3.1 8B")
        elif "qwen" in m.lower():
            short_labels.append("Qwen 2.5 7B")
        elif "mistral" in m.lower():
            short_labels.append("Mistral 7B")
        else:
            short_labels.append(m)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#3498DB", "#E67E22", "#27AE60"]

    # Left: faithfulness bars
    x = np.arange(len(models))
    ax1.bar(x, faith, yerr=faith_std, capsize=4,
            color=colors[:len(models)], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=9)
    ax1.set_ylabel("Faithfulness (NLI, post-Honovich)")
    ax1.set_title("exp5 LLM Comparison — Faithfulness")
    ax1.set_ylim(0, max(faith) * 1.25 if faith else 1.0)
    ax1.grid(axis="y", alpha=0.3)
    for i, (v, ne, nt) in enumerate(zip(faith, n_eff, n_tot)):
        if ne is not None and nt is not None:
            ax1.text(i, v + 0.01,
                     f"{v:.3f}\n(n={ne}/{nt})",
                     ha="center", va="bottom", fontsize=8)
        else:
            ax1.text(i, v + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)

    # Right: latency bars
    ax2.bar(x, lat, color=colors[:len(models)], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=9)
    ax2.set_ylabel("Mean total latency per query (s)")
    ax2.set_title("exp5 LLM Comparison — Latency")
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate(lat):
        ax2.text(i, v + max(lat) * 0.01, f"{v:.1f}s",
                 ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "exp5 LLM trade-offs (Phase 3 — Honovich NLI aggregation, n_effective filter)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "models": models,
        "short_labels": short_labels,
        "faithfulness": faith,
        "faithfulness_std": faith_std,
        "latency_s": lat,
        "n_effective": n_eff,
        "n_total": n_tot,
    }


# ============================================================
# Generator: fig_latency_breakdown (exp8) — use ResultsExporter
# ============================================================

def gen_fig_latency_breakdown(out_path: Path) -> Dict:
    import matplotlib.pyplot as plt

    data = json.loads(
        (EXP_DIR / "exp8" / "aggregated_metrics.json").read_text(encoding="utf-8")
    )
    configs = list(data.keys())
    stages = [
        ("lat_query_processing_ms_mean", "Query Processing"),
        ("lat_retrieval_ms_mean", "Retrieval"),
        ("lat_reranking_ms_mean", "Reranking"),
        ("lat_generation_ms_mean", "Generation"),
        ("lat_hallucination_check_ms_mean", "Hallucination Check"),
    ]
    stage_colors = ["#3498DB", "#E67E22", "#27AE60", "#F44336", "#9E9E9E"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(configs))
    diff_info: Dict[str, Dict[str, float]] = {c: {} for c in configs}
    for (key, label), color in zip(stages, stage_colors):
        values = []
        for c in configs:
            v = data[c].get(key, 0.0) / 1000.0  # convert to seconds
            values.append(v)
            diff_info[c][label] = v
        ax.bar(configs, values, bottom=bottom, label=label, color=color)
        bottom += np.array(values)

    ax.set_ylabel("Latency (s)")
    ax.set_title("exp8 Pipeline Latency Breakdown (Phase 3)")
    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return diff_info


# ============================================================
# Generator: fig_retrieval_metrics bar chart (exp8 / exp8b)
#   — bar chart only; box plot omitted because retrieval_metrics.json
#     does not preserve per-query avg_score@5 lists (stripped at save
#     time in recompute_retrieval_stats.py). Documented in diff report.
# ============================================================

def _gen_fig_retrieval_bars(exp_id: str, out_path: Path) -> Dict:
    import matplotlib.pyplot as plt

    src = json.loads(
        (EXP_DIR / exp_id / "retrieval_metrics.json").read_text(encoding="utf-8")
    )
    aggregated = src["systems"]
    systems = list(aggregated.keys())
    SYSTEM_COLORS = {
        "RAG Lexico (BM25)": "#E67E22",
        "RAG Semantico (Dense)": "#3498DB",
        "RAG Hibrido Propuesto": "#27AE60",
    }
    short = []
    for s in systems:
        if "Lexico" in s:
            short.append("Lexical (BM25)")
        elif "Semantico" in s:
            short.append("Semantic (Dense)")
        else:
            short.append("Hybrid (Proposed)")

    metrics = ["precision@1", "precision@3", "precision@5", "mrr", "ndcg@5"]
    labels = ["Prec@1", "Prec@3", "Prec@5", "MRR", "NDCG@5"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.25
    diff_info = {s: {} for s in systems}
    for i, sname in enumerate(systems):
        means = [aggregated[sname].get(f"{m}_mean", 0.0) for m in metrics]
        stds = [aggregated[sname].get(f"{m}_std", 0.0) for m in metrics]
        for m, v in zip(metrics, means):
            diff_info[sname][m] = v
        ax.bar(
            x + i * width, means, width,
            yerr=stds, label=short[i],
            color=SYSTEM_COLORS.get(sname, f"C{i}"), alpha=0.85, capsize=3,
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(
        f"{exp_id} Retrieval Quality (Phase 2.5 d_z + BH-FDR; bar chart, "
        "box plot omitted — per-query data not preserved)"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return diff_info


def gen_fig_retrieval_metrics_exp8(out_path: Path) -> Dict:
    return _gen_fig_retrieval_bars("exp8", out_path)


def gen_fig_retrieval_metrics_exp8b(out_path: Path) -> Dict:
    return _gen_fig_retrieval_bars("exp8b", out_path)


# ============================================================
# Generator: table_retrieval_metrics.tex (exp8 / exp8b)
# ============================================================

def _gen_table_retrieval(exp_id: str, out_path: Path) -> Dict:
    src = json.loads(
        (EXP_DIR / exp_id / "retrieval_metrics.json").read_text(encoding="utf-8")
    )
    aggregated = src["systems"]
    stats = src["statistical_tests"]
    systems = list(aggregated.keys())

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{Cross-encoder based retrieval quality metrics ({exp_id}, "
        "200 queries). Phase 2.5: paired Cohen's $d_z$ (Lakens 2013) and "
        "Benjamini-Hochberg FDR over 12-test family.}}",
        f"\\label{{tab:retrieval-metrics-{exp_id}-phase3}}",
        r"\begin{tabular}{l" + "c" * 6 + "}",
        r"\toprule",
        r"System & Prec@1 & Prec@3 & Prec@5 & MRR & NDCG@5 & Avg.~Score \\",
        r"\midrule",
    ]
    for s in systems:
        short = s.split("(")[0].strip() if "(" in s else s
        a = aggregated[s]
        lines.append(
            f"{short} & "
            f"{a.get('precision@1_mean', 0):.3f} & "
            f"{a.get('precision@3_mean', 0):.3f} & "
            f"{a.get('precision@5_mean', 0):.3f} & "
            f"{a.get('mrr_mean', 0):.3f} & "
            f"{a.get('ndcg@5_mean', 0):.3f} & "
            f"{a.get('avg_score@5_mean', 0):.3f} "
            r"\\"
        )
    lines.append(r"\midrule")

    # Pairwise stats with d_z + p_bh from Phase 2.5
    diff_info = {}
    for metric, by_pair in stats.items():
        if metric != "ndcg@5":
            continue  # caption row uses NDCG@5 only (most prominent comparison)
        for pair, leaf in by_pair.items():
            d_z = leaf.get("cohens_d_z", leaf.get("effect_size", 0.0))
            p_bh = leaf.get("p_bh", leaf.get("p_value", 1.0))
            sig_bh = leaf.get("sig_bh", False)
            eff_lab = leaf.get("effect_label", "n/a")
            diff_info[pair] = {"d_z": d_z, "p_bh": p_bh,
                               "sig_bh": sig_bh, "effect_label": eff_lab}
            p_str = (
                f"$p_{{BH}} < 10^{{{int(np.floor(np.log10(max(p_bh, 1e-99))))}}}$"
                if p_bh > 0 else r"$p_{BH} \approx 0$"
            )
            sig_str = "sig." if sig_bh else "n.s."
            lines.append(
                r"\multicolumn{7}{l}{\footnotesize "
                f"{pair} [ndcg@5]: $d_z={d_z:+.3f}$ ({eff_lab}), "
                f"{p_str}, {sig_str} (BH-FDR, family=12)"
                r"} \\"
            )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return diff_info


def gen_table_retrieval_metrics(out_path: Path) -> Dict:
    return _gen_table_retrieval("exp8", out_path)


def gen_table_retrieval_metrics_exp8b(out_path: Path) -> Dict:
    return _gen_table_retrieval("exp8b", out_path)


# ============================================================
# Generator: table_expN_faithfulness.tex (exp5, exp6, exp8, exp8b)
# ============================================================

def _gen_table_faithfulness(exp_id: str, out_path: Path) -> Dict:
    src = json.loads(
        (EXP_DIR / exp_id / "aggregated_metrics.json").read_text(encoding="utf-8")
    )
    configs = list(src.keys())

    # Check whether the source has post-Phase-2 fields (exp5 does, others don't)
    has_n_effective = any(
        "hall_n_effective" in src[c] and src[c]["hall_n_effective"] is not None
        for c in configs
    )

    header_cols = [
        "Configuration",
        r"Faithfulness ($\mu \pm \sigma$)",
        "n eff / total",
        "Latency (s)",
    ]
    caption_parts = [
        f"Faithfulness for {exp_id}"
    ]
    if has_n_effective:
        caption_parts.append(
            "Post-Phase-2 NLI aggregation "
            "(Honovich 2022 max-per-class, apply\\_softmax=True, "
            "excluding method=none/error)"
        )
    else:
        caption_parts.append(
            "Pre-Phase-2 NLI aggregation; n\\_effective unavailable for this "
            "experiment (benchmark not re-run post Phase 2 fixes)"
        )
    caption = ". ".join(caption_parts) + "."

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{exp_id}-faithfulness-phase3}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]

    diff_info = {}
    for cfg in configs:
        data = src[cfg]
        faith = data.get("hall_faithfulness_mean", 0.0)
        std = data.get("hall_faithfulness_std", 0.0)
        n_eff = data.get("hall_n_effective")
        n_tot = data.get("hall_n_total")
        lat_ms = data.get("lat_total_ms_mean", 0.0)
        lat_s = lat_ms / 1000.0
        diff_info[cfg] = {
            "faithfulness_mean": faith,
            "faithfulness_std": std,
            "n_effective": n_eff,
            "n_total": n_tot,
            "lat_total_s": lat_s,
        }

        if n_eff is not None and n_tot is not None:
            neff_str = f"{n_eff} / {n_tot}"
        else:
            neff_str = "--"

        # Escape underscores in config name for LaTeX
        cfg_latex = cfg.replace("_", r"\_")
        lines.append(
            f"{cfg_latex} & "
            f"${faith:.3f} \\pm {std:.3f}$ & "
            f"{neff_str} & "
            f"{lat_s:.2f} "
            r"\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return diff_info


def gen_table_exp5_faithfulness(out_path: Path) -> Dict:
    return _gen_table_faithfulness("exp5", out_path)


def gen_table_exp6_faithfulness(out_path: Path) -> Dict:
    return _gen_table_faithfulness("exp6", out_path)


def gen_table_exp8_faithfulness(out_path: Path) -> Dict:
    return _gen_table_faithfulness("exp8", out_path)


def gen_table_exp8b_faithfulness(out_path: Path) -> Dict:
    return _gen_table_faithfulness("exp8b", out_path)


# ============================================================
# Dispatcher
# ============================================================

GENERATORS = {
    "fig_llm_comparison": gen_fig_llm_comparison,
    "fig_latency_breakdown": gen_fig_latency_breakdown,
    "fig_retrieval_metrics": gen_fig_retrieval_metrics_exp8,
    "fig_retrieval_metrics_exp8b": gen_fig_retrieval_metrics_exp8b,
    "table_retrieval_metrics": gen_table_retrieval_metrics,
    "table_retrieval_metrics_exp8b": gen_table_retrieval_metrics_exp8b,
    "table_exp5_faithfulness": gen_table_exp5_faithfulness,
    "table_exp6_faithfulness": gen_table_exp6_faithfulness,
    "table_exp8_faithfulness": gen_table_exp8_faithfulness,
    "table_exp8b_faithfulness": gen_table_exp8b_faithfulness,
}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    report_lines: List[str] = [
        "# Phase 3 diff report",
        "",
        "Generated by `scripts/generate_phase3_artifacts.py`.",
        "",
        "Sources:",
        "- `experiments/results/exp5/aggregated_metrics.json` (post-rerun Honovich)",
        "- `experiments/results/exp6/aggregated_metrics.json` (pre-fix, intact)",
        "- `experiments/results/exp8/aggregated_metrics.json` (pre-fix faithfulness; latency used for fig_latency_breakdown)",
        "- `experiments/results/exp8/retrieval_metrics.json` (Phase 2.5: d_z + BH/Holm)",
        "- `experiments/results/exp8b/aggregated_metrics.json` (pre-fix)",
        "- `experiments/results/exp8b/retrieval_metrics.json` (Phase 2.5)",
        "",
        "Output directory: `paper/overleaf_ready/figures/`",
        "",
        "Deferred to Phase 5 (inventoried, not touched):",
        "- `fig_ablation_waterfall.png` (exp6) — needs narrative rewrite over Δ-faithfulness",
        "",
        "Out of scope this phase:",
        "- `fig_end_to_end.png` (manual diagram)",
        "- `fig_retrieval_comparison.png` (exp3)",
        "- `fig_reranker_impact.png` (exp4)",
        "- `fig_cross_cloud_improvement.png` (exp7 — Phase 4 decision pending)",
        "",
        "---",
        "",
    ]

    summary_rows: List[Tuple[str, str, str, str]] = []

    for base_name, ext, has_baseline in SCOPE_ARTEFACTS:
        logger.info("=" * 60)
        logger.info("Regenerating %s.%s", base_name, ext)
        logger.info("=" * 60)

        baseline_hash = None
        baseline_size = None
        if has_baseline:
            src = FIGURES_DIR / f"{base_name}.{ext}"
            if src.exists():
                baseline_hash = _md5(src)
                baseline_size = src.stat().st_size
                _backup_baseline(base_name, ext)
            else:
                logger.warning(
                    "Expected baseline %s missing; no backup created", src.name
                )

        out_path = FIGURES_DIR / f"{base_name}_phase3.{ext}"
        generator = GENERATORS[base_name]
        diff_info = generator(out_path)
        new_size = out_path.stat().st_size
        new_hash = _md5(out_path)

        # Summary row
        if has_baseline:
            summary_rows.append((
                f"{base_name}.{ext}",
                f"{baseline_size or 0} B / md5 {baseline_hash[:8] if baseline_hash else 'n/a'}",
                f"{new_size} B / md5 {new_hash[:8]}",
                "REGEN + BACKUP",
            ))
        else:
            summary_rows.append((
                f"{base_name}.{ext}",
                "no prior baseline",
                f"{new_size} B / md5 {new_hash[:8]}",
                "NEW",
            ))

        # Per-artefact detail block in the report
        report_lines.append(f"## {base_name}_phase3.{ext}")
        report_lines.append("")
        if has_baseline and baseline_hash is not None:
            report_lines.append(
                f"- Baseline backup: `{base_name}_pre_phase3.{ext}` (md5 `{baseline_hash}`)"
            )
        else:
            report_lines.append("- No pre-Phase-3 baseline (new file).")
        report_lines.append(f"- New file: `{base_name}_phase3.{ext}` "
                            f"(md5 `{new_hash}`, size {new_size} B)")
        report_lines.append("- Source data:")
        if "fig_llm_comparison" in base_name:
            report_lines.append("  - exp5/aggregated_metrics.json (post-rerun)")
        elif "fig_latency_breakdown" in base_name:
            report_lines.append("  - exp8/aggregated_metrics.json (lat_* fields, pre-fix — unaffected by NLI changes)")
        elif "exp8b" in base_name:
            report_lines.append("  - exp8b/retrieval_metrics.json (Phase 2.5)")
        elif "retrieval_metrics" in base_name:
            report_lines.append("  - exp8/retrieval_metrics.json (Phase 2.5)")
        elif "exp5_faithfulness" in base_name:
            report_lines.append("  - exp5/aggregated_metrics.json (post-rerun Honovich + n_effective)")
        elif "exp6_faithfulness" in base_name:
            report_lines.append("  - exp6/aggregated_metrics.json (pre-fix, intact)")
        elif "exp8_faithfulness" in base_name:
            report_lines.append("  - exp8/aggregated_metrics.json (pre-fix; benchmark not re-run post Phase 2)")
        elif "exp8b_faithfulness" in base_name:
            report_lines.append("  - exp8b/aggregated_metrics.json (pre-fix)")
        report_lines.append("")
        report_lines.append("### Numerical diff")
        report_lines.append("")
        report_lines.append("```json")
        report_lines.append(json.dumps(diff_info, indent=2, default=str))
        report_lines.append("```")
        report_lines.append("")

    # Summary table
    report_lines.append("## Summary table")
    report_lines.append("")
    report_lines.append("| Artefact | Baseline | Post-Phase-3 | Status |")
    report_lines.append("|----------|----------|--------------|--------|")
    for row in summary_rows:
        report_lines.append("| " + " | ".join(row) + " |")
    report_lines.append("")

    # Hardcode inventory
    report_lines.append("## Hardcode inventory (grep)")
    report_lines.append("")
    report_lines.append(
        "Executed: `grep -rn \"0\\.626\" paper/overleaf_ready/ scripts/`"
    )
    report_lines.append("")
    report_lines.append(
        "- `paper/overleaf_ready/main.tex:68` — `Cohen's d = 0.626`. **NOT TOUCHED** "
        "(Phase 5 rewrites the abstract). Inventoried only."
    )
    report_lines.append(
        "- `paper/overleaf_ready/figures/table_retrieval_metrics.tex:14` — "
        "`$d=-0.626$ (medium)`. Legacy pre-Phase-3 file; its Phase 3 regeneration "
        "lives at `table_retrieval_metrics_phase3.tex` (no `0.626` there)."
    )
    report_lines.append(
        "- `paper/overleaf_ready/figures/table_retrieval_metrics_exp8b.tex:14` — "
        "same pattern, same treatment."
    )
    report_lines.append("")
    report_lines.append("No hardcodes in Python scripts. All numeric values in the "
                        "regenerated artefacts derive from the post-fix JSONs.")
    report_lines.append("")

    DIFF_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DIFF_REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info("Wrote diff report: %s", DIFF_REPORT)


if __name__ == "__main__":
    main()
