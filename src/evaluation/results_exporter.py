"""
Results Exporter for thesis publication-ready outputs.

Exports:
  - LaTeX tables (booktabs format)
  - Matplotlib figures (300 DPI, publication quality)
  - CSV files for reproducibility

Figures generated:
  1. fig_chunking_heatmap.png - NDCG@5 heatmap (strategy x size)
  2. fig_retrieval_comparison.png - Bar chart (BM25 vs Dense vs Hybrid)
  3. fig_reranker_impact.png - Before/after reranking
  4. fig_ablation_waterfall.png - Component contribution waterfall
  5. fig_llm_comparison.png - Radar chart (3 LLMs)
  6. fig_latency_breakdown.png - Stacked bar chart per stage
  7. fig_cross_cloud_improvement.png - Normalization improvement
  8. fig_end_to_end.png - Full system comparison with CIs
  9. fig_statistical_significance.png - P-value heatmap
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Thesis color palette
COLORS = {
    "primary": "#2196F3",      # Blue
    "secondary": "#4CAF50",    # Green
    "accent": "#FF9800",       # Orange
    "danger": "#F44336",       # Red
    "neutral": "#9E9E9E",      # Gray
    "dark": "#212121",         # Dark gray
    "lexical": "#F44336",      # Red for BM25
    "semantic": "#2196F3",     # Blue for Dense
    "hybrid": "#4CAF50",       # Green for Hybrid
}

DPI = 300
FIGSIZE_WIDE = (12, 6)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_SMALL = (8, 5)


class ResultsExporter:
    """Export experiment results to publication-ready formats."""

    def __init__(
        self,
        results_dir: str = None,
        output_dir: str = None,
    ):
        self.results_dir = Path(results_dir or (PROJECT_ROOT / "experiments" / "results"))
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.csv_dir = self.output_dir / "csv"

        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Data loading helpers
    # ============================================================

    def _load_aggregated(self, experiment_id: str) -> Dict:
        """Load aggregated metrics for an experiment."""
        path = self.results_dir / experiment_id / "aggregated_metrics.json"
        if not path.exists():
            logger.warning("No aggregated metrics for %s", experiment_id)
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_results(self, experiment_id: str) -> Dict:
        """Load full results for an experiment."""
        path = self.results_dir / experiment_id / "results.json"
        if not path.exists():
            logger.warning("No results for %s", experiment_id)
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    # ============================================================
    # LaTeX table export
    # ============================================================

    def export_latex_table(
        self,
        data: List[Dict],
        columns: List[str],
        headers: List[str],
        caption: str,
        label: str,
        filename: str,
        highlight_best: bool = True,
        highlight_col: int = None,
    ):
        """
        Export data as a LaTeX table with booktabs formatting.

        Args:
            data: List of row dicts
            columns: Keys to extract from each row
            headers: Column headers for the table
            caption: LaTeX caption
            label: LaTeX label (e.g., tab:retrieval)
            filename: Output filename (no extension)
            highlight_best: Bold the best value in each column
            highlight_col: Specific column index to highlight (None = all numeric)
        """
        output_path = self.tables_dir / f"{filename}.tex"

        lines = [
            r"\begin{table}[htbp]",
            r"  \centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            "  \\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
            r"    \toprule",
            "    " + " & ".join(headers) + r" \\",
            r"    \midrule",
        ]

        # Find best values for highlighting
        best_values = {}
        if highlight_best:
            for col_idx in range(1, len(columns)):
                values = []
                for row in data:
                    v = row.get(columns[col_idx])
                    if isinstance(v, (int, float)):
                        values.append(v)
                if values:
                    best_values[col_idx] = max(values)

        # Data rows
        for row in data:
            cells = []
            for col_idx, col in enumerate(columns):
                v = row.get(col, "")
                if isinstance(v, float):
                    formatted = f"{v:.4f}"
                    # Bold best value
                    if (
                        highlight_best
                        and col_idx in best_values
                        and abs(v - best_values[col_idx]) < 1e-6
                    ):
                        formatted = f"\\textbf{{{formatted}}}"
                    cells.append(formatted)
                else:
                    cells.append(str(v))
            lines.append("    " + " & ".join(cells) + r" \\")

        lines.extend([
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ])

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("LaTeX table saved: %s", output_path)

    # ============================================================
    # CSV export
    # ============================================================

    def export_csv(
        self,
        data: List[Dict],
        columns: List[str],
        headers: List[str],
        filename: str,
    ):
        """Export data as CSV."""
        output_path = self.csv_dir / f"{filename}.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for row in data:
                writer.writerow([row.get(col, "") for col in columns])

        logger.info("CSV saved: %s", output_path)

    # ============================================================
    # Figure generation
    # ============================================================

    def _setup_matplotlib(self):
        """Configure matplotlib for publication quality."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": FIGSIZE_WIDE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        })
        return plt

    def fig_chunking_heatmap(self, experiment_id: str = "exp1"):
        """Figure 1: NDCG@5 heatmap for chunking strategies."""
        plt = self._setup_matplotlib()
        import matplotlib.colors as mcolors

        data = self._load_aggregated(experiment_id)
        if not data:
            logger.warning("No data for %s heatmap", experiment_id)
            return

        strategies = ["fixed", "recursive", "semantic", "hierarchical", "adaptive"]
        sizes = [300, 500, 700]

        matrix = np.zeros((len(strategies), len(sizes)))
        for i, strategy in enumerate(strategies):
            for j, size in enumerate(sizes):
                key = f"chunk_{strategy}_{size}"
                if key in data:
                    matrix[i][j] = data[key].get("ret_ndcg@5_mean", 0)

        fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels([s.capitalize() for s in strategies])

        # Annotate
        for i in range(len(strategies)):
            for j in range(len(sizes)):
                text = f"{matrix[i][j]:.3f}"
                color = "white" if matrix[i][j] > matrix.max() * 0.7 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

        ax.set_xlabel("Chunk Size (tokens)")
        ax.set_ylabel("Chunking Strategy")
        ax.set_title("NDCG@5 by Chunking Strategy and Size")
        fig.colorbar(im, ax=ax, label="NDCG@5")

        path = self.figures_dir / "fig_chunking_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    def fig_retrieval_comparison(self, experiment_id: str = "exp3"):
        """Figure 2: Retrieval method bar chart."""
        plt = self._setup_matplotlib()

        data = self._load_aggregated(experiment_id)
        if not data:
            return

        methods = list(data.keys())
        metrics = ["ret_recall@5_mean", "ret_precision@5_mean", "ret_ndcg@5_mean", "ret_mrr_mean"]
        metric_labels = ["Recall@5", "Precision@5", "NDCG@5", "MRR"]

        x = np.arange(len(metric_labels))
        width = 0.8 / len(methods)

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        colors = [COLORS["lexical"], COLORS["semantic"], COLORS["hybrid"], COLORS["accent"]]

        for i, method in enumerate(methods):
            values = [data[method].get(m, 0) for m in metrics]
            offset = (i - len(methods) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=method, color=colors[i % len(colors)])

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ax.set_title("Retrieval Method Comparison")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        path = self.figures_dir / "fig_retrieval_comparison.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    def fig_reranker_impact(self, experiment_id: str = "exp4"):
        """Figure 3: Reranker impact comparison."""
        plt = self._setup_matplotlib()

        data = self._load_aggregated(experiment_id)
        if not data:
            return

        configs = list(data.keys())
        ndcg5 = [data[c].get("ret_ndcg@5_mean", 0) for c in configs]
        recall5 = [data[c].get("ret_recall@5_mean", 0) for c in configs]

        x = np.arange(len(configs))
        width = 0.35

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        ax.bar(x - width/2, ndcg5, width, label="NDCG@5", color=COLORS["primary"])
        ax.bar(x + width/2, recall5, width, label="Recall@5", color=COLORS["secondary"])

        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=15, ha="right")
        ax.set_ylabel("Score")
        ax.set_title("Reranker Model Impact on Retrieval Quality")
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        path = self.figures_dir / "fig_reranker_impact.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    def fig_ablation_waterfall(self, experiment_id: str = "exp6"):
        """Figure 4: Ablation study waterfall chart."""
        plt = self._setup_matplotlib()

        data = self._load_aggregated(experiment_id)
        if not data:
            return

        stages = [
            "ablation_bm25_only",
            "ablation_+dense",
            "ablation_+reranker",
            "ablation_+expansion",
            "ablation_+normalization",
        ]
        labels = ["BM25", "+Dense", "+Reranker", "+Expansion", "+Normalization"]

        values = [data.get(s, {}).get("ret_ndcg@5_mean", 0) for s in stages]
        deltas = [values[0]] + [values[i] - values[i-1] for i in range(1, len(values))]

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        cumulative = 0
        colors_list = []
        bottoms = []

        for i, delta in enumerate(deltas):
            if i == 0:
                bottoms.append(0)
                colors_list.append(COLORS["neutral"])
            else:
                bottoms.append(cumulative)
                colors_list.append(COLORS["secondary"] if delta >= 0 else COLORS["danger"])
            cumulative += delta

        ax.bar(labels, deltas, bottom=bottoms, color=colors_list, edgecolor="white")

        # Add value labels
        for i, (d, b) in enumerate(zip(deltas, bottoms)):
            y = b + d / 2
            label = f"{d:.3f}" if i == 0 else f"+{d:.3f}"
            ax.text(i, y, label, ha="center", va="center", fontsize=9, fontweight="bold")

        # Add total line
        for i, v in enumerate(values):
            ax.plot([i - 0.4, i + 0.4], [v, v], "k-", linewidth=0.5)

        ax.set_ylabel("NDCG@5")
        ax.set_title("Ablation Study: Component Contribution to Retrieval Quality")
        ax.grid(axis="y", alpha=0.3)

        path = self.figures_dir / "fig_ablation_waterfall.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    def fig_llm_comparison(self, experiment_id: str = "exp5"):
        """Figure 5: LLM comparison radar chart."""
        plt = self._setup_matplotlib()

        data = self._load_aggregated(experiment_id)
        if not data:
            return

        models = list(data.keys())
        metrics = [
            ("hall_faithfulness_mean", "Faithfulness"),
            ("ret_ndcg@5_mean", "NDCG@5"),
            ("gen_f1_token_mean", "F1 Token"),
            ("gen_rouge_l_mean", "ROUGE-L"),
        ]

        # Only use metrics that have data
        available_metrics = []
        for key, label in metrics:
            if any(data[m].get(key, 0) > 0 for m in models):
                available_metrics.append((key, label))

        if not available_metrics:
            logger.warning("No metrics data available for LLM comparison radar")
            return

        N = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE, subplot_kw=dict(polar=True))
        colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"]]

        for i, model in enumerate(models):
            values = [data[model].get(m[0], 0) for m in available_metrics]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m[1] for m in available_metrics])
        ax.set_title("LLM Model Comparison", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        path = self.figures_dir / "fig_llm_comparison.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    def fig_latency_breakdown(self, experiment_id: str = "exp8"):
        """Figure 6: Latency breakdown stacked bar chart."""
        plt = self._setup_matplotlib()

        data = self._load_aggregated(experiment_id)
        if not data:
            return

        configs = list(data.keys())
        stages = [
            ("lat_query_processing_ms_mean", "Query Processing"),
            ("lat_retrieval_ms_mean", "Retrieval"),
            ("lat_reranking_ms_mean", "Reranking"),
            ("lat_generation_ms_mean", "Generation"),
            ("lat_hallucination_check_ms_mean", "Hallucination Check"),
        ]

        fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
        bottom = np.zeros(len(configs))
        stage_colors = [COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                       COLORS["danger"], COLORS["neutral"]]

        for i, (key, label) in enumerate(stages):
            values = np.array([data[c].get(key, 0) for c in configs])
            ax.bar(configs, values, bottom=bottom, label=label, color=stage_colors[i])
            bottom += values

        ax.set_ylabel("Latency (ms)")
        ax.set_title("Pipeline Latency Breakdown")
        ax.legend(loc="upper right")
        ax.set_xticklabels(configs, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)

        path = self.figures_dir / "fig_latency_breakdown.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    def fig_cross_cloud_improvement(self, experiment_id: str = "exp7"):
        """Figure 7: Cross-cloud normalization improvement."""
        plt = self._setup_matplotlib()

        data = self._load_aggregated(experiment_id)
        if not data:
            return

        configs = list(data.keys())
        metrics = ["ret_recall@5_mean", "ret_ndcg@5_mean", "ret_mrr_mean"]
        metric_labels = ["Recall@5", "NDCG@5", "MRR"]

        fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
        x = np.arange(len(metric_labels))
        width = 0.35

        for i, config in enumerate(configs):
            values = [data[config].get(m, 0) for m in metrics]
            offset = (i - 0.5) * width
            color = COLORS["danger"] if "no_norm" in config else COLORS["secondary"]
            label = "Without Normalization" if "no_norm" in config else "With Normalization"
            ax.bar(x + offset, values, width, label=label, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Score")
        ax.set_title("Cross-Cloud Query Performance: Effect of Terminology Normalization")
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        path = self.figures_dir / "fig_cross_cloud_improvement.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    def fig_end_to_end(self, experiment_id: str = "exp8"):
        """Figure 8: End-to-end system comparison with CIs."""
        plt = self._setup_matplotlib()

        results_data = self._load_results(experiment_id)
        if not results_data:
            return

        configs = results_data.get("configs", {})
        config_names = list(configs.keys())

        # Collect per-query scores for CI computation
        metrics_to_plot = ["ndcg@5", "recall@5", "mrr"]
        metric_labels = ["NDCG@5", "Recall@5", "MRR"]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))
        colors = [COLORS["lexical"], COLORS["semantic"], COLORS["hybrid"]]

        for m_idx, (metric_key, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax = axes[m_idx] if len(metrics_to_plot) > 1 else axes
            means = []
            cis = []

            for c_idx, config_name in enumerate(config_names):
                query_results = configs[config_name].get("results", [])
                values = [
                    r.get("retrieval_metrics", {}).get(metric_key, 0)
                    for r in query_results
                    if not r.get("error")
                ]
                if values:
                    mean = np.mean(values)
                    std = np.std(values, ddof=1) if len(values) > 1 else 0
                    ci = 1.96 * std / np.sqrt(len(values))
                    means.append(mean)
                    cis.append(ci)
                else:
                    means.append(0)
                    cis.append(0)

            x = np.arange(len(config_names))
            ax.bar(x, means, yerr=cis, capsize=5,
                   color=colors[:len(config_names)], edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(config_names, rotation=15, ha="right", fontsize=8)
            ax.set_title(metric_label)
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("End-to-End System Comparison (with 95% CI)", fontsize=13)
        fig.tight_layout()

        path = self.figures_dir / "fig_end_to_end.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    def fig_statistical_significance(self, stat_results: List = None):
        """Figure 9: P-value heatmap for statistical tests."""
        plt = self._setup_matplotlib()

        if not stat_results:
            logger.warning("No statistical results for significance heatmap")
            return

        # Organize by system pair and metric
        metrics = sorted(set(r.metric_name for r in stat_results))
        systems = sorted(set(r.system_b for r in stat_results))

        matrix = np.ones((len(systems), len(metrics)))
        for r in stat_results:
            if r.system_b in systems and r.metric_name in metrics:
                i = systems.index(r.system_b)
                j = metrics.index(r.metric_name)
                matrix[i][j] = r.p_value

        fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=0.1, aspect="auto")

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_yticks(range(len(systems)))
        ax.set_yticklabels(systems)

        # Annotate with p-values
        for i in range(len(systems)):
            for j in range(len(metrics)):
                p = matrix[i][j]
                sig = "*" if p < 0.05 else ""
                text = f"{p:.3f}{sig}"
                color = "white" if p < 0.05 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

        ax.set_title("Statistical Significance (p-values)")
        fig.colorbar(im, ax=ax, label="p-value")

        path = self.figures_dir / "fig_statistical_significance.png"
        fig.savefig(path)
        plt.close(fig)
        logger.info("Figure saved: %s", path)

    # ============================================================
    # High-level export
    # ============================================================

    def export_experiment(self, experiment_id: str):
        """Export all outputs for a single experiment."""
        data = self._load_aggregated(experiment_id)
        if not data:
            return

        # CSV export
        configs = list(data.keys())
        if configs:
            first = data[configs[0]]
            all_keys = sorted(first.keys())
            csv_data = [data[c] for c in configs]
            self.export_csv(
                csv_data, all_keys, all_keys,
                f"{experiment_id}_metrics",
            )

        # LaTeX table
        self._export_experiment_latex(experiment_id, data)

        logger.info("Exported experiment %s", experiment_id)

    def _export_experiment_latex(self, experiment_id: str, data: Dict):
        """Generate LaTeX table for an experiment."""
        configs = list(data.keys())
        if not configs:
            return

        # Key metrics for table
        columns = ["config_name", "ret_ndcg@5_mean", "ret_recall@5_mean", "ret_mrr_mean"]
        headers = ["Configuration", "NDCG@5", "Recall@5", "MRR"]

        table_data = []
        for config in configs:
            row = {"config_name": config}
            for col in columns[1:]:
                row[col] = data[config].get(col, 0)
            table_data.append(row)

        self.export_latex_table(
            data=table_data,
            columns=columns,
            headers=headers,
            caption=f"Results for {experiment_id}",
            label=f"tab:{experiment_id}",
            filename=f"table_{experiment_id}",
        )

    def export_all(self, experiment_ids: List[str] = None):
        """Export everything for all experiments."""
        if experiment_ids is None:
            # Discover experiments
            experiment_ids = []
            if self.results_dir.exists():
                for p in sorted(self.results_dir.iterdir()):
                    if p.is_dir() and (p / "aggregated_metrics.json").exists():
                        experiment_ids.append(p.name)

        for exp_id in experiment_ids:
            logger.info("Exporting %s...", exp_id)
            self.export_experiment(exp_id)

        # Generate figures
        figure_methods = {
            "exp1": self.fig_chunking_heatmap,
            "exp3": self.fig_retrieval_comparison,
            "exp4": self.fig_reranker_impact,
            "exp5": self.fig_llm_comparison,
            "exp6": self.fig_ablation_waterfall,
            "exp7": self.fig_cross_cloud_improvement,
            "exp8": self.fig_end_to_end,
        }

        for exp_id, fig_method in figure_methods.items():
            if exp_id in experiment_ids:
                try:
                    fig_method(exp_id)
                except Exception as e:
                    logger.warning("Failed to generate figure for %s: %s", exp_id, e)

        # Latency breakdown (uses exp8 data)
        if "exp8" in experiment_ids:
            try:
                self.fig_latency_breakdown("exp8")
            except Exception as e:
                logger.warning("Failed to generate latency figure: %s", e)

        logger.info("All exports complete. Output dir: %s", self.output_dir)
