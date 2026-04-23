"""
Smoke test for Flag 76 fix in src/evaluation/results_exporter.py.

Verifies that lines 540 (fig_end_to_end) and 655 (_export_experiment_latex)
now raise KeyError with a diagnostic message when required retrieval metrics
are missing, instead of silently fabricating 0.

Run with:
    python scripts/audit/smoke_fix_flag76.py

Expected: prints "PASS" twice (one per code path). If either raises
something other than KeyError, or prints FAIL, the fix is broken.
"""

import sys
from pathlib import Path

# Repo-relative import
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.results_exporter import ResultsExporter  # noqa: E402


def test_export_experiment_latex_raises_on_missing_key() -> None:
    """_export_experiment_latex must raise KeyError on missing metric."""
    exporter = ResultsExporter(
        results_dir=str(REPO_ROOT / "experiments" / "results"),
        output_dir=str(REPO_ROOT / "paper" / "overleaf_ready"),
    )
    bad_data = {
        "config_a": {"config_name": "config_a"},  # no ret_ndcg@5_mean, etc.
    }
    try:
        exporter._export_experiment_latex("smoke_test_exp", bad_data)
    except KeyError as exc:
        msg = str(exc)
        assert "ret_ndcg@5_mean" in msg, f"Missing metric name in error: {msg}"
        assert "smoke_test_exp" in msg, f"Missing experiment id in error: {msg}"
        assert "config_a" in msg, f"Missing config name in error: {msg}"
        print("PASS: _export_experiment_latex raised KeyError with diagnostic")
        return
    raise AssertionError(
        "FAIL: _export_experiment_latex did NOT raise on missing key"
    )


def test_fig_end_to_end_raises_on_missing_metric() -> None:
    """fig_end_to_end must raise KeyError when retrieval_metrics are missing."""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        exp_dir = results_dir / "smoke_exp"
        exp_dir.mkdir(parents=True)
        results_path = exp_dir / "results.json"

        # Craft a results.json where every query result lacks retrieval_metrics
        fake = {
            "configs": {
                "RAG_Test": {
                    "results": [
                        {"query_id": "q1"},  # no retrieval_metrics, no error
                        {"query_id": "q2"},
                    ]
                }
            }
        }
        results_path.write_text(json.dumps(fake), encoding="utf-8")

        exporter = ResultsExporter(
            results_dir=str(results_dir),
            output_dir=tmp,
        )
        try:
            exporter.fig_end_to_end("smoke_exp")
        except KeyError as exc:
            msg = str(exc)
            assert "retrieval_metrics" in msg, f"Missing key hint: {msg}"
            assert "RAG_Test" in msg, f"Missing config name: {msg}"
            print("PASS: fig_end_to_end raised KeyError with diagnostic")
            return
        raise AssertionError(
            "FAIL: fig_end_to_end did NOT raise on missing retrieval_metrics"
        )


def test_extended_figures_raise_on_missing_metric() -> None:
    """
    Flag 76-extended: all aggregated-data figures must raise KeyError, not
    silently fabricate 0, when their required metrics are missing. Covers
    the 9 additional call-sites listed in paper/audit_findings_cc_addenda.md
    §A1 (lines 241, 287, 313, 314, 353, 410, 425, 463, 495).
    """
    exporter = ResultsExporter(
        results_dir=str(REPO_ROOT / "experiments" / "results"),
        output_dir=str(REPO_ROOT / "paper" / "overleaf_ready"),
    )

    # fig_chunking_heatmap: config present but missing ret_ndcg@5_mean.
    # Patch _load_aggregated to return the poisoned payload.
    original_loader = exporter._load_aggregated

    try:
        exporter._load_aggregated = lambda _: {
            "chunk_fixed_300": {"config_name": "chunk_fixed_300"},
        }
        try:
            exporter.fig_chunking_heatmap("smoke_exp")
        except KeyError as exc:
            assert "ret_ndcg@5_mean" in str(exc)
            assert "chunk_fixed_300" in str(exc)
            print("PASS: fig_chunking_heatmap raised on missing ret_ndcg@5_mean")
        else:
            raise AssertionError("FAIL: fig_chunking_heatmap did not raise")

        # fig_retrieval_comparison: method present but missing metric.
        exporter._load_aggregated = lambda _: {
            "bm25": {"config_name": "bm25"},
        }
        try:
            exporter.fig_retrieval_comparison("smoke_exp")
        except KeyError as exc:
            assert "ret_recall@5_mean" in str(exc) or "ret_precision" in str(exc) or "ret_ndcg@5_mean" in str(exc)
            assert "bm25" in str(exc)
            print("PASS: fig_retrieval_comparison raised on missing metric")
        else:
            raise AssertionError("FAIL: fig_retrieval_comparison did not raise")

        # fig_reranker_impact: config missing ret_ndcg@5_mean / ret_recall@5_mean.
        exporter._load_aggregated = lambda _: {
            "rerank_mini12": {"config_name": "rerank_mini12"},
        }
        try:
            exporter.fig_reranker_impact("smoke_exp")
        except KeyError as exc:
            assert "rerank_mini12" in str(exc)
            assert "Flag 69" in str(exc) or "retrieval" in str(exc).lower()
            print("PASS: fig_reranker_impact raised with audit cross-reference")
        else:
            raise AssertionError("FAIL: fig_reranker_impact did not raise")

        # fig_ablation_waterfall: missing stage.
        exporter._load_aggregated = lambda _: {
            "ablation_bm25_only": {"ret_ndcg@5_mean": 0.5},
            # Missing +dense, +reranker, +expansion, +normalization
        }
        try:
            exporter.fig_ablation_waterfall("smoke_exp")
        except KeyError as exc:
            assert "ablation_+dense" in str(exc)
            print("PASS: fig_ablation_waterfall raised on missing ablation stage")
        else:
            raise AssertionError("FAIL: fig_ablation_waterfall did not raise")

        # fig_llm_comparison: model missing a declared metric.
        exporter._load_aggregated = lambda _: {
            "llama3.1": {"hall_faithfulness_mean": 0.4},
            "mistral": {"hall_faithfulness_mean": 0.5},
            # missing ret_ndcg@5_mean, gen_f1_token_mean, gen_rouge_l_mean
        }
        try:
            exporter.fig_llm_comparison("smoke_exp")
        except KeyError as exc:
            assert "llama3.1" in str(exc) or "mistral" in str(exc)
            print("PASS: fig_llm_comparison raised on any missing metric (no silent skip)")
        else:
            raise AssertionError("FAIL: fig_llm_comparison did not raise")

        # fig_latency_breakdown: config missing a latency stage.
        exporter._load_aggregated = lambda _: {
            "cfg_a": {"lat_query_processing_ms_mean": 1.0},
        }
        try:
            exporter.fig_latency_breakdown("smoke_exp")
        except KeyError as exc:
            assert "cfg_a" in str(exc)
            assert "lat_" in str(exc)
            print("PASS: fig_latency_breakdown raised on missing latency stage")
        else:
            raise AssertionError("FAIL: fig_latency_breakdown did not raise")

        # fig_cross_cloud_improvement: config missing retrieval metric.
        exporter._load_aggregated = lambda _: {
            "cross_cloud_no_norm": {"config_name": "cross_cloud_no_norm"},
            "cross_cloud_with_norm": {"config_name": "cross_cloud_with_norm"},
        }
        try:
            exporter.fig_cross_cloud_improvement("smoke_exp")
        except KeyError as exc:
            assert "cross_cloud_no_norm" in str(exc) or "cross_cloud_with_norm" in str(exc)
            print("PASS: fig_cross_cloud_improvement raised on missing metric")
        else:
            raise AssertionError(
                "FAIL: fig_cross_cloud_improvement did not raise"
            )
    finally:
        exporter._load_aggregated = original_loader


if __name__ == "__main__":
    test_export_experiment_latex_raises_on_missing_key()
    test_fig_end_to_end_raises_on_missing_metric()
    test_extended_figures_raise_on_missing_metric()
    print("\nAll smoke tests passed.")
