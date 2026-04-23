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


if __name__ == "__main__":
    test_export_experiment_latex_raises_on_missing_key()
    test_fig_end_to_end_raises_on_missing_metric()
    print("\nAll smoke tests passed.")
