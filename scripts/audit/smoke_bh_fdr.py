"""
Smoke test for Flag 103/108 fix in src/evaluation/statistical_analysis.py.

Validates that apply_multiple_comparison_correction reproduces, to numerical
tolerance 1e-9, the BH and Holm-adjusted p-values pre-computed in
paper/audit_outputs/exp8_stats_corrected.csv (Audit Module 16, "m16"
recompute).

Per plan Paso 1.4: this smoke is autocontained on the CSV only — it does
NOT load experiments/results/exp8/retrieval_metrics.json. The CSV has
both p_raw (input) and p_bh/p_holm (expected output) in the same file.

Run with:
    python scripts/audit/smoke_bh_fdr.py

Exit 0 on success, raises on numerical mismatch.
"""

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.statistical_analysis import (  # noqa: E402
    apply_multiple_comparison_correction,
)

CSV_PATH = REPO_ROOT / "paper" / "audit_outputs" / "exp8_stats_corrected.csv"


def load_csv_columns():
    """Return (p_raw, p_bh, p_holm, sig_bh, sig_holm, row_labels)."""
    p_raw, p_bh, p_holm = [], [], []
    sig_bh, sig_holm = [], []
    labels = []
    with CSV_PATH.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            p_raw.append(float(row["p_raw"]))
            p_bh.append(float(row["p_bh"]))
            p_holm.append(float(row["p_holm"]))
            sig_bh.append(row["sig_bh"].strip().lower() == "true")
            sig_holm.append(row["sig_holm"].strip().lower() == "true")
            labels.append(
                f"{row['metric']}:{row['system_a']}->{row['system_b']}"
            )
    return p_raw, p_bh, p_holm, sig_bh, sig_holm, labels


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Expected reference CSV at {CSV_PATH}. This smoke cannot run "
            "without paper/audit_outputs/exp8_stats_corrected.csv."
        )

    p_raw, p_bh_expected, p_holm_expected, sig_bh_expected, sig_holm_expected, labels = (
        load_csv_columns()
    )
    n = len(p_raw)
    print(f"Loaded {n} comparisons from {CSV_PATH.name}")

    # BH-FDR
    p_bh_computed, sig_bh_computed = apply_multiple_comparison_correction(
        p_raw, method="fdr_bh", alpha=0.05
    )
    np.testing.assert_allclose(
        p_bh_computed,
        p_bh_expected,
        rtol=1e-9,
        err_msg="BH-FDR p-values diverge from CSV reference beyond rtol=1e-9",
    )
    assert list(sig_bh_computed) == sig_bh_expected, (
        "BH-FDR significance flags diverge:\n"
        f"  expected: {sig_bh_expected}\n"
        f"  computed: {sig_bh_computed}"
    )
    print(f"PASS: BH-FDR p_bh matches CSV for all {n} comparisons (rtol=1e-9)")

    # Holm (secondary check)
    p_holm_computed, sig_holm_computed = apply_multiple_comparison_correction(
        p_raw, method="holm", alpha=0.05
    )
    np.testing.assert_allclose(
        p_holm_computed,
        p_holm_expected,
        rtol=1e-9,
        err_msg="Holm p-values diverge from CSV reference beyond rtol=1e-9",
    )
    assert list(sig_holm_computed) == sig_holm_expected, (
        "Holm significance flags diverge:\n"
        f"  expected: {sig_holm_expected}\n"
        f"  computed: {sig_holm_computed}"
    )
    print(f"PASS: Holm p_holm matches CSV for all {n} comparisons (rtol=1e-9)")

    print("\nAll multiple-comparison corrections validated against m16 CSV.")


if __name__ == "__main__":
    main()
