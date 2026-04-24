"""
Smoke test for Phase 3.5 — NLI recompute over saved answers.

Runs AFTER:
    python scripts/recompute_nli_over_saved.py --all

Asserts:

  1. For exp in {exp6, exp8, exp8b}:
     experiments/results/{exp}/aggregated_metrics_pre_nli_fix.json exists
     (backup of the PRE-Phase-2 aggregated metrics).
  2. For every config in each of those experiments:
     aggregated_metrics.json carries hall_n_effective, hall_n_total,
     hall_n_excluded_none_error, hall_method_counts.
  3. hall_n_effective <= hall_n_total per config.
  4. Expected direction: hall_faithfulness_mean post-fix <= pre-fix for
     every config (Honovich aggregation + method="none" filter both
     remove sources of inflation). Anomalies (post > pre) are reported
     as warnings, not failures, and written to
     paper/phase3_5_anomalies.md for review.
  5. exp5 was NOT processed by Phase 3.5 (it was already post-fix):
     no aggregated_metrics_pre_nli_fix.json under exp5/.
  6. exp7 was NOT processed (Phase 4 decision pending):
     no aggregated_metrics_pre_nli_fix.json under exp7/.
  7. For every config: hall_method_counts totals equal hall_n_total
     (consistency of the diagnostic counter).

Run with:
    python scripts/audit/smoke_nli_recompute.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = REPO_ROOT / "experiments" / "results"
ANOMALIES_PATH = REPO_ROOT / "paper" / "phase3_5_anomalies.md"

PROCESSED = ("exp6", "exp8", "exp8b")
UNTOUCHED = ("exp5", "exp7")


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_1_backups_exist() -> None:
    missing = []
    for exp in PROCESSED:
        p = EXP_DIR / exp / "aggregated_metrics_pre_nli_fix.json"
        if not p.exists():
            missing.append(str(p.relative_to(REPO_ROOT)))
    assert not missing, f"missing pre_nli_fix backups: {missing}"
    print(f"PASS #1: all {len(PROCESSED)} _pre_nli_fix.json backups exist")


def assert_2_new_fields_present() -> None:
    required = {"hall_n_total", "hall_n_effective",
                "hall_n_excluded_none_error", "hall_method_counts"}
    missing: List[str] = []
    for exp in PROCESSED:
        agg = _load(EXP_DIR / exp / "aggregated_metrics.json")
        for cfg, data in agg.items():
            gaps = required - set(data.keys())
            if gaps:
                missing.append(f"{exp}/{cfg} missing: {sorted(gaps)}")
    assert not missing, f"new fields missing:\n  " + "\n  ".join(missing)
    print(f"PASS #2: every config in {PROCESSED} carries hall_n_effective / "
          "n_total / n_excluded / method_counts")


def assert_3_n_effective_le_total() -> None:
    for exp in PROCESSED:
        agg = _load(EXP_DIR / exp / "aggregated_metrics.json")
        for cfg, data in agg.items():
            n_eff = data["hall_n_effective"]
            n_tot = data["hall_n_total"]
            assert n_eff <= n_tot, (
                f"{exp}/{cfg} hall_n_effective={n_eff} > hall_n_total={n_tot}"
            )
    print("PASS #3: hall_n_effective <= hall_n_total for every config")


def assert_4_direction_post_le_pre() -> List[Dict]:
    """Returns list of anomalies (post > pre) rather than failing outright."""
    anomalies: List[Dict] = []
    all_comparisons: List[Dict] = []
    for exp in PROCESSED:
        pre = _load(EXP_DIR / exp / "aggregated_metrics_pre_nli_fix.json")
        post = _load(EXP_DIR / exp / "aggregated_metrics.json")
        for cfg in pre:
            pre_val = pre.get(cfg, {}).get("hall_faithfulness_mean")
            post_val = post.get(cfg, {}).get("hall_faithfulness_mean")
            if pre_val is None or post_val is None:
                continue
            comp = {
                "exp": exp,
                "config": cfg,
                "pre": pre_val,
                "post": post_val,
                "delta": post_val - pre_val,
            }
            all_comparisons.append(comp)
            if post_val > pre_val + 1e-9:
                anomalies.append(comp)
    total = len(all_comparisons)
    monotone = total - len(anomalies)
    print(f"PASS #4 (warn-only): post <= pre for {monotone}/{total} configs; "
          f"{len(anomalies)} anomalies recorded")
    if anomalies:
        lines = [
            "# Phase 3.5 anomalies — post-fix faithfulness > pre-fix",
            "",
            "Context: the expected direction is "
            "post-fix hall_faithfulness_mean <= pre-fix, because the Phase 2 "
            "fixes remove two sources of inflation (apply_softmax=True on "
            "NLI, filter method=none/error). Any config where post > pre "
            "warrants manual inspection — could indicate that the pre-fix "
            "number happened to be suppressed by the early-exit bug "
            "(Flag 138) voting in favor of contradiction over entailment.",
            "",
            "| Experiment | Config | pre | post | delta |",
            "|------------|--------|-----|------|-------|",
        ]
        for a in anomalies:
            lines.append(
                f"| {a['exp']} | {a['config']} | "
                f"{a['pre']:.4f} | {a['post']:.4f} | {a['delta']:+.4f} |"
            )
        ANOMALIES_PATH.parent.mkdir(parents=True, exist_ok=True)
        ANOMALIES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"       → details in {ANOMALIES_PATH.relative_to(REPO_ROOT)}")
    return anomalies


def assert_5_exp5_untouched() -> None:
    p = EXP_DIR / "exp5" / "aggregated_metrics_pre_nli_fix.json"
    assert not p.exists(), (
        f"exp5 was NOT in Phase 3.5 scope but a backup appeared at {p}"
    )
    # Also check exp5's aggregated_metrics still has hall_n_effective=190 for llama
    agg = _load(EXP_DIR / "exp5" / "aggregated_metrics.json")
    llama = agg.get("llm_llama3.1", {})
    assert llama.get("hall_n_effective") == 190, (
        f"exp5 llm_llama3.1 hall_n_effective={llama.get('hall_n_effective')} "
        "(expected 190 per Phase 2 rerun)"
    )
    print("PASS #5: exp5 untouched (no backup; llm_llama3.1 n_effective=190 stable)")


def assert_6_exp7_untouched() -> None:
    p = EXP_DIR / "exp7" / "aggregated_metrics_pre_nli_fix.json"
    assert not p.exists(), (
        f"exp7 was NOT in Phase 3.5 scope but a backup appeared at {p}"
    )
    print("PASS #6: exp7 untouched (no backup — Phase 4 decision pending)")


def assert_7_method_counts_sum() -> None:
    for exp in PROCESSED:
        agg = _load(EXP_DIR / exp / "aggregated_metrics.json")
        for cfg, data in agg.items():
            mc = data.get("hall_method_counts", {})
            total = sum(mc.values())
            expected = data["hall_n_total"]
            assert total == expected, (
                f"{exp}/{cfg} method_counts sum={total} != n_total={expected}"
            )
    print("PASS #7: hall_method_counts sum == hall_n_total for every config")


def main() -> None:
    assert_1_backups_exist()
    assert_2_new_fields_present()
    assert_3_n_effective_le_total()
    anomalies = assert_4_direction_post_le_pre()
    assert_5_exp5_untouched()
    assert_6_exp7_untouched()
    assert_7_method_counts_sum()
    print("\nAll Phase 3.5 smoke assertions passed "
          f"({'0 anomalies' if not anomalies else f'{len(anomalies)} anomalies reported, non-failing'})")


if __name__ == "__main__":
    main()
