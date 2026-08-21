"""exp19b — the pre-registered primary: paired faithfulness delta + TOST, three verifiers.

Seccion de Claude Code — 2026-08-21 14:55 (hora local).

PRIMARY, fixed before the run: the paired difference in faithfulness between `claim_selected`
and `baseline_repro`, per query, decline-aware, read on NLI-small, NLI-base and HHEM.

THE FAMILY, DECLARED RATHER THAN IMPLIED. exp18 declared "3 arm-vs-baseline contrasts per
verifier". exp19b has ONE arm, so its family is of size one and Benjamini-Hochberg is the
identity: p_BH == p_raw. That is stated in the artifact instead of quietly reporting a "p_BH"
that corrected nothing. The three verifiers are TRIANGULATION, not a family -- the phase
standard since exp17, kept here rather than switched mid-phase so exp19b stays comparable
with exp17 and exp18.

TOST, band +/- 0.081, alpha 0.05. The band is the exp17 HHEM effect of provider balancing: a
PRE-EXISTING quantity from another experiment, blind to this contrast, not a threshold tuned
until something passes. Two-sided, as exp18's pre-registration. `tost` is IMPORTED from
compute_exp18_diagnosis.py so both experiments answer equivalence with the same code.

WHAT THIS SCRIPT REFUSES TO COMPUTE. exp18's selection bound is motivation for exp19b and
nothing else: it holds the answer fixed while a real selector changes it, so a "percent of the
margin recovered" would be a fabricated number. No such ratio is produced here, and
tests/test_exp19b_runner.py pins that.

Estimators are imported, never reimplemented: `paired_comparison`, `cohens_d` and the BH
correction come from src/evaluation/statistical_analysis.py, the signed v4 building blocks.

Usage: python scripts/compute_exp19b_stats.py [--exp-dir DIR] [--smoke]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
Writes <exp dir>/equivalence__{small,base,hhem}.{json,md}
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistical_analysis import (  # noqa: E402
    paired_comparison, cohens_d, apply_multiple_comparison_correction)

EXP_ID = "exp19b_anchored_selector"
EXP_DIR = PROJECT_ROOT / "experiments/results" / EXP_ID
BASELINE_ARM = "baseline_repro"
ARM = "claim_selected"
VERIFIERS = ["small", "base", "hhem"]
TOST_BAND = 0.081
TOST_ALPHA = 0.05
SEED = 42
BOOT = 10000

_spec = importlib.util.spec_from_file_location(
    "exp18_diag", PROJECT_ROOT / "scripts/compute_exp18_diagnosis.py")
_diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_diag)
tost = _diag.tost


def declared_family():
    """The BH family for exp19b, written down before the numbers exist."""
    return {
        "scope": "per_verifier",
        "size": 1,
        "contrasts": [f"{ARM} vs {BASELINE_ARM}"],
        "note": ("one contrast per verifier, so BH is the identity and p_BH == p_raw; "
                 "declared explicitly rather than reported as if a correction had been "
                 "applied. The three verifiers are triangulation, not a family."),
    }


def bh_adjust(pvalues, alpha=0.05):
    """Benjamini-Hochberg over a declared family. General on purpose.

    A family of one is a DECLARATION, not a reason to skip the machinery: keeping the general
    helper means the artifact for exp19b and the artifacts for the multi-arm experiments come
    out of the same code path.
    """
    ps = [p for p in pvalues if p is not None]
    if not ps:
        return list(pvalues)
    adj, _ = apply_multiple_comparison_correction(ps, method="fdr_bh", alpha=alpha)
    return [float(x) for x in adj]


def rows_path(exp_dir, verifier):
    return (exp_dir / "faithfulness_rows__hhem.json" if verifier == "hhem"
            else exp_dir / f"faithfulness_rows__{verifier}__vb_agree.json")


def paired_faithfulness(exp_dir, verifier):
    """(qids, baseline, arm, dropped) — decline-aware pairing, the v4-consistent rule.

    A query whose faithfulness is None on either side (the model declined, so there is no
    genuine claim to verify) is dropped from the pair and counted. Vacuous answers keep their
    1.0, per pass_n.
    """
    path = rows_path(exp_dir, verifier)
    if not path.exists():
        return None
    cfgs = json.loads(path.read_text(encoding="utf-8"))["configs"]
    by_arm = {}
    for cname, per_q in cfgs.items():
        by_arm[cname.split(" | ")[0]] = per_q
    if BASELINE_ARM not in by_arm or ARM not in by_arm:
        return None
    base, arm = by_arm[BASELINE_ARM], by_arm[ARM]
    qids, a, b, dropped = [], [], [], []
    for qid in base:
        if qid not in arm:
            continue
        fb, fa = base[qid].get("faithfulness"), arm[qid].get("faithfulness")
        if fb is None or fa is None:
            dropped.append(qid)
            continue
        qids.append(qid)
        a.append(float(fb))
        b.append(float(fa))
    return qids, a, b, dropped


def analyse(exp_dir, verifier, selection):
    got = paired_faithfulness(exp_dir, verifier)
    if got is None:
        return None
    qids, base, arm, dropped = got
    if len(qids) < 3:
        return {"verifier": verifier, "n_paired": len(qids), "error": "too few pairs"}

    diffs = np.array(arm, float) - np.array(base, float)
    test = paired_comparison(base, arm)
    d, d_label = cohens_d(base, arm)
    rng = np.random.default_rng(SEED)
    boot = np.array([float(diffs[rng.integers(0, len(diffs), len(diffs))].mean())
                     for _ in range(BOOT)])
    eq = tost(diffs, band=TOST_BAND, alpha=TOST_ALPHA)
    fam = declared_family()
    p_raw = test.get("p_value")
    p_bh = bh_adjust([p_raw])[0] if p_raw is not None else None

    return {
        "verifier": verifier, "arm": ARM, "baseline": BASELINE_ARM,
        "n_paired": len(qids), "n_dropped_decline_aware": len(dropped),
        "qids_dropped": dropped,
        # Means over the PAIRED set only, so they decompose exactly into mean_paired_diff.
        # compute_tierA_arm_stats.py reports per-arm means over every non-None query instead;
        # the two answer different questions and the names say which is which.
        "mean_baseline_paired": round(float(np.mean(base)), 4),
        "mean_arm_paired": round(float(np.mean(arm)), 4),
        "mean_paired_diff": round(float(diffs.mean()), 4),
        "boot95": [round(float(np.percentile(boot, 2.5)), 4),
                   round(float(np.percentile(boot, 97.5)), 4)],
        # Scalars picked out of paired_comparison rather than the raw dict: it returns numpy
        # scalars, which json refuses. The dry run caught that before the GPU was spent.
        "test": test.get("test_name"),
        "statistic": round(float(test["statistic"]), 4) if test.get("statistic") is not None else None,
        "d_z": round(float(d), 4), "d_label": d_label,
        "p_raw": round(float(p_raw), 5) if p_raw is not None else None,
        "p_BH": round(float(p_bh), 5) if p_bh is not None else None, "bh_family": fam,
        "equivalence_tost": eq,
        "n_fallback_queries": selection.get("n_fallback"),
        "fallback_note": ("queries whose draft asserted nothing keep the baseline top-5, so "
                          "they contribute an exact zero difference and shrink the observable "
                          "effect; the count is reported so the reader can see the dilution"),
        "bound_note": ("exp18's selection bound motivated this arm and is NOT a denominator: "
                       "it holds the answer fixed while a real selector changes it, so any "
                       "percent-of-margin figure would be fabricated"),
    }


def render(res):
    eq = res.get("equivalence_tost") or {}
    L = [f"# exp19b — {res['arm']} vs {res['baseline']} ({res['verifier']})", "",
         f"n pareado **{res['n_paired']}** (descartadas por declinacion: "
         f"{res['n_dropped_decline_aware']}; queries en fallback: {res['n_fallback_queries']})",
         "",
         "| cantidad | valor |", "|---|---|",
         f"| fidelidad baseline (pareada) | {res['mean_baseline_paired']} |",
         f"| fidelidad {res['arm']} (pareada) | {res['mean_arm_paired']} |",
         f"| **diferencia pareada** | **{res['mean_paired_diff']}** |",
         f"| IC95 bootstrap | {res['boot95'][0]} a {res['boot95'][1]} |",
         f"| d_z | {res['d_z']} ({res['d_label']}) |",
         f"| p (crudo) | {res['p_raw']} |",
         f"| p_BH | {res['p_BH']} |", "",
         "**Familia BH declarada:** " + res["bh_family"]["note"], ""]
    if eq:
        L += [f"**TOST (banda ±{TOST_BAND}):** p_TOST={eq.get('p_tost')} · "
              f"IC90 {eq.get('ci90')} · equivalente={eq.get('equivalent')}", ""]
    L += [res["fallback_note"], "", res["bound_note"], ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir) if args.exp_dir else (
        EXP_DIR / "_smoke" if args.smoke else EXP_DIR)

    sel_path = exp_dir / "selection_ids.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}

    wrote = 0
    for v in VERIFIERS:
        res = analyse(exp_dir, v, selection)
        if res is None:
            print(f"[{v}] no paired rows yet ({rows_path(exp_dir, v).name} missing) — skipped")
            continue
        (exp_dir / f"equivalence__{v}.json").write_text(
            json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        (exp_dir / f"equivalence__{v}.md").write_text(render(res), encoding="utf-8")
        print(render(res))
        wrote += 1
    if wrote == 0:
        sys.exit("no verifier had paired rows — score the arms before reading the primary")
    if wrote != len(VERIFIERS):
        sys.exit(f"INCOMPLETE: {wrote} of {len(VERIFIERS)} verifiers analysed. The primary is "
                 f"declared on all three; reading a subset is choosing an instrument after "
                 f"seeing the data.")


if __name__ == "__main__":
    main()
