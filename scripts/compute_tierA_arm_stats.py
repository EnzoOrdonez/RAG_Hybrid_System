"""
Tier A (exp15_ablation_tierA) — paired arm-vs-baseline_repro faithfulness stats.

Question: does transforming the CONTEXT (reranker off, final top-k=3, context reversed,
lost-in-the-middle permutation) change answer faithfulness relative to baseline_repro,
holding the generator (granite4.1:8b, temp 0, seed 42) and the query set fixed?

This is NOT the between-scenario v4 family. It is a within-subject paired contrast
per query_id: baseline_repro vs each transform arm, on the shared 60-query subset.

Protocol (mirrors signed v4 building blocks in src/evaluation/statistical_analysis.py):
  - paired_comparison  -> Wilcoxon signed-rank (or paired t if all-normal); NaN pairs dropped
  - cohens_d           -> Cohen's d_z = mean(diff)/std(diff, ddof=1)
  - bootstrap of the paired mean difference, seed 42, 10000 resamples (percentile 95% CI)
  - Benjamini-Hochberg (fdr_bh) across the 4 arm-vs-baseline contrasts (the family here)

Decline handling (decline-aware, v4-consistent): a query whose faithfulness is None
(the model declined -> no genuine claim to verify) is dropped from that pair. Vacuous
answers (genuine==0 -> faithfulness 1.0, per pass_n) are kept as 1.0. n_paired and the
per-arm decline counts are reported so the reader sees exactly what was compared.

Usage:
  python scripts/compute_tierA_arm_stats.py --verifier small
  python scripts/compute_tierA_arm_stats.py --verifier base
Writes: experiments/results/exp15_ablation_tierA/arm_stats__{verifier}.{json,md}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.statistical_analysis import (  # noqa: E402
    paired_comparison, cohens_d, apply_multiple_comparison_correction)

DEFAULT_EXP_DIR = PROJECT_ROOT / "experiments" / "results" / "exp15_ablation_tierA"


def derive_layout(exp_dir, baseline_arm="baseline_repro"):
    """From results.json: baseline config name, ordered arm config names (baseline
    excluded), and a {config_name: det_3x_bool} map from probe_report. Baseline is the
    config whose scenario == baseline_arm; arms keep results.json order."""
    res = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
    probe = res.get("probe_report", {})
    baseline = None
    arms, det3x = [], {}
    for cname, c in res["configs"].items():
        scen = c.get("scenario", cname.split(" | ")[0])
        det3x[cname] = probe.get(scen, {}).get("determinism_3x_identical")
        if scen == baseline_arm:
            baseline = cname
        else:
            arms.append(cname)
    if baseline is None:
        sys.exit(f"no baseline arm '{baseline_arm}' in {exp_dir}/results.json")
    return baseline, arms, det3x


def paired_bootstrap_meandiff(a, b, n_boot=10000, seed=42):
    """Percentile 95% CI of mean(b - a) over paired resamples (seed 42)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    diff = b - a
    if len(diff) < 2:
        return None, None, float(np.mean(diff)) if len(diff) else None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    return (float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
            float(np.mean(diff)))


def pair(rows_base, rows_arm):
    """Return (base_vec, arm_vec, n_common, n_decl_base, n_decl_arm) over shared qids."""
    qids = sorted(set(rows_base) & set(rows_arm))
    bvec, avec = [], []
    n_decl_base = n_decl_arm = 0
    for q in qids:
        fb = rows_base[q].get("faithfulness")
        fa = rows_arm[q].get("faithfulness")
        if fb is None:
            n_decl_base += 1
        if fa is None:
            n_decl_arm += 1
        if fb is None or fa is None:
            continue
        bvec.append(float(fb))
        avec.append(float(fa))
    return bvec, avec, len(qids), n_decl_base, n_decl_arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="small", choices=["small", "base", "hhem"])
    ap.add_argument("--exp-dir", default=str(DEFAULT_EXP_DIR),
                    help="results dir (default: Tier A)")
    ap.add_argument("--baseline-arm", default="baseline_repro",
                    help="scenario name of the anchor arm (default baseline_repro; exp17 uses 'baseline')")
    args = ap.parse_args()
    EXP_DIR = Path(args.exp_dir)
    BASELINE, ARM_ORDER, DET_3X = derive_layout(EXP_DIR, args.baseline_arm)

    if args.verifier == "hhem":
        rows_path = EXP_DIR / "faithfulness_rows__hhem.json"
        vlabel, vvariant, vthresh = "HHEM grounding", "max_chunk", [0.5]
    else:
        rows_path = EXP_DIR / f"faithfulness_rows__{args.verifier}__vb_agree.json"
        vlabel, vvariant, vthresh = f"NLI {args.verifier}", "vb_agree", [0.7, 0.7]
    doc = json.loads(rows_path.read_text(encoding="utf-8"))
    cfgs = doc["configs"]
    if BASELINE not in cfgs:
        sys.exit(f"baseline config missing in {rows_path}")
    rows_base = cfgs[BASELINE]

    def mean_ffn(rows):
        vals = [r["faithfulness"] for r in rows.values() if r.get("faithfulness") is not None]
        return round(float(np.mean(vals)), 4) if vals else None, len(vals)

    base_mean, base_n = mean_ffn(rows_base)
    contrasts = []
    pvals = []
    for arm in ARM_ORDER:
        if arm not in cfgs:
            continue
        rows_arm = cfgs[arm]
        bvec, avec, n_common, ndb, nda = pair(rows_base, rows_arm)
        arm_mean, arm_n = mean_ffn(rows_arm)
        # paired stats: A = baseline, B = arm  -> diff = arm - baseline
        pc = paired_comparison(bvec, avec)
        d_z, d_lab = cohens_d(bvec, avec)
        lo, hi, mdiff = paired_bootstrap_meandiff(bvec, avec)
        pvals.append(pc["p_value"])
        contrasts.append({
            "arm": arm.split(" | ")[0],
            "det_3x": DET_3X.get(arm),
            "n_paired": pc["n"],
            "n_common_qids": n_common,
            "n_decline_baseline": ndb,
            "n_decline_arm": nda,
            "baseline_mean_ffn": base_mean,
            "arm_mean_ffn": arm_mean,
            "mean_diff_arm_minus_base": None if mdiff is None else round(mdiff, 4),
            "boot95_lo": None if lo is None else round(lo, 4),
            "boot95_hi": None if hi is None else round(hi, 4),
            "test": pc["test_name"],
            "statistic": round(pc["statistic"], 4),
            "p_value": round(pc["p_value"], 5),
            "cohens_d_z": round(d_z, 4),
            "d_label": d_lab,
        })

    p_bh, sig_bh = apply_multiple_comparison_correction(pvals, method="fdr_bh")
    for c, pb, sg in zip(contrasts, p_bh, sig_bh):
        c["p_bh"] = round(float(pb), 5)
        c["sig_bh"] = bool(sg)

    out = {
        "experiment_id": EXP_DIR.name,
        "verifier": args.verifier,
        "variant": vvariant, "thresholds": vthresh,
        "baseline": "baseline_repro", "n_queries_subset": 60,
        "baseline_mean_faithfulness": base_mean, "baseline_n_scored": base_n,
        "bh_family": "4 arm-vs-baseline_repro contrasts (fdr_bh)",
        "decline_rule": "None-faithfulness pairs dropped (decline-aware); vacuous kept as 1.0",
        "bootstrap": {"n_boot": 10000, "seed": 42, "ci": "percentile 95%"},
        "contrasts": contrasts,
        "generated_by": "scripts/compute_tierA_arm_stats.py",
    }
    (EXP_DIR / f"arm_stats__{args.verifier}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")

    # markdown
    thr = "τ0.5" if args.verifier == "hhem" else "τ0.7"
    nondet = [c["arm"] for c in contrasts if c["det_3x"] is False]
    L = [f"# {out['experiment_id']} — arm vs baseline_repro ({vlabel}, {vvariant} {thr})",
         "",
         f"baseline_repro mean faithfulness: **{base_mean}** (n={base_n} scored). "
         f"Decline-aware: None pairs dropped, vacuous=1.0. BH family = {len(contrasts)} contrasts.",
         "",
         "| Arm | det3x | n_pair | base | arm | Δ(arm-base) | boot95 | test p | d_z | p_BH | sig |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in contrasts:
        boot = f"[{c['boot95_lo']}, {c['boot95_hi']}]"
        L.append(
            f"| {c['arm']} | {c['det_3x']} | {c['n_paired']} | {c['baseline_mean_ffn']} | "
            f"{c['arm_mean_ffn']} | {c['mean_diff_arm_minus_base']} | {boot} | "
            f"{c['p_value']} | {c['cohens_d_z']} ({c['d_label']}) | {c['p_bh']} | "
            f"{'YES' if c['sig_bh'] else 'no'} |")
    n_sig = sum(c["sig_bh"] for c in contrasts)
    L += ["", f"**{n_sig}/{len(contrasts)} arm-vs-baseline contrasts significant (BH).**"]
    if nondet:
        L.append(f"det3x=False ({', '.join(nondet)}) => answers carry H5 cold/warm-cache "
                 "noise; paired Δ still valid (same session/queries) but weigh softly.")
    (EXP_DIR / f"arm_stats__{args.verifier}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
