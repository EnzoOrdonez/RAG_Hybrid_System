"""exp17 — higher-power reanalysis of the SAME 25 cross-cloud queries (no new queries).

The per-query paired test (arm_stats) treats each query's faithfulness mean as one unit -> 25
units, and weights a 3-claim query the same as a 30-claim one. That throws away within-query
information. Here we reanalyze at CLAIM resolution while keeping the query pairing, which is a
legitimate power gain from the existing data (no invented evidence):

1. Binomial GLMM: supported(0/1) ~ arm + (1|query). The random query intercept is shared by
   both arms' claims in that query, so the arm fixed effect is the within-query contrast at
   claim resolution (a 30-claim query informs more than a 3-claim one). Reported: posterior
   mean/SD of the arm log-odds, odds ratio, one-sided P(effect>0).
2. Cluster bootstrap (frequentist cross-check): resample the 25 QUERIES with replacement (the
   valid independent unit -> no claim pseudoreplication), recompute the paired micro-averaged
   faithfulness diff (balanced - baseline) each time. 95% CI + one-sided p = frac(diff<=0).

Claim-level analyses are conditional on the model having asserted a genuine claim (declines and
vacuous answers contribute 0 claims and drop out); this answers "among asserted claims, are
balanced's more likely to be grounded?", complementing the per-query decline-aware means.

Usage: python scripts/compute_exp17_powered.py
Reads faithfulness_rows__{small,base}__vb_agree.json + faithfulness_rows__hhem.json.
Writes experiments/results/exp17_crosscloud_balanced/powered_reanalysis.{json,md}
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = PROJECT_ROOT / "experiments/results/exp17_crosscloud_balanced"
BASE, BAL = "baseline | granite4.1-8b", "balanced | granite4.1-8b"
VERIFIERS = [("small", "faithfulness_rows__small__vb_agree.json"),
             ("base", "faithfulness_rows__base__vb_agree.json"),
             ("hhem", "faithfulness_rows__hhem.json")]
N_BOOT, SEED = 10000, 42


def load_counts(fname):
    """Return {query_id: {arm: (supported, genuine)}} for queries with genuine>0 in either arm."""
    cfg = json.loads((EXP_DIR / fname).read_text(encoding="utf-8"))["configs"]
    base, bal = cfg[BASE], cfg[BAL]
    out = {}
    for q in set(base) | set(bal):
        bg = base.get(q, {}); lg = bal.get(q, {})
        out[q] = {"baseline": (int(bg.get("supported") or 0), int(bg.get("genuine") or 0)),
                  "balanced": (int(lg.get("supported") or 0), int(lg.get("genuine") or 0))}
    return out


def to_claim_frame(counts):
    """Expand per-(query,arm) (supported,genuine) into one row per genuine claim."""
    rows = []
    for q, arms in counts.items():
        for arm, (s, g) in arms.items():
            for i in range(g):
                rows.append({"query": q, "arm": 1 if arm == "balanced" else 0,
                             "y": 1 if i < s else 0})
    return pd.DataFrame(rows)


def glmm(df):
    """Binomial GLMM y ~ arm + (1|query) via variational Bayes. Returns dict or None."""
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        m = BinomialBayesMixedGLM.from_formula("y ~ arm", {"query": "0 + C(query)"}, df)
        r = m.fit_vb()
        names = list(r.model.exog_names)
        i = names.index("arm")
        mean, sd = float(r.fe_mean[i]), float(r.fe_sd[i])
        z = mean / sd if sd > 0 else 0.0
        p_gt0 = float(0.5 * (1 + math.erf(z / math.sqrt(2))))  # P(effect>0)
        return {"arm_logodds_mean": round(mean, 4), "arm_logodds_sd": round(sd, 4),
                "odds_ratio": round(math.exp(mean), 4), "z": round(z, 3),
                "p_one_sided_effect_gt0": round(1 - p_gt0, 5), "n_claims": int(len(df))}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def cluster_bootstrap(counts, n_boot=N_BOOT, seed=SEED):
    """Query-level cluster bootstrap of the paired micro-averaged faithfulness diff."""
    qids = list(counts)
    def micro(qsel):
        bs = bg = ls = lg = 0
        for q in qsel:
            s, g = counts[q]["baseline"]; bs += s; bg += g
            s, g = counts[q]["balanced"]; ls += s; lg += g
        fb = bs / bg if bg else float("nan")
        fl = ls / lg if lg else float("nan")
        return fb, fl
    fb0, fl0 = micro(qids)
    obs = fl0 - fb0
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        sel = [qids[i] for i in rng.integers(0, len(qids), len(qids))]
        fb, fl = micro(sel)
        if not (math.isnan(fb) or math.isnan(fl)):
            diffs.append(fl - fb)
    diffs = np.array(diffs)
    return {"micro_baseline": round(fb0, 4), "micro_balanced": round(fl0, 4),
            "micro_diff": round(obs, 4),
            "boot95": [round(float(np.percentile(diffs, 2.5)), 4),
                       round(float(np.percentile(diffs, 97.5)), 4)],
            "p_one_sided_diff_le0": round(float(np.mean(diffs <= 0)), 5),
            "n_boot": len(diffs)}


def main():
    results = {}
    for tag, fname in VERIFIERS:
        counts = load_counts(fname)
        df = to_claim_frame(counts)
        results[tag] = {"glmm": glmm(df), "cluster_bootstrap": cluster_bootstrap(counts)}

    out = {"experiment_id": "exp17_crosscloud_balanced",
           "analysis": "higher-power reanalysis of the same 25 queries at claim resolution",
           "models": {"glmm": "supported ~ arm + (1|query), binomial VB; keeps query pairing",
                      "cluster_bootstrap": "resample 25 queries; paired micro-faithfulness diff; seed 42"},
           "note": "claim-level is conditional on a genuine claim (declines/vacuous drop out).",
           "verifiers": results, "generated_by": "scripts/compute_exp17_powered.py"}
    (EXP_DIR / "powered_reanalysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    L = ["# exp17 — higher-power reanalysis (same 25 queries, claim resolution)", "",
         "GLMM = supported ~ arm + (1|query) (binomial VB, keeps pairing). Bootstrap = query-level "
         "cluster resample of paired micro-faithfulness diff (balanced - baseline), seed 42.", "",
         "| Verifier | micro base | micro bal | diff | boot95 | boot p(1-sided) | GLMM OR | GLMM p(1-sided) |",
         "|---|---|---|---|---|---|---|---|"]
    for tag, _ in VERIFIERS:
        r = results[tag]; cb = r["cluster_bootstrap"]; gm = r["glmm"]
        orr = gm.get("odds_ratio", "err"); gp = gm.get("p_one_sided_effect_gt0", "err")
        L.append(f"| {tag} | {cb['micro_baseline']} | {cb['micro_balanced']} | {cb['micro_diff']} | "
                 f"{cb['boot95']} | {cb['p_one_sided_diff_le0']} | {orr} | {gp} |")
    L += ["", "One-sided tests (H1: balanced > baseline). Claim-level conditional on a genuine claim; "
          "read with the per-query decline-aware arm_stats (declines drop here).",
          "balanced also has MORE genuine claims than baseline (less decline) — a separate gain the "
          "conditional analysis does not capture."]
    (EXP_DIR / "powered_reanalysis.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
