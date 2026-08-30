"""exp18 — claim-level reanalysis of the three arm-vs-baseline contrasts.

Why this is a first-line reading here and not an extra: per-query faithfulness is
`supported/genuine`, and the arms carry very different denominators -- baseline 10.6,
oracle 10.7, swapped 4.8, top-10 15.0 genuine claims per response. A per-query mean weights
a 3-claim answer the same as a 30-claim one, so part of any per-query delta between arms is
a change of denominator rather than a change of grounding rate. The claim-level model keeps
the query pairing while letting each query contribute in proportion to what it asserted.

Estimators are IMPORTED from compute_exp17_powered.py, not reimplemented: the binomial GLMM
`supported ~ arm + (1|query)` and the query-level cluster bootstrap are the same validated
code, invoked once per contrast by mapping each exp18 arm onto the two-arm interface. Copying
them would be the duplication pattern that produced several silent defects in this phase.

DIRECTION. exp17 pre-specified a one-sided test (H1: balanced > baseline). exp18's
pre-registration (ledger entry 19) is TWO-SIDED for all three contrasts, so the one-sided
p-values are converted here and reported as such.

Claim-level analysis is conditional on the model having asserted a genuine claim, so declines
and vacuous answers drop out. That matters most for `evidence_swapped`, which declines in 88%
of its queries: the conditional result answers "among the claims it did assert, were they
grounded?", and must be read next to the per-query decline-aware arm_stats and the guards.

Usage: python scripts/compute_exp18_powered.py
Writes experiments/results/exp18_evidence_ceiling/powered_reanalysis.{json,md}
"""
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
BASELINE = "baseline_repro"
ARMS = ["oracle_evidence", "evidence_swapped", "final_top_k_10"]
VERIFIERS = [("small", "faithfulness_rows__small__vb_agree.json"),
             ("base", "faithfulness_rows__base__vb_agree.json"),
             ("hhem", "faithfulness_rows__hhem.json")]
MODEL = "granite4.1-8b"

_spec = importlib.util.spec_from_file_location(
    "p17", PROJECT_ROOT / "scripts" / "compute_exp17_powered.py")
p17 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p17)


def two_sided(p_one):
    """Convert a one-sided tail probability to two-sided, capped at 1."""
    if p_one is None or isinstance(p_one, str):
        return None
    return round(min(1.0, 2 * min(p_one, 1 - p_one)), 5)


def counts_for(rows_cfg, arm):
    """{query: {'baseline': (sup, gen), 'balanced': (sup, gen)}} for one contrast.

    The two generic key names are what compute_exp17_powered's estimators expect; 'balanced'
    is simply the arm under test. Only queries present in BOTH arms are kept, which is what
    makes the contrast paired.
    """
    base = rows_cfg[f"{BASELINE} | {MODEL}"]
    other = rows_cfg[f"{arm} | {MODEL}"]
    out = {}
    for q in sorted(set(base) & set(other)):
        b, o = base[q], other[q]
        out[q] = {"baseline": (int(b.get("supported") or 0), int(b.get("genuine") or 0)),
                  "balanced": (int(o.get("supported") or 0), int(o.get("genuine") or 0))}
    return out


def main():
    results = {}
    for tag, fname in VERIFIERS:
        cfg = json.loads((EXP_DIR / fname).read_text(encoding="utf-8"))["configs"]
        per_arm = {}
        for arm in ARMS:
            counts = counts_for(cfg, arm)
            df = p17.to_claim_frame(counts)
            gm = p17.glmm(df)
            cb = p17.cluster_bootstrap(counts)
            per_arm[arm] = {
                "n_queries_paired": len(counts),
                "n_claims": int(len(df)),
                "micro_baseline": cb["micro_baseline"],
                "micro_arm": cb["micro_balanced"],
                "micro_diff": cb["micro_diff"],
                "boot95": cb["boot95"],
                "boot_p_two_sided": two_sided(cb["p_one_sided_diff_le0"]),
                "glmm_odds_ratio": gm.get("odds_ratio"),
                "glmm_p_two_sided": two_sided(gm.get("p_one_sided_effect_gt0")),
            }
        results[tag] = per_arm

    out = {
        "experiment_id": "exp18_evidence_ceiling",
        "analysis": "claim-level reanalysis of the 3 arm-vs-baseline_repro contrasts",
        "baseline_arm": BASELINE, "arms": ARMS,
        "direction": "TWO-SIDED (ledger entry 19 pre-registration); exp17's one-sided "
                     "p-values are converted, not reused as-is",
        "estimators": "imported from scripts/compute_exp17_powered.py (same validated code)",
        "models": {"glmm": "supported ~ arm + (1|query), binomial VB; keeps query pairing",
                   "cluster_bootstrap": "resample queries; paired micro-faithfulness diff; seed 42"},
        "conditionality": ("claim-level is conditional on a genuine claim, so declines and "
                          "vacuous answers drop out. evidence_swapped declines in 88% of its "
                          "queries, so its conditional result answers only 'among the claims it "
                          "did assert, were they grounded?' -- read with arm_stats and guards."),
        "denominator_note": ("genuine claims per response differ sharply across arms "
                            "(baseline 10.6, oracle 10.7, swapped 4.8, top-10 15.0), which is "
                            "why the claim-level view is a first-line reading here."),
        "verifiers": results, "generated_by": "scripts/compute_exp18_powered.py",
    }
    (EXP_DIR / "powered_reanalysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    L = ["# exp18 — reanalisis claim-level (3 contrastes vs baseline_repro)", "",
         "GLMM = `supported ~ arm + (1|query)` (binomial VB, conserva el pareo). Bootstrap = "
         "remuestreo de cluster por query del diff micro-promediado, seed 42. **Tests "
         "BILATERALES** (pre-registro, entrada 19).", "",
         "| Verificador | Brazo | n_q | n_claims | micro base | micro brazo | diff | boot95 | boot p | GLMM OR | GLMM p |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, _ in VERIFIERS:
        for arm in ARMS:
            r = results[tag][arm]
            L.append(f"| {tag} | {arm} | {r['n_queries_paired']} | {r['n_claims']} | "
                     f"{r['micro_baseline']} | {r['micro_arm']} | {r['micro_diff']} | "
                     f"{r['boot95']} | {r['boot_p_two_sided']} | {r['glmm_odds_ratio']} | "
                     f"{r['glmm_p_two_sided']} |")
    L += ["", out["conditionality"], "", out["denominator_note"]]
    (EXP_DIR / "powered_reanalysis.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
