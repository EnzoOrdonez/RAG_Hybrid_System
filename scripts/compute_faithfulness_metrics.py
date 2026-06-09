"""
Paired faithfulness statistics for the generation matrix (Nota 3).

Extends the retrieval stats pipeline (compute_retrieval_metrics.py) to the
answer-faithfulness side, reusing src.evaluation.statistical_analysis
(Wilcoxon signed-rank + Cohen's d_z + bootstrap CI + BH/Holm). Adds McNemar
for the BINARY metrics (honest_decline / no_evidence) where Wilcoxon is
inappropriate.

Declared BH families (one correction per family — decision: "familias por RQ"):
  * between-scenario (per model):   for each model, all scenario pairs.
  * between-model    (per scenario): for each scenario, all model pairs.
(The retrieval family lives in compute_retrieval_metrics.py.)

Config-name convention for the matrix: "<scenario> | <model>". Legacy configs
without " | " are treated as a single (unnamed) model so this script also runs
on exp8/exp8b.

Per audit §19.3 Flag 137, queries whose hallucination method is "none" or
"error" carry synthetic faithfulness and are EXCLUDED from the paired vectors
(pairwise, by query_id).

Usage:
  python scripts/compute_faithfulness_metrics.py --experiment exp12_matrix
  python scripts/compute_faithfulness_metrics.py --experiment exp8   # validation
"""

import argparse
import itertools
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("faithfulness_metrics")

EXCLUDED_METHODS = {"none", "error"}  # synthetic faithfulness; see Flag 137


def parse_config(name: str):
    """('<scenario> | <model>') -> (scenario, model); legacy -> (name, '(single)')."""
    if " | " in name:
        scen, model = name.split(" | ", 1)
        return scen.strip(), model.strip()
    return name.strip(), "(single)"


def faithfulness_of(hm: dict):
    """Read faithfulness from a per-query hallucination_metrics dict."""
    if not hm:
        return None
    v = hm.get("faithfulness", hm.get("faithfulness_score"))
    return float(v) if v is not None else None


def mcnemar(b: int, c: int):
    """McNemar paired binary test (continuity-corrected). Returns (chi2, p)."""
    from scipy.stats import chi2 as _chi2
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    return float(stat), float(_chi2.sf(stat, df=1))


def load_per_config(results_path: Path):
    """config_name -> query_id -> {faithfulness, method, honest_decline, no_evidence}."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    configs = data["configs"]
    out = {}
    for cname, cdata in configs.items():
        per_q = {}
        for r in cdata["results"]:
            hm = r.get("hallucination_metrics", {}) or {}
            method = hm.get("method", "none")
            per_q[r["query_id"]] = {
                "faithfulness": faithfulness_of(hm),
                "method": method,
                "honest_decline": r.get("is_honest_decline",
                                        hm.get("is_honest_decline")),
                "no_evidence": method == "no_evidence",
            }
        out[cname] = per_q
    return out


def aligned_vectors(per_config, ca, cb):
    """Aligned faithfulness lists over shared query_ids, excluding none/error."""
    qa, qb = per_config[ca], per_config[cb]
    a, b = [], []
    for qid in qa.keys() & qb.keys():
        ra, rb = qa[qid], qb[qid]
        if ra["method"] in EXCLUDED_METHODS or rb["method"] in EXCLUDED_METHODS:
            continue
        if ra["faithfulness"] is None or rb["faithfulness"] is None:
            continue
        a.append(ra["faithfulness"])
        b.append(rb["faithfulness"])
    return a, b


def binary_discordance(per_config, ca, cb, field):
    """McNemar discordant counts for a binary field over shared queries."""
    qa, qb = per_config[ca], per_config[cb]
    b = c = n = 0
    for qid in qa.keys() & qb.keys():
        va, vb = qa[qid].get(field), qb[qid].get(field)
        if va is None or vb is None:
            continue
        n += 1
        if va and not vb:
            b += 1
        elif vb and not va:
            c += 1
    return b, c, n


def run_family(per_config, pairs, label):
    """Run compare_systems over `pairs` and apply BH/Holm within the family."""
    from src.evaluation.statistical_analysis import (
        apply_corrections_to_results,
        compare_systems,
    )
    results, keys = [], []
    for ca, cb in pairs:
        a, b = aligned_vectors(per_config, ca, cb)
        if len(a) < 3:
            logger.warning("  [%s] %s vs %s: n=%d too small, skipped", label, ca, cb, len(a))
            continue
        if all(abs(x - y) < 1e-12 for x, y in zip(a, b)):
            # All paired diffs zero (e.g. sin_rag faithfulness 0-vs-0) -> Wilcoxon
            # returns NaN and would poison the family-wide BH correction. Skip.
            logger.warning("  [%s] %s vs %s: degenerate (all paired diffs 0), skipped", label, ca, cb)
            continue
        sr = compare_systems("faithfulness", ca, cb, a, b)
        results.append(sr)
        keys.append((ca, cb))
    if results:
        apply_corrections_to_results(results)
    nested = {}
    for sr, (ca, cb) in zip(results, keys):
        d = sr.to_dict()
        # McNemar on honest_decline if the field exists for these configs.
        hb, hc, hn = binary_discordance(per_config, ca, cb, "honest_decline")
        if hn and (hb + hc) > 0:
            stat, p = mcnemar(hb, hc)
            d["honest_decline_mcnemar"] = {"b": hb, "c": hc, "chi2": stat, "p": p}
        nested[f"{ca} vs {cb}"] = d
    return nested


def main():
    ap = argparse.ArgumentParser(description="Paired faithfulness statistics")
    ap.add_argument("--experiment", default="exp12_matrix")
    args = ap.parse_args()

    exp_dir = PROJECT_ROOT / "experiments" / "results" / args.experiment
    results_path = exp_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"Not found: {results_path}")

    per_config = load_per_config(results_path)
    configs = list(per_config.keys())
    parsed = {c: parse_config(c) for c in configs}
    scenarios = sorted({parsed[c][0] for c in configs})
    models = sorted({parsed[c][1] for c in configs})
    logger.info("Configs=%d scenarios=%s models=%s", len(configs), scenarios, models)

    # ---- Descriptive per config (effective faithfulness, excl none/error) ----
    systems = {}
    for c in configs:
        vals = [v["faithfulness"] for v in per_config[c].values()
                if v["method"] not in EXCLUDED_METHODS and v["faithfulness"] is not None]
        methods = defaultdict(int)
        for v in per_config[c].values():
            methods[v["method"]] += 1
        n_total = len(per_config[c])
        hd = [v["honest_decline"] for v in per_config[c].values() if v["honest_decline"] is not None]
        systems[c] = {
            "scenario": parsed[c][0], "model": parsed[c][1],
            "faithfulness_mean": float(np.mean(vals)) if vals else 0.0,
            "faithfulness_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_effective": len(vals), "n_total": n_total,
            "no_evidence_rate": methods.get("no_evidence", 0) / n_total if n_total else 0.0,
            "honest_decline_rate": (sum(1 for x in hd if x) / len(hd)) if hd else None,
            "method_counts": dict(methods),
        }

    # ---- Family B: between-scenario, per model ----
    fam_b_pairs = []
    for m in models:
        cfgs_m = [c for c in configs if parsed[c][1] == m]
        scen_to_cfg = {parsed[c][0]: c for c in cfgs_m}
        for s1, s2 in itertools.combinations(sorted(scen_to_cfg), 2):
            fam_b_pairs.append((scen_to_cfg[s1], scen_to_cfg[s2]))
    between_scenario = run_family(per_config, fam_b_pairs, "between-scenario")

    # ---- Family C: between-model, per scenario ----
    fam_c_pairs = []
    for s in scenarios:
        if s == "sin_rag":
            continue  # faithfulness 0-by-construction for no-RAG (N3): no meaningful between-model contrast
        cfgs_s = [c for c in configs if parsed[c][0] == s]
        model_to_cfg = {parsed[c][1]: c for c in cfgs_s}
        for m1, m2 in itertools.combinations(sorted(model_to_cfg), 2):
            fam_c_pairs.append((model_to_cfg[m1], model_to_cfg[m2]))
    between_model = run_family(per_config, fam_c_pairs, "between-model")

    out = {
        "experiment": args.experiment,
        "metric": "faithfulness",
        "excluded_methods": sorted(EXCLUDED_METHODS),
        "bh_families": {
            "between_scenario_per_model": len(between_scenario),
            "between_model_per_scenario": len(between_model),
        },
        "systems": systems,
        "statistical_tests": {
            "faithfulness__between_scenario": between_scenario,
            "faithfulness__between_model": between_model,
        },
    }
    out_path = exp_dir / "faithfulness_metrics.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False,
                   default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
        encoding="utf-8",
    )

    # ---- Console summary (ascii-safe) ----
    print("\n" + "=" * 78)
    print(f"FAITHFULNESS METRICS — {args.experiment}")
    print("=" * 78)
    print(f"{'Config':<40} {'faith_mean':>10} {'n_eff':>6} {'no_ev%':>7} {'decl%':>7}")
    print("-" * 78)
    for c in configs:
        s = systems[c]
        dr = "" if s["honest_decline_rate"] is None else f"{100*s['honest_decline_rate']:.1f}"
        print(f"{c[:40]:<40} {s['faithfulness_mean']:>10.3f} {s['n_effective']:>6} "
              f"{100*s['no_evidence_rate']:>6.1f} {dr:>7}")
    for fam_name, fam in (("BETWEEN-SCENARIO (per model)", between_scenario),
                          ("BETWEEN-MODEL (per scenario)", between_model)):
        print("\n" + "-" * 78)
        print(f"{fam_name}  [BH family size = {len(fam)}]")
        print("-" * 78)
        for pair, r in fam.items():
            mc = r.get("honest_decline_mcnemar")
            mc_s = f" | McNemar p={mc['p']:.3f}" if mc else ""
            print(f"  {pair[:54]:<54} d_z={r.get('effect_size',0):+.2f} "
                  f"p_bh={r.get('p_bh', r['p_value']):.4f} sig={r.get('sig_bh')}{mc_s}")
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()
