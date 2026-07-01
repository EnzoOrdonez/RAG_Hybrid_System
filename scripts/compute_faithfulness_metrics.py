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

v2 (ledger N5, 2026-06-11) — decline-aware denominators
-------------------------------------------------------
The v1 metric averaged faithfulness over every query with claims, MIXING real
answers with honest-decline texts whose explanatory sentences also yield NLI
claims. The bias is model-asymmetric (deflates granite 0.202->0.316, inflates
qwen 0.306->0.268 on the hybrid arm), so v1 is kept only as a labeled
sensitivity. Because `is_honest_decline` itself mislabels in both directions
(unmarked refusal openers; long cited partial answers flagged as declines),
v2 re-classifies every response at the ANALYSIS layer into:

  * pure_decline   — refusal marker inside the first OPENING_WINDOW chars
                     (the model led with a refusal);
  * hedged_partial — refusal marker only later in the text (substantive
                     attempt that hedges mid-answer);
  * answered       — no refusal marker anywhere.

PRIMARY metric `faithfulness_answered` = mean over answered + hedged_partial
(i.e. excludes only pure_decline). Sensitivities: (a) exclude v1 flag,
(b) exclude any-marker (strict), (c) v1 all-with-claims (published).
Paired tests run on the INTERSECTION of non-excluded rows per pair; per-pair n
is reported, with a power note when n < 60.

This script never rewrites the published v1 artifact: v1 output is only
written with --write-v1; the default emits faithfulness_metrics_v2.json
alongside it (historical exp12_matrix files stay untouched).

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

# ---------------------------------------------------------------------------
# v2 decline classifier (analysis layer; ledger N5)
# ---------------------------------------------------------------------------
# Canonical runtime patterns (the 14 in response_formatter.DECLINE_PATTERNS)
# are imported lazily in classify_response so this module stays importable
# without src/ on path for legacy experiments.
#
# Extended markers: refusal variants observed in exp12 answers that the
# canonical list misses (validated 2026-06-11 against the 16 configs; they
# drive the v1 false-negative rate of ~18 % on mistral's "answered" rows).
EXTENDED_REFUSAL_PATTERNS = [
    r"there is no (information|mention|specific|detail)",
    r"does not (mention|cover|provide|contain|include|specify|describe|address)",
    r"do not (mention|cover|provide|contain|include|specify)",
    r"is not (covered|mentioned|available|present|described|addressed|included|provided|specified)",
    r"are not (covered|mentioned|available|described|included)",
    r"not (covered|mentioned|available|specified|detailed|explicitly stated) in",
    r"insufficient (information|context|detail)",
    r"not enough (information|detail|context)",
    r"i (was|am) (unable|not able) to (find|locate|provide)",
    r"could ?n[o']t find",
    r"cannot (provide|answer|determine|confirm|be determined)",
    r"no (specific |further |additional )?(information|details?|mention)\b",
    r"beyond the (scope|provided)",
    r"outside the (scope|provided)",
]

# Chars of the lowercased answer inspected for the PURE-decline call: a
# refusal marker inside this window means the model led with a refusal.
OPENING_WINDOW = 300

_REFUSAL_MARKERS = None  # canonical + extended, compiled lazily


def _refusal_markers():
    global _REFUSAL_MARKERS
    if _REFUSAL_MARKERS is None:
        import re
        try:
            from src.generation.response_formatter import DECLINE_PATTERNS
        except Exception:  # legacy runs without src on path
            DECLINE_PATTERNS = []
            logger.warning("response_formatter unavailable; v2 classifier uses extended patterns only")
        _REFUSAL_MARKERS = [re.compile(p) for p in
                            list(DECLINE_PATTERNS) + EXTENDED_REFUSAL_PATTERNS]
    return _REFUSAL_MARKERS


def classify_response(answer: str):
    """'pure_decline' | 'hedged_partial' | 'answered' (None for empty text)."""
    if not answer or not answer.strip():
        return None
    low = answer.lower()
    opening = low[:OPENING_WINDOW]
    if any(rx.search(opening) for rx in _refusal_markers()):
        return "pure_decline"
    if any(rx.search(low) for rx in _refusal_markers()):
        return "hedged_partial"
    return "answered"


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


def load_per_config(results_path: Path, faith_override: dict = None):
    """config_name -> query_id -> per-query record (v1 fields + v2 class/claims).

    faith_override (ledger N8 / v3): optional
    {config_name: {query_id: {faithfulness, supported, contradicted,
    unsupported, total_claims}}} from a v3 re-score. When present it REPLACES
    the per-query NLI faithfulness and claim counts; the answer text — hence
    the decline classification — is untouched. Configs/queries absent from the
    override fall back to the signed results.json values (e.g. sin_rag).
    """
    data = json.loads(results_path.read_text(encoding="utf-8"))
    configs = data["configs"]
    out = {}
    for cname, cdata in configs.items():
        ov = (faith_override or {}).get(cname, {})
        per_q = {}
        for r in cdata["results"]:
            hm = r.get("hallucination_metrics", {}) or {}
            method = hm.get("method", "none")
            o = ov.get(r["query_id"])
            per_q[r["query_id"]] = {
                "faithfulness": (float(o["faithfulness"]) if o else faithfulness_of(hm)),
                "method": method,
                "honest_decline": r.get("is_honest_decline",
                                        hm.get("is_honest_decline")),
                "no_evidence": method == "no_evidence",
                # v2 fields
                "class_v2": classify_response(r.get("answer")),
                "pure_decline": (classify_response(r.get("answer")) == "pure_decline"
                                 if (r.get("answer") or "").strip() else None),
                "total_claims": (o["total_claims"] if o else (hm.get("total_claims") or 0)),
                "supported_claims": (o["supported"] if o else (hm.get("supported_claims") or 0)),
                "contradicted_claims": (o["contradicted"] if o else (hm.get("contradicted_claims") or 0)),
                "unsupported_claims": (o["unsupported"] if o else (hm.get("unsupported_claims") or 0)),
            }
        out[cname] = per_q
    return out


# Denominator definitions (v2). Each maps a per-query record -> include?
# (on top of the always-on method/None exclusions).
def _incl_primary(rec):    # exclude only pure_decline (answered + hedged_partial)
    return rec["class_v2"] != "pure_decline"


def _incl_v1flag(rec):     # sensitivity (a): exclude runtime is_honest_decline
    return not rec["honest_decline"]


def _incl_strict(rec):     # sensitivity (b): exclude any refusal marker
    return rec["class_v2"] == "answered"


def _incl_all(rec):        # sensitivity (c) = published v1 (all-with-claims)
    return True


DENOMINATORS = {
    "primary_answered": _incl_primary,
    "sens_a_v1flag": _incl_v1flag,
    "sens_b_strict": _incl_strict,
    "sens_c_published": _incl_all,
}


def aligned_vectors(per_config, ca, cb, include=None):
    """Aligned faithfulness lists over shared query_ids, excluding none/error.

    `include`: optional per-record predicate (a DENOMINATORS entry). When set,
    a query enters the pair only if BOTH arms pass it (intersection of
    non-excluded rows), per the N5 pairing rule.
    """
    qa, qb = per_config[ca], per_config[cb]
    a, b = [], []
    for qid in qa.keys() & qb.keys():
        ra, rb = qa[qid], qb[qid]
        if ra["method"] in EXCLUDED_METHODS or rb["method"] in EXCLUDED_METHODS:
            continue
        if ra["faithfulness"] is None or rb["faithfulness"] is None:
            continue
        if include is not None and not (include(ra) and include(rb)):
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


def run_family(per_config, pairs, label, include=None, metric_label="faithfulness"):
    """Run compare_systems over `pairs` and apply BH/Holm within the family.

    With `include` set (v2), pairing is restricted to the intersection of
    non-excluded rows and each pair carries a power note when n < 60.
    """
    from src.evaluation.statistical_analysis import (
        apply_corrections_to_results,
        compare_systems,
    )
    results, keys = [], []
    for ca, cb in pairs:
        a, b = aligned_vectors(per_config, ca, cb, include=include)
        if len(a) < 3:
            logger.warning("  [%s] %s vs %s: n=%d too small, skipped", label, ca, cb, len(a))
            continue
        if all(abs(x - y) < 1e-12 for x, y in zip(a, b)):
            # All paired diffs zero (e.g. sin_rag faithfulness 0-vs-0) -> Wilcoxon
            # returns NaN and would poison the family-wide BH correction. Skip.
            logger.warning("  [%s] %s vs %s: degenerate (all paired diffs 0), skipped", label, ca, cb)
            continue
        sr = compare_systems(metric_label, ca, cb, a, b)
        results.append(sr)
        keys.append((ca, cb))
    if results:
        apply_corrections_to_results(results)
    nested = {}
    for sr, (ca, cb) in zip(results, keys):
        d = sr.to_dict()
        if include is not None and d.get("n", 0) < 60:
            d["power_note"] = (f"n={d.get('n')} < 60 tras excluir declinaciones en ambos "
                               "brazos: potencia limitada, leer el d_z con cautela")
        # McNemar on the binary decline fields over ALL shared rows (not the
        # faithfulness-filtered intersection): decline behaviour itself.
        hb, hc, hn = binary_discordance(per_config, ca, cb, "honest_decline")
        if hn and (hb + hc) > 0:
            stat, p = mcnemar(hb, hc)
            d["honest_decline_mcnemar"] = {"b": hb, "c": hc, "chi2": stat, "p": p}
        pb, pc, pn = binary_discordance(per_config, ca, cb, "pure_decline")
        if pn and (pb + pc) > 0:
            stat, p = mcnemar(pb, pc)
            d["pure_decline_v2_mcnemar"] = {"b": pb, "c": pc, "chi2": stat, "p": p}
        nested[f"{ca} vs {cb}"] = d
    return nested


def main():
    ap = argparse.ArgumentParser(description="Paired faithfulness statistics")
    ap.add_argument("--experiment", default="exp12_matrix")
    ap.add_argument("--write-v1", action="store_true",
                    help="Also (re)write the legacy faithfulness_metrics.json. "
                         "Default off so the published v1 artifact stays untouched.")
    ap.add_argument("--faithfulness-source", default=None,
                    help="v3 re-score JSON (faithfulness_rescore_v3__*.json). When set, "
                         "overrides per-query NLI faithfulness/counts and writes "
                         "faithfulness_metrics_v3.json (ledger N8). v1/v2 artifacts untouched.")
    ap.add_argument("--out-tag", default=None,
                    help="Override the output suffix, e.g. v3_small -> "
                         "faithfulness_metrics_v3_small.json (default: v3 when "
                         "--faithfulness-source is set, else v2).")
    args = ap.parse_args()

    exp_dir = PROJECT_ROOT / "experiments" / "results" / args.experiment
    results_path = exp_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"Not found: {results_path}")

    faith_override = None
    out_tag = "v2"
    if args.faithfulness_source:
        src = json.loads(Path(args.faithfulness_source).read_text(encoding="utf-8"))
        faith_override = src.get("configs", src)
        out_tag = "v3"
    if args.out_tag:
        out_tag = args.out_tag

    per_config = load_per_config(results_path, faith_override)
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

    # ---- v2 descriptive per config: 4 denominators + decline census ----
    systems_v2 = {}
    for c in configs:
        recs = [v for v in per_config[c].values()
                if v["method"] not in EXCLUDED_METHODS and v["faithfulness"] is not None]
        entry = {"scenario": parsed[c][0], "model": parsed[c][1],
                 "n_total": len(per_config[c])}
        for dname, pred in DENOMINATORS.items():
            vals = [r["faithfulness"] for r in recs if pred(r)]
            entry[dname] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        census = defaultdict(int)
        for v in per_config[c].values():
            census[v["class_v2"] or "(empty)"] += 1
        n_classified = sum(n for k, n in census.items() if k != "(empty)")
        entry["decline_census_v2"] = {
            **{k: census.get(k, 0) for k in
               ("answered", "hedged_partial", "pure_decline", "(empty)")},
            "pure_decline_rate": (census.get("pure_decline", 0) / n_classified
                                  if n_classified else None),
        }
        entry["honest_decline_rate_v1"] = systems[c]["honest_decline_rate"]
        systems_v2[c] = entry

    # ---- Claims breakdown per config (f3): instrument-level view ----
    claims_breakdown = {}
    for c in configs:
        nli = [v for v in per_config[c].values()
               if v["method"] == "nli" and v["total_claims"] > 0]
        T = sum(v["total_claims"] for v in nli)
        if not T:
            continue
        claims_breakdown[c] = {
            "n_responses": len(nli),
            "total_claims": T,
            "claims_per_response": T / len(nli),
            "supported_pct": 100 * sum(v["supported_claims"] for v in nli) / T,
            "contradicted_pct": 100 * sum(v["contradicted_claims"] for v in nli) / T,
            "unsupported_pct": 100 * sum(v["unsupported_claims"] for v in nli) / T,
            # claim-level pooled support over the PRIMARY denominator rows
            "supported_pct_primary_rows": (
                lambda rows: (100 * sum(v["supported_claims"] for v in rows)
                              / max(1, sum(v["total_claims"] for v in rows)))
            )([v for v in nli if _incl_primary(v)]),
        }

    # ---- Families (v1, unchanged definition; only written with --write-v1) ----
    fam_b_pairs = []
    for m in models:
        cfgs_m = [c for c in configs if parsed[c][1] == m]
        scen_to_cfg = {parsed[c][0]: c for c in cfgs_m}
        for s1, s2 in itertools.combinations(sorted(scen_to_cfg), 2):
            fam_b_pairs.append((scen_to_cfg[s1], scen_to_cfg[s2]))
    fam_c_pairs = []
    for s in scenarios:
        if s == "sin_rag":
            continue  # faithfulness 0-by-construction for no-RAG (N3): no meaningful between-model contrast
        cfgs_s = [c for c in configs if parsed[c][0] == s]
        model_to_cfg = {parsed[c][1]: c for c in cfgs_s}
        for m1, m2 in itertools.combinations(sorted(model_to_cfg), 2):
            fam_c_pairs.append((model_to_cfg[m1], model_to_cfg[m2]))

    if args.write_v1:
        between_scenario = run_family(per_config, fam_b_pairs, "between-scenario")
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
        print("Wrote (v1):", out_path)

    # ---- Families on the v2 PRIMARY metric (intersection pairing) ----
    between_scenario_v2 = run_family(
        per_config, fam_b_pairs, "between-scenario-v2",
        include=_incl_primary, metric_label="faithfulness_answered")
    between_model_v2 = run_family(
        per_config, fam_c_pairs, "between-model-v2",
        include=_incl_primary, metric_label="faithfulness_answered")

    try:
        from src.generation.response_formatter import DECLINE_PATTERNS as _canon
    except Exception:
        _canon = []
    out_v2 = {
        "experiment": args.experiment,
        "metric": f"faithfulness_answered ({out_tag}, ledger {'N8' if out_tag == 'v3' else 'N5'})",
        "generated": "2026-06-11",
        "faithfulness_source": args.faithfulness_source,
        "excluded_methods": sorted(EXCLUDED_METHODS),
        "decline_classifier_v2": {
            "opening_window_chars": OPENING_WINDOW,
            "canonical_patterns": list(_canon),
            "extended_patterns": EXTENDED_REFUSAL_PATTERNS,
            "classes": {
                "pure_decline": "refusal marker within the first "
                                f"{OPENING_WINDOW} chars (model led with refusal)",
                "hedged_partial": "refusal marker only after the opening window",
                "answered": "no refusal marker anywhere",
            },
        },
        "denominators": {
            "primary_answered": "mean over answered+hedged_partial (excludes pure_decline only) — PRIMARY",
            "sens_a_v1flag": "sensitivity (a): excludes runtime is_honest_decline",
            "sens_b_strict": "sensitivity (b): excludes any refusal marker (answered only)",
            "sens_c_published": "sensitivity (c): v1 all-with-claims (published Tabla 6)",
        },
        "pairing_rule": "paired tests on the intersection of non-excluded rows in both arms; "
                        "power_note attached when n<60",
        "bh_families": {
            "between_scenario_per_model": len(between_scenario_v2),
            "between_model_per_scenario": len(between_model_v2),
        },
        "systems_v2": systems_v2,
        "claims_breakdown": claims_breakdown,
        "statistical_tests_v2": {
            "faithfulness_answered__between_scenario": between_scenario_v2,
            "faithfulness_answered__between_model": between_model_v2,
        },
    }
    out_path_v2 = exp_dir / f"faithfulness_metrics_{out_tag}.json"
    out_path_v2.write_text(
        json.dumps(out_v2, indent=2, ensure_ascii=False,
                   default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
        encoding="utf-8",
    )

    # ---- Console summary (ascii-safe) ----
    print("\n" + "=" * 96)
    print(f"FAITHFULNESS METRICS {out_tag} — {args.experiment}")
    print("=" * 96)
    print(f"{'Config':<34} {'PRIMARY':>8} {'n':>4} {'sensA':>7} {'sensB':>7} "
          f"{'pub':>6} {'n_pub':>6} {'pure%':>6} {'hedg%':>6}")
    print("-" * 96)
    for c in configs:
        s = systems_v2[c]
        cen = s["decline_census_v2"]
        ncl = max(1, s["n_total"] - cen["(empty)"])
        print(f"{c[:34]:<34} {s['primary_answered']['mean']:>8.3f} {s['primary_answered']['n']:>4} "
              f"{s['sens_a_v1flag']['mean']:>7.3f} {s['sens_b_strict']['mean']:>7.3f} "
              f"{s['sens_c_published']['mean']:>6.3f} {s['sens_c_published']['n']:>6} "
              f"{100*cen['pure_decline']/ncl:>6.1f} {100*cen['hedged_partial']/ncl:>6.1f}")
    for fam_name, fam in (("BETWEEN-SCENARIO v2 (per model)", between_scenario_v2),
                          ("BETWEEN-MODEL v2 (per scenario)", between_model_v2)):
        print("\n" + "-" * 96)
        print(f"{fam_name}  [BH family size = {len(fam)}]")
        print("-" * 96)
        for pair, r in fam.items():
            mc = r.get("pure_decline_v2_mcnemar")
            mc_s = f" | McNemar(v2) p={mc['p']:.3f}" if mc else ""
            pw = " [n<60]" if r.get("power_note") else ""
            print(f"  {pair[:50]:<50} n={r.get('n'):>3} d_z={r.get('effect_size',0):+.2f} "
                  f"p_bh={r.get('p_bh', r['p_value']):.4f} sig={r.get('sig_bh')}{pw}{mc_s}")
    print("\nWrote:", out_path_v2)


if __name__ == "__main__":
    main()
