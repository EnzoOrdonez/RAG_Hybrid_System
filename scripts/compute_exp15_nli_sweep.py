"""exp15 Tier 0 — variant x threshold sweep over the persisted raw NLI probs.

Pure-CPU re-aggregation: no GPU, no LLM, exp12 read-only. For every operating
point (variant, ent_t, contr_t) and each verifier it rebuilds the v3-format
faithfulness rows from nli_probs__<verifier>.json.gz, then applies the exact
v4 methodology (load_per_config + --exclude-vacuous semantics + primary_answered
+ paired families with BH) imported from scripts/compute_faithfulness_metrics.py
(functions only; its main() writes into signed dirs and is never invoked).

Readouts per point:
  * the 12 RAG-cell primary_answered means/n,
  * # significant among the 12 RAG-vs-RAG scenario pairs (the 0/12 question),
  * the granite hibrido-vs-lexico contrast (d_z, p_bh, sig),
  * small-vs-base claim-level agreement (Cohen's kappa, 3-class).

Sanity anchor: the point (vb_agree, 0.7, 0.7) must reproduce the signed v4
numbers exactly; the sweep aborts if it does not.

Outputs (experiments/results/exp15_ablation_nli/):
  sweep_results.json   per-point full summaries
  sweep_summary.md     robustness map (matrix tables)

Usage:
  python scripts/compute_exp15_nli_sweep.py
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import gzip
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import decide_nli_status  # noqa: E402

EXP12_DIR = PROJECT_ROOT / "experiments/results/exp12_matrix"      # READ-ONLY
OUT_DIR = PROJECT_ROOT / "experiments/results/exp15_ablation_nli"
RESULTS_PATH = EXP12_DIR / "results.json"

VARIANTS = [("v0", 0.0), ("vb_agree", 0.0), ("va_margin", 0.1), ("va_margin", 0.2)]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
# run_family keys pairs in sorted-scenario order: (hibrido, lexico), not (lexico, hibrido)
GRANITE_PAIR = "hibrido | granite4.1-8b vs lexico | granite4.1-8b"

spec = importlib.util.spec_from_file_location(
    "cfm", PROJECT_ROOT / "scripts" / "compute_faithfulness_metrics.py")
cfm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfm)


def load_probs(verifier):
    with gzip.open(OUT_DIR / f"nli_probs__{verifier}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)["configs"]


def aggregate(probs_cfgs, claims_cfgs, variant, margin, ent_t, contr_t):
    """Rebuild v3-format rows {config: {qid: rec}} for one operating point."""
    out = {}
    for cname, qids in probs_cfgs.items():
        rows = {}
        for qid, per_claim in qids.items():
            meta = claims_cfgs[cname][qid]
            total = len(meta["claims"])
            n_art = sum(1 for a in meta["artifact"] if a)
            g = total - n_art
            if not per_claim:  # vacuous: no genuine claim to score
                rows[qid] = {"total_claims": total, "not_a_claim": n_art, "genuine": 0,
                             "supported": 0, "contradicted": 0, "unsupported": 0,
                             "faithfulness": 1.0}
                continue
            agg = {"supported": 0, "contradicted": 0, "unsupported": 0}
            for chunk_probs in per_claim:
                contr = [p[0] for p in chunk_probs]
                ent = [p[1] for p in chunk_probs]
                st, _, _ = decide_nli_status(contr, ent, ent_t, contr_t,
                                             variant=variant, margin=margin)
                agg[st] += 1
            rows[qid] = {"total_claims": total, "not_a_claim": n_art, "genuine": g,
                         **agg, "faithfulness": round(agg["supported"] / g, 4)}
        out[cname] = rows
    return out


def evaluate_point(rows, label):
    """v4 methodology on one point: cells + between-scenario family."""
    cfm.EXCLUDED_METHODS.clear()
    cfm.EXCLUDED_METHODS.update({"none", "error", "vacuous"})
    per_config = cfm.load_per_config(RESULTS_PATH, rows, exclude_vacuous=True)
    configs = list(per_config.keys())
    parsed = {c: cfm.parse_config(c) for c in configs}
    models = sorted({parsed[c][1] for c in configs})

    cells = {}
    for c in configs:
        if parsed[c][0] == "sin_rag":
            continue
        recs = [v for v in per_config[c].values()
                if v["method"] not in cfm.EXCLUDED_METHODS and v["faithfulness"] is not None]
        vals = [r["faithfulness"] for r in recs if cfm._incl_primary(r)]
        cells[c] = {"mean": round(float(np.mean(vals)), 4) if vals else 0.0, "n": len(vals)}

    fam_b_pairs = []
    for m in models:
        cfgs_m = [c for c in configs if parsed[c][1] == m]
        scen_to_cfg = {parsed[c][0]: c for c in cfgs_m}
        for s1, s2 in itertools.combinations(sorted(scen_to_cfg), 2):
            fam_b_pairs.append((scen_to_cfg[s1], scen_to_cfg[s2]))
    fam_b = cfm.run_family(per_config, fam_b_pairs, label,
                           include=cfm._incl_primary, metric_label="faithfulness_answered")

    rag_pairs = {p: r for p, r in fam_b.items() if "sin_rag" not in p}
    sig_rag = sorted(p for p, r in rag_pairs.items() if r.get("sig_bh"))
    gp = fam_b.get(GRANITE_PAIR, {})
    return {
        "cells": cells,
        "n_rag_pairs": len(rag_pairs),
        "sig_rag_pairs": sig_rag,
        # d_z sign (cohens_d: diff = b - a = second - first): NEGATIVE = hibrido better
        "granite_hib_vs_lex": {"d_z": round(gp.get("effect_size", float("nan")), 4),
                               "p_bh": round(gp.get("p_bh", float("nan")), 6),
                               "n": gp.get("n"), "sig_bh": bool(gp.get("sig_bh"))},
        "rag_pair_stats": {p: {"d_z": round(r.get("effect_size", 0), 4),
                               "p_bh": round(r.get("p_bh", 1), 6), "n": r.get("n")}
                           for p, r in rag_pairs.items()},
    }


def claim_kappa(rows_small, rows_base, probs_small, probs_base, claims_cfgs,
                variant, margin, ent_t, contr_t):
    """Cohen's kappa (3-class) small-vs-base over aligned genuine claims."""
    cats = {"supported": 0, "contradicted": 1, "unsupported": 2}
    a, b = [], []
    for cname in probs_small:
        for qid in probs_small[cname]:
            ps, pb = probs_small[cname][qid], probs_base.get(cname, {}).get(qid)
            if pb is None or len(ps) != len(pb):
                continue
            for cs, cb in zip(ps, pb):
                for probs, dest in ((cs, a), (cb, b)):
                    st, _, _ = decide_nli_status([p[0] for p in probs], [p[1] for p in probs],
                                                 ent_t, contr_t, variant=variant, margin=margin)
                    dest.append(cats[st])
    a, b = np.array(a), np.array(b)
    if not len(a):
        return None, 0
    po = float((a == b).mean())
    pe = sum(float((a == k).mean()) * float((b == k).mean()) for k in range(3))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return round(kappa, 4), int(len(a))


def main():
    t0 = time.time()
    claims_cfgs = json.loads((OUT_DIR / "claims_extraction.json").read_text(encoding="utf-8"))["configs"]
    probs = {v: load_probs(v) for v in ("small", "base")}

    # ---- sanity anchor: canonical point must reproduce signed v4 -----------
    for verifier, tag in (("small", "v4_small"), ("base", "v4")):
        rows = aggregate(probs[verifier], claims_cfgs, "vb_agree", 0.0, 0.7, 0.7)
        signed = json.loads(
            (EXP12_DIR / f"faithfulness_rescore_v3__{verifier}__vb_agree.json").read_text(
                encoding="utf-8"))["configs"]
        n_bad = 0
        for cname, qrows in rows.items():
            for qid, rec in qrows.items():
                s = signed[cname].get(qid)
                if s is None or any(rec[k] != s[k] for k in
                                    ("genuine", "supported", "contradicted", "unsupported",
                                     "faithfulness")):
                    n_bad += 1
        if n_bad:
            sys.exit(f"ANCHOR FAIL {verifier}: {n_bad} rows differ from signed rescore "
                     "at (vb_agree, 0.7, 0.7) — sweep aborted, investigate before trusting it.")
        print(f"anchor {verifier}: all rows == signed rescore at canonical point", flush=True)

    # ---- the sweep ----------------------------------------------------------
    sweep = {"generated_by": "scripts/compute_exp15_nli_sweep.py",
             "grid": {"variants": [f"{v}{'' if m == 0 else f'_d{m}'}" for v, m in VARIANTS],
                      "thresholds": THRESHOLDS},
             "canonical_point": "vb_agree ent0.7 contr0.7 (== signed v4)",
             "points": []}
    for (variant, margin), ent_t, contr_t in itertools.product(VARIANTS, THRESHOLDS, THRESHOLDS):
        vtag = f"{variant}{'' if margin == 0 else f'_d{margin}'}"
        point = {"variant": vtag, "ent_t": ent_t, "contr_t": contr_t, "verifiers": {}}
        rows_by_verifier = {}
        for verifier in ("small", "base"):
            rows = aggregate(probs[verifier], claims_cfgs, variant, margin, ent_t, contr_t)
            rows_by_verifier[verifier] = rows
            point["verifiers"][verifier] = evaluate_point(
                rows, f"{vtag}-{ent_t}-{contr_t}-{verifier}")
        kappa, n_claims = claim_kappa(rows_by_verifier["small"], rows_by_verifier["base"],
                                      probs["small"], probs["base"], claims_cfgs,
                                      variant, margin, ent_t, contr_t)
        point["kappa_small_base"] = kappa
        point["n_claims_aligned"] = n_claims
        sweep["points"].append(point)
        gs = point["verifiers"]["small"]["granite_hib_vs_lex"]
        print(f"{vtag:14s} ent={ent_t} contr={contr_t}  "
              f"sig_rag small={len(point['verifiers']['small']['sig_rag_pairs'])}/12 "
              f"base={len(point['verifiers']['base']['sig_rag_pairs'])}/12  "
              f"granite_hib-lex(small) d_z={gs['d_z']:+.2f} p_bh={gs['p_bh']:.3f}  "
              f"kappa={kappa}  ({time.time()-t0:.0f}s)", flush=True)

    (OUT_DIR / "sweep_results.json").write_text(
        json.dumps(sweep, indent=1), encoding="utf-8")

    # ---- robustness map (markdown) -----------------------------------------
    md = ["# Tier 0 — mapa de robustez del instrumento NLI (exp15)", "",
          f"Grid: {len(sweep['points'])} puntos = {len(VARIANTS)} variantes x "
          f"{len(THRESHOLDS)}^2 umbrales; ancla (vb_agree, 0.7, 0.7) == v4 firmado.", ""]
    for verifier in ("small", "base"):
        md += [f"## Significativos RAG-vs-RAG /12 — verificador {verifier}",
               "", "| variante | " + " | ".join(
                   f"ent {e} / contr {c}" for e, c in
                   itertools.product(THRESHOLDS, THRESHOLDS)) + " |",
               "|" + "---|" * (1 + len(THRESHOLDS) ** 2)]
        for (variant, margin) in VARIANTS:
            vtag = f"{variant}{'' if margin == 0 else f'_d{margin}'}"
            cells = []
            for e, c in itertools.product(THRESHOLDS, THRESHOLDS):
                p = next(x for x in sweep["points"]
                         if x["variant"] == vtag and x["ent_t"] == e and x["contr_t"] == c)
                cells.append(str(len(p["verifiers"][verifier]["sig_rag_pairs"])))
            md.append(f"| {vtag} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Kappa small-vs-base (nivel claim, 3 clases)", "",
           "| variante | " + " | ".join(
               f"ent {e} / contr {c}" for e, c in
               itertools.product(THRESHOLDS, THRESHOLDS)) + " |",
           "|" + "---|" * (1 + len(THRESHOLDS) ** 2)]
    for (variant, margin) in VARIANTS:
        vtag = f"{variant}{'' if margin == 0 else f'_d{margin}'}"
        cells = []
        for e, c in itertools.product(THRESHOLDS, THRESHOLDS):
            p = next(x for x in sweep["points"]
                     if x["variant"] == vtag and x["ent_t"] == e and x["contr_t"] == c)
            cells.append(str(p["kappa_small_base"]))
        md.append(f"| {vtag} | " + " | ".join(cells) + " |")
    md.append("")
    (OUT_DIR / "sweep_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote sweep_results.json + sweep_summary.md ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
