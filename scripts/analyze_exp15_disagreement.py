"""Tier 3 · Block A — anatomy of the small-vs-base NLI disagreement (kappa 0.32).

Pure CPU over the persisted Tier 0 probabilities (exp12 read-only). Decomposes
WHERE the two verifiers disagree and HOW fragile each decision is, to tell
apart "the low faithfulness is partly a measurement artifact" from "it is real".

Analyses (all at the canonical operating point vb_agree, ent 0.7, contr 0.7,
unless noted), over the 14 469 genuine claims aligned 1:1 across verifiers:
  1. 3x3 small x base claim-label confusion matrix (decomposes kappa).
  2. Disagreement rate by model / scenario / claim-length bucket.
  3. Threshold fragility: % of claims whose supported<->not label flips when
     ent_t moves +-0.05 (best_ent in [0.65,0.75]); same on the contr gate.
  4. False-contradicted candidates: small=contradicted (best_contr>=0.9) AND
     base=supported -> count + ranked CSV (feeds the gold oversample, Block D).
  5. Aggregation sensitivity: max vs mean-top2 vs noisy-or (under a fixed v0
     gate to isolate the reduction from the vb_agree guard) -> label-change rate
     and whether the 0/12 null or the granite hibrido-vs-lexico contrast move.

Outputs (experiments/results/exp15_ablation_nli/):
  disagreement_analysis.json, disagreement_summary.md,
  false_contradicted_candidates.csv

Usage: python scripts/analyze_exp15_disagreement.py
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import gzip
import importlib.util
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/results/exp15_ablation_nli"
CHUNK_MAP = ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"

from src.generation.hallucination_detector import decide_nli_status  # noqa: E402

# reuse the sweep's evaluate_point (v4 methodology per operating point)
_spec = importlib.util.spec_from_file_location(
    "sweep", ROOT / "scripts" / "compute_exp15_nli_sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

ENT_T = CONTR_T = 0.7
LABELS = ["supported", "contradicted", "unsupported"]
GRANITE_PAIR = "hibrido | granite4.1-8b vs lexico | granite4.1-8b"


def load_probs(tag):
    with gzip.open(OUT / f"nli_probs__{tag}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)["configs"]


def genuine_claims(meta):
    return [c for c, a in zip(meta["claims"], meta["artifact"]) if not a]


def decide(chunk_probs, variant="vb_agree", ent_t=ENT_T, contr_t=CONTR_T):
    contr = [p[0] for p in chunk_probs]
    ent = [p[1] for p in chunk_probs]
    st, _, _ = decide_nli_status(contr, ent, ent_t, contr_t, variant=variant)
    return st, max(contr), max(ent)


def main():
    probs = {t: load_probs(t) for t in ("small", "base")}
    claims = json.loads((OUT / "claims_extraction.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    report = {"operating_point": "vb_agree ent0.7 contr0.7", "analyses": {}}
    md = ["# Tier 3 · Bloque A — anatomía del desacuerdo NLI small-vs-base", ""]

    # ---- iterate aligned genuine claims -----------------------------------
    # record per claim: labels + best scores for both verifiers, + slice keys
    rows = []  # dict per (cfg,qid,claim_idx)
    for cfg in probs["small"]:
        scen, model = cfg.split(" | ", 1)
        for qid, per_claim_s in probs["small"][cfg].items():
            per_claim_b = probs["base"][cfg][qid]
            gc = genuine_claims(claims[cfg][qid])
            cids = claims[cfg][qid]["chunk_ids"]
            for i, (cp_s, cp_b) in enumerate(zip(per_claim_s, per_claim_b)):
                ls, cs, es = decide(cp_s)
                lb, cb, eb = decide(cp_b)
                rows.append({
                    "cfg": cfg, "scen": scen, "model": model, "qid": qid,
                    "claim": gc[i] if i < len(gc) else "",
                    "chunk_ids": cids,
                    "s_label": ls, "s_contr": cs, "s_ent": es,
                    "b_label": lb, "b_contr": cb, "b_ent": eb,
                    "s_probs": cp_s,
                })
    n = len(rows)
    report["n_genuine_claims"] = n

    # ---- 1. confusion matrix ----------------------------------------------
    conf = {a: {b: 0 for b in LABELS} for a in LABELS}
    for r in rows:
        conf[r["s_label"]][r["b_label"]] += 1
    agree = sum(conf[l][l] for l in LABELS)
    po = agree / n
    ps = Counter(r["s_label"] for r in rows)
    pb = Counter(r["b_label"] for r in rows)
    pe = sum((ps[l] / n) * (pb[l] / n) for l in LABELS)
    kappa = (po - pe) / (1 - pe)
    report["analyses"]["confusion"] = {"matrix_small_rows_base_cols": conf,
                                       "n": n, "agreement": round(po, 4),
                                       "kappa": round(kappa, 4)}
    md += ["## 1. Matriz de confusión small(filas) × base(cols)",
           "", f"n={n}, acuerdo={po:.3f}, κ={kappa:.4f}", "",
           "| small\\base | " + " | ".join(LABELS) + " |",
           "|" + "---|" * (len(LABELS) + 1)]
    for a in LABELS:
        md.append(f"| {a} | " + " | ".join(str(conf[a][b]) for b in LABELS) + " |")
    md.append("")

    # ---- 2. disagreement rate by slice ------------------------------------
    def disagree_rate(pred):
        sub = [r for r in rows if pred(r)]
        if not sub:
            return None, 0
        d = sum(1 for r in sub if r["s_label"] != r["b_label"])
        return round(d / len(sub), 4), len(sub)

    pooled = sum(1 for r in rows if r["s_label"] != r["b_label"]) / n
    slices = {"pooled": (round(pooled, 4), n)}
    for m in sorted({r["model"] for r in rows}):
        slices[f"model:{m}"] = disagree_rate(lambda r, m=m: r["model"] == m)
    for s in sorted({r["scen"] for r in rows}):
        slices[f"scen:{s}"] = disagree_rate(lambda r, s=s: r["scen"] == s)
    def lenbucket(claim):
        w = len(claim.split())
        return "<10" if w < 10 else "10-20" if w < 20 else "20-35" if w < 35 else ">=35"
    for lb in ("<10", "10-20", "20-35", ">=35"):
        slices[f"len:{lb}"] = disagree_rate(lambda r, lb=lb: lenbucket(r["claim"]) == lb)
    report["analyses"]["disagreement_by_slice"] = slices
    md += ["## 2. Tasa de desacuerdo por slice (pooled = "
           f"{pooled:.3f})", "", "| slice | tasa | n |", "|---|---|---|"]
    for k, (rate, cnt) in slices.items():
        flag = " ⚠️" if rate and rate > 2 * pooled else ""
        md.append(f"| {k} | {rate} | {cnt}{flag} |")
    md.append("")

    # ---- 3. threshold fragility -------------------------------------------
    PFX = {"small": "s", "base": "b"}

    def supported_at(r, p, ent_t):
        # v0-style supported test on the reduced scalars (guard-independent)
        return (r[f"{p}_ent"] > ent_t and r[f"{p}_ent"] > r[f"{p}_contr"])
    frag = {}
    for tag in ("small", "base"):
        p = PFX[tag]
        near = [r for r in rows if 0.65 <= r[f"{p}_ent"] <= 0.75]
        flip = sum(1 for r in near
                   if supported_at(r, p, 0.65) != supported_at(r, p, 0.75))
        near_c = [r for r in rows if 0.65 <= r[f"{p}_contr"] <= 0.75]
        frag[tag] = {"near_ent_gate": len(near),
                     "ent_flip_pm05": flip,
                     "ent_fragile_pct_of_all": round(100 * flip / n, 2),
                     "near_contr_gate": len(near_c),
                     "near_contr_pct_of_all": round(100 * len(near_c) / n, 2)}
    report["analyses"]["threshold_fragility"] = frag
    md += ["## 3. Fragilidad de umbral (best_ent∈[0.65,0.75] → flip supported al mover ent_t ±0.05)",
           "", "| verif | near_ent | flips | % del total | near_contr |",
           "|---|---|---|---|---|"]
    for tag in ("small", "base"):
        f = frag[tag]
        md.append(f"| {tag} | {f['near_ent_gate']} | {f['ent_flip_pm05']} | "
                  f"{f['ent_fragile_pct_of_all']}% | {f['near_contr_gate']} |")
    md.append("")

    # ---- 4. false-contradicted candidates ---------------------------------
    fc = [r for r in rows if r["s_label"] == "contradicted" and r["s_contr"] >= 0.9
          and r["b_label"] == "supported"]
    fc.sort(key=lambda r: r["s_contr"], reverse=True)
    report["analyses"]["false_contradicted"] = {
        "count": len(fc),
        "by_model": dict(Counter(r["model"] for r in fc)),
        "by_scenario": dict(Counter(r["scen"] for r in fc))}
    csv = ["idx;config;query_id;s_contr;b_ent;claim;best_chunk_id;best_chunk_source;best_chunk_text"]
    for i, r in enumerate(fc, 1):
        # best chunk = argmax small contradiction
        bci = max(range(len(r["s_probs"])), key=lambda k: r["s_probs"][k][0])
        cid = r["chunk_ids"][bci] if bci < len(r["chunk_ids"]) else ""
        ch = chunk_map.get(cid, {})
        src = f"{ch.get('cloud_provider','')}/{ch.get('service_name','')} :: {ch.get('heading_path','')}"
        def esc(v):
            v = str(v).replace("\r", " ").replace("\n", " ⏎ ")
            return '"' + v.replace('"', '""') + '"' if ('"' in v or ";" in v) else v
        csv.append(";".join(esc(x) for x in
                   [i, r["cfg"], r["qid"], round(r["s_contr"], 4), round(r["b_ent"], 4),
                    r["claim"], cid, src, ch.get("text", "")[:600]]))
    (OUT / "false_contradicted_candidates.csv").write_text(
        "\n".join(csv), encoding="utf-8-sig")
    md += ["## 4. Falso-contradicted (small=contradicted conf≥0.9 ∧ base=supported)",
           "", f"**{len(fc)} claims** → `false_contradicted_candidates.csv`",
           f"por modelo: {report['analyses']['false_contradicted']['by_model']}",
           f"por escenario: {report['analyses']['false_contradicted']['by_scenario']}", ""]

    # ---- 5. aggregation sensitivity ---------------------------------------
    def reduce_scores(chunk_probs, mode):
        contr = sorted((p[0] for p in chunk_probs), reverse=True)
        ent = sorted((p[1] for p in chunk_probs), reverse=True)
        if mode == "max":
            return contr[0], ent[0]
        if mode == "mean_top2":
            return float(np.mean(contr[:2])), float(np.mean(ent[:2]))
        if mode == "noisy_or":
            return (1 - np.prod([1 - p[0] for p in chunk_probs]),
                    1 - np.prod([1 - p[1] for p in chunk_probs]))
        raise ValueError(mode)

    def v0_label(c, e):
        if e > ENT_T and e > c:
            return "supported"
        if c > CONTR_T:
            return "contradicted"
        return "unsupported"

    def rows_for_mode(tag, mode):
        """v3-format rows for evaluate_point under an aggregator (v0 gate)."""
        out = {}
        for cfg in probs[tag]:
            per_cfg = {}
            for qid, per_claim in probs[tag][cfg].items():
                meta = claims[cfg][qid]
                total = len(meta["claims"])
                n_art = sum(1 for a in meta["artifact"] if a)
                g = total - n_art
                if not per_claim:
                    per_cfg[qid] = {"total_claims": total, "not_a_claim": n_art, "genuine": 0,
                                    "supported": 0, "contradicted": 0, "unsupported": 0,
                                    "faithfulness": 1.0}
                    continue
                agg = {"supported": 0, "contradicted": 0, "unsupported": 0}
                for cp in per_claim:
                    c, e = reduce_scores(cp, mode)
                    agg[v0_label(c, e)] += 1
                per_cfg[qid] = {"total_claims": total, "not_a_claim": n_art, "genuine": g,
                                **agg, "faithfulness": round(agg["supported"] / g, 4)}
            out[cfg] = per_cfg
        return out

    agg_report = {}
    # label-change rate vs v0-max, small verifier
    base_labels = [v0_label(*reduce_scores(r["s_probs"], "max")) for r in rows]
    for mode in ("mean_top2", "noisy_or"):
        alt = [v0_label(*reduce_scores(r["s_probs"], mode)) for r in rows]
        changed = sum(1 for a, b in zip(base_labels, alt) if a != b)
        agg_report[f"small_labelchange_{mode}_vs_max"] = round(changed / n, 4)
    # downstream: 0/12 + granite pair per aggregator (small)
    agg_down = {}
    for mode in ("max", "mean_top2", "noisy_or"):
        ev = sweep.evaluate_point(rows_for_mode("small", mode), f"aggA-{mode}")
        gp = ev.get("granite_hib_vs_lex", {})
        agg_down[mode] = {"sig_rag_pairs": len(ev["sig_rag_pairs"]),
                          "granite_d_z": gp.get("d_z"), "granite_p_bh": gp.get("p_bh")}
    agg_report["downstream_small"] = agg_down
    report["analyses"]["aggregation_sensitivity"] = agg_report
    md += ["## 5. Sensibilidad de agregación (gate v0 fijo, verificador small)",
           "", "Cambio de etiqueta vs max: "
           f"mean_top2={agg_report['small_labelchange_mean_top2_vs_max']}, "
           f"noisy_or={agg_report['small_labelchange_noisy_or_vs_max']}", "",
           "| agregador | sig RAG /12 | granite d_z | granite p_bh |",
           "|---|---|---|---|"]
    for mode in ("max", "mean_top2", "noisy_or"):
        d = agg_down[mode]
        md.append(f"| {mode} | {d['sig_rag_pairs']} | {d['granite_d_z']} | {d['granite_p_bh']} |")
    md.append("")

    # ---- gates ------------------------------------------------------------
    maxchange = max(agg_report["small_labelchange_mean_top2_vs_max"],
                    agg_report["small_labelchange_noisy_or_vs_max"])
    moved = any(agg_down[m]["sig_rag_pairs"] != agg_down["max"]["sig_rag_pairs"]
                for m in ("mean_top2", "noisy_or"))
    gate = ("A-G1: agregación mueve >15% etiquetas o el 0/12 → decisión Enzo del agregador"
            if (maxchange > 0.15 or moved) else
            "A-G1: agregación NO es fuente de ruido (cambio <15%, 0/12 estable) → cerrar hilo")
    report["gates"] = {"A_G1": gate,
                       "A_G2_contradicted_oversample_size": len(fc)}
    md += ["## Gates", "", f"- {gate}",
           f"- A-G2: {len(fc)} falso-contradicted → tamaño del estrato contradicted en el gold (Bloque D)", ""]

    (OUT / "disagreement_analysis.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "disagreement_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote disagreement_analysis.json + disagreement_summary.md + "
          f"false_contradicted_candidates.csv ({len(fc)} rows)")


if __name__ == "__main__":
    main()
