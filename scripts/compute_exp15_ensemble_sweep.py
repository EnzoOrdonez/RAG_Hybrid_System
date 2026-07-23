"""Tier 3 · Block B — ensemble & guard sweep with a negative-control criterion.

Consumes the persisted real probs of all four verifiers (small/base/large NLI +
HHEM grounding) and the negative-control scores, and evaluates every candidate
instrument on TWO axes:

  (1) SELECTION criterion (anti-p-hacking, pre-registered): the negative-control
      false-contradicted rate (NLI) / false-grounded rate (HHEM) on 400 random
      (claim, 5-random-chunk) tuples. Lower = better construct validity. This is
      blind to scenario labels and to the downstream contrast.
  (2) DESCRIPTIVE ONLY: the downstream between-scenario family on the real probs
      (sig_rag /12, granite hibrido-vs-lexico) — reported side by side but NEVER
      used to choose the instrument (rigor rule: no tuning toward hybrid-favoring
      outcomes).

Candidates: small/base/large (vb_agree@0.7), HHEM (tau 0.5), and ensembles over
the NLI trio — E1 prob-mean, E2 majority-vote (disagree->unsupported), E4
symmetric guard on base (supported also needs >=2 chunks or a margin), E5
cross-family base AND HHEM. Aggregator max vs noisy_or reported for the NLI ones
(Block A gate A-G1).

Outputs (experiments/results/exp15_ablation_nli/):
  ensemble_sweep_results.json, ensemble_summary.md

Usage: python scripts/compute_exp15_ensemble_sweep.py
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/results/exp15_ablation_nli"

from src.generation.hallucination_detector import decide_nli_status  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "sweep", ROOT / "scripts" / "compute_exp15_nli_sweep.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

ENT_T = CONTR_T = 0.7
HHEM_TAU = 0.5
NLI_TRIO = ["small", "base", "large"]


def load_nli(tag):
    with gzip.open(OUT / f"nli_probs__{tag}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)["configs"]


def load_hhem():
    with gzip.open(OUT / "grounding_probs__hhem.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)["configs"]


# ---- per-claim decision functions (return "supported"/"contradicted"/"unsupported")
def decide_single_nli(chunk_probs, variant="vb_agree", agg="max"):
    """chunk_probs: list of [contr,ent,neut]. agg controls the scalar reduction
    fed to the v0-style gate; vb_agree keeps its native multi-chunk guard."""
    if agg == "max":
        contr = [p[0] for p in chunk_probs]
        ent = [p[1] for p in chunk_probs]
        st, _, _ = decide_nli_status(contr, ent, ENT_T, CONTR_T, variant=variant)
        return st
    # noisy_or / mean_top2 reduce to scalars then v0 gate (guard-independent)
    c_sorted = sorted((p[0] for p in chunk_probs), reverse=True)
    e_sorted = sorted((p[1] for p in chunk_probs), reverse=True)
    if agg == "noisy_or":
        c = 1 - np.prod([1 - p[0] for p in chunk_probs])
        e = 1 - np.prod([1 - p[1] for p in chunk_probs])
    elif agg == "mean_top2":
        c, e = float(np.mean(c_sorted[:2])), float(np.mean(e_sorted[:2]))
    else:
        raise ValueError(agg)
    if e > ENT_T and e > c:
        return "supported"
    if c > CONTR_T:
        return "contradicted"
    return "unsupported"


def decide_ensemble(per_chunk_by_member, kind):
    """per_chunk_by_member: {member: [[c,e,n],...]} aligned by chunk for NLI trio."""
    if kind == "E1_mean":
        arrs = [np.array(per_chunk_by_member[m]) for m in NLI_TRIO]
        mean = np.mean(arrs, axis=0)
        return decide_single_nli(mean.tolist(), variant="vb_agree")
    if kind == "E2_vote":
        labels = [decide_single_nli(per_chunk_by_member[m]) for m in NLI_TRIO]
        from collections import Counter
        c = Counter(labels)
        top, n = c.most_common(1)[0]
        return top if n >= 2 else "unsupported"
    if kind == "E3_conservative":
        # contr=max, ent=min across members, per chunk
        arrs = [np.array(per_chunk_by_member[m]) for m in NLI_TRIO]
        contr = np.max([a[:, 0] for a in arrs], axis=0)
        ent = np.min([a[:, 1] for a in arrs], axis=0)
        merged = [[contr[i], ent[i], 0.0] for i in range(len(contr))]
        return decide_single_nli(merged, variant="vb_agree")
    raise ValueError(kind)


def decide_hhem(scores):
    return "supported" if (scores and max(scores) > HHEM_TAU) else "unsupported"


# ---- build v3-format rows for a candidate over the REAL data -----------------
def rows_for_candidate(cand, nli, hhem, claims):
    out = {}
    cfgs = nli["small"].keys()
    for cfg in cfgs:
        per = {}
        for qid in nli["small"][cfg]:
            meta = claims[cfg][qid]
            total = len(meta["claims"])
            n_art = sum(1 for a in meta["artifact"] if a)
            g = total - n_art
            per_claim_members = {m: nli[m][cfg][qid] for m in NLI_TRIO}
            hh = hhem[cfg][qid]
            if g == 0:
                per[qid] = {"total_claims": total, "not_a_claim": n_art, "genuine": 0,
                            "supported": 0, "contradicted": 0, "unsupported": 0,
                            "faithfulness": 1.0}
                continue
            agg = {"supported": 0, "contradicted": 0, "unsupported": 0}
            for ci in range(g):
                lbl = label_one(cand, {m: per_claim_members[m][ci] for m in NLI_TRIO},
                                hh[ci] if ci < len(hh) else [])
                agg[lbl] += 1
            per[qid] = {"total_claims": total, "not_a_claim": n_art, "genuine": g,
                        **agg, "faithfulness": round(agg["supported"] / g, 4)}
        out[cfg] = per
    return out


def label_one(cand, members_chunkprobs, hhem_chunkscores):
    """One claim's label under a candidate. members_chunkprobs: {member:[[c,e,n]..]}."""
    if cand in ("small", "base", "large"):
        return decide_single_nli(members_chunkprobs[cand])
    if cand.startswith("agg:"):
        _, tag, agg = cand.split(":")
        return decide_single_nli(members_chunkprobs[tag], agg=agg)
    if cand == "hhem":
        return decide_hhem(hhem_chunkscores)
    if cand in ("E1_mean", "E2_vote", "E3_conservative"):
        return decide_ensemble(members_chunkprobs, cand)
    if cand == "E4_sym_base":
        # supported requires the base entailment gate AND >=2 chunks over ent_t
        cp = members_chunkprobs["base"]
        ent = [p[1] for p in cp]
        n_over = sum(1 for e in ent if e > ENT_T)
        base_lbl = decide_single_nli(cp)
        if base_lbl == "supported" and n_over < 2:
            return "unsupported"
        return base_lbl
    if cand == "E5_base_and_hhem":
        b = decide_single_nli(members_chunkprobs["base"])
        h = decide_hhem(hhem_chunkscores)
        if b == "supported" and h == "supported":
            return "supported"
        if b == "contradicted":
            return "contradicted"
        return "unsupported"
    raise ValueError(cand)


# ---- negative control --------------------------------------------------------
def negative_control(cand, nc):
    """Rate at which random pairs are labeled contradicted (NLI) / supported (HHEM)."""
    n = nc["n"]
    bad = 0
    for i in range(n):
        members = {m: nc["verifiers"][m]["scores"][i] for m in NLI_TRIO
                   if m in nc["verifiers"]}
        hh = nc["verifiers"]["hhem"]["scores"][i] if "hhem" in nc["verifiers"] else []
        lbl = label_one(cand, members, hh)
        # for grounding-only candidates, "false-grounded" = supported on random
        if cand in ("hhem", "E5_base_and_hhem"):
            if lbl == "supported":
                bad += 1
        else:
            if lbl == "contradicted":
                bad += 1
    return round(bad / n, 4)


def main():
    nli = {t: load_nli(t) for t in NLI_TRIO}
    hhem = load_hhem()
    claims = json.loads((OUT / "claims_extraction.json").read_text(encoding="utf-8"))["configs"]
    nc = json.loads((OUT / "negative_control_scores.json").read_text(encoding="utf-8"))

    candidates = ["small", "base", "large", "hhem",
                  "agg:small:noisy_or", "agg:base:noisy_or",
                  "E1_mean", "E2_vote", "E3_conservative", "E4_sym_base", "E5_base_and_hhem"]

    report = {"selection_criterion": "negative-control false-contradicted / false-grounded rate "
              "(lower=better); downstream is DESCRIPTIVE ONLY",
              "hhem_tau": HHEM_TAU, "ent_t": ENT_T, "candidates": {}}
    for cand in candidates:
        nc_rate = negative_control(cand, nc)
        rows = rows_for_candidate(cand, nli, hhem, claims)
        ev = sweep.evaluate_point(rows, f"ens-{cand}")
        gp = ev.get("granite_hib_vs_lex", {})
        report["candidates"][cand] = {
            "neg_control_bad_rate": nc_rate,
            "downstream_sig_rag_12": len(ev["sig_rag_pairs"]),
            "downstream_granite_d_z": gp.get("d_z"),
            "downstream_granite_p_bh": gp.get("p_bh"),
        }
        print(f"{cand:22s} neg_ctrl={nc_rate:.3f}  sig_rag={len(ev['sig_rag_pairs'])}/12  "
              f"granite p_bh={gp.get('p_bh')}", flush=True)

    ranked = sorted(report["candidates"].items(), key=lambda kv: kv[1]["neg_control_bad_rate"])
    report["provisional_front_runner"] = ranked[0][0]
    report["ranking_by_selection_criterion"] = [k for k, _ in ranked]

    md = ["# Tier 3 · Bloque B — ensembles + control negativo", "",
          "**Criterio de selección (pre-registrado, anti-p-hacking):** tasa de "
          "falso-contradicted (NLI) / falso-grounded (HHEM) en el control negativo "
          "(400 pares aleatorios). Menor = mejor. El downstream es DESCRIPTIVO, NO criterio.", "",
          "| candidato | control neg (↓) | sig RAG /12 | granite p_bh |",
          "|---|---|---|---|"]
    for k, v in ranked:
        md.append(f"| {k} | {v['neg_control_bad_rate']} | {v['downstream_sig_rag_12']} | "
                  f"{v['downstream_granite_p_bh']} |")
    md += ["", f"**Front-runner provisional (por control negativo):** `{report['provisional_front_runner']}`",
           "", "Nota: la selección definitiva del instrumento espera el gold humano (Bloque D). "
           "El downstream NO se usa para elegir.", ""]

    (OUT / "ensemble_sweep_results.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "ensemble_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
