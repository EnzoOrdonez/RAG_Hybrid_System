"""exp18 — the two readings faithfulness alone cannot give.

`compute_tierA_arm_stats.py` answers "did faithfulness move?". exp18 needs two more
things, because its arms are diagnostic rather than competitive:

1. ANSWER DIVERGENCE (the `evidence_swapped` reading). If the answer barely changes when
   the context is replaced by another query's evidence, the generator is not reading the
   context at all -- and then no retrieval improvement could ever have moved faithfulness,
   which would explain the whole 0/12 null in one stroke. Faithfulness cannot show this:
   a swapped arm scores low either way (the claims no longer match the given chunks),
   whether the model ignored the evidence or followed it. Only comparing the ANSWERS to
   the baseline's separates the two.

   Reported per arm vs baseline_repro, paired by query:
     - jaccard_5gram   answer word-5-gram overlap with the baseline answer
     - token_jaccard   bag-of-words overlap (looser, catches paraphrase)
     - identical_rate  byte-identical answers
     - claim_overlap   fraction of the baseline's genuine claims that reappear
   High divergence => the generator tracks the evidence. Near-zero divergence => it does
   not, and the ceiling is the generator, not the retrieval.

2. THE TRUNCATION SPLIT (the `final_top_k_10` reading). At k=10 roughly 40% of the subset
   exceeds the 4096-token window, so this arm confounds "more evidence" with "evidence
   cut off" BY CONSTRUCTION. Averaging over that confound produces an uninterpretable
   number. Faithfulness is therefore reported split by OBSERVED truncation
   (tokens.input >= 4096, recorded by the runner), so the untruncated subset gives the
   only clean local read on whether more evidence helps.

Usage: python scripts/compute_exp18_diagnosis.py [--verifier hhem]
Writes: experiments/results/exp18_evidence_ceiling/diagnosis__{verifier}.{json,md}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
BASELINE_ARM = "baseline_repro"
CTX_LIMIT = 4096

# ---------------------------------------------------------------- pre-registered TOST
# The decision matrix uses "oracle_evidence is no better than baseline" to argue that the
# selection lever is exhausted and the ceiling is generator capacity -- i.e. it ACCEPTS a
# null to justify spending on cloud. A non-significant Wilcoxon cannot carry that: absence
# of evidence is not evidence of absence, and at n=57 this design could not even detect the
# largest effect the phase ever measured. So the null has to be a POSITIVE claim of
# equivalence, declared before the data exist.
#
# Band = +/- 0.081, the exp17 HHEM effect of provider balancing. Read as: "selecting evidence
# with an independent oracle does not buy even what balancing coverage bought." It is a
# PRE-EXISTING quantity from another experiment, blind to this contrast -- not a threshold
# tuned until something passes. Fixed here, in code, before the run.
TOST_BAND = 0.081
TOST_ARM = "oracle_evidence"
TOST_ALPHA = 0.05


def word_ngrams(text, n=5):
    w = re.findall(r"\w+", text.lower())
    return set(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) if len(w) >= n else set()


def tokens(text):
    return set(re.findall(r"\w+", text.lower()))


def jac(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def tost(diffs, band=TOST_BAND, alpha=TOST_ALPHA):
    """Two one-sided tests for equivalence of a paired difference within +/- band.

    Equivalence is declared when BOTH one-sided t-tests reject, which is the same as the
    (1-2*alpha) CI falling entirely inside the band. Reporting the CI alongside p keeps the
    claim readable: it says which effect sizes the data exclude, not merely that nothing
    reached significance.
    """
    from scipy import stats
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return None
    mean, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(n))
    if se == 0:
        return {"n": n, "mean_diff": round(mean, 4), "equivalent": bool(abs(mean) < band),
                "note": "zero variance"}
    df = n - 1
    t_lo = (mean + band) / se          # H0: diff <= -band
    t_hi = (mean - band) / se          # H0: diff >= +band
    p_lo = float(stats.t.sf(t_lo, df))
    p_hi = float(stats.t.cdf(t_hi, df))
    p = max(p_lo, p_hi)
    crit = float(stats.t.ppf(1 - alpha, df))
    lo, hi = mean - crit * se, mean + crit * se   # (1-2a) CI
    return {
        "n": n, "band": band, "alpha": alpha,
        "mean_diff": round(mean, 4), "se": round(se, 4),
        "ci90": [round(lo, 4), round(hi, 4)],
        "p_tost": round(p, 5),
        "equivalent": bool(p < alpha),
        # This arm alone CANNOT conclude "selection is exhausted". The oracle ranks by
        # TOPICAL RELEVANCE to the query, while the metric asks whether the claims the model
        # chose to assert are supported -- so equivalence here rules out headroom reachable
        # by relevance ranking, and nothing more. Whether any selection could help is
        # answered by selection_bound.json, and the verdict is the JOINT reading of the two
        # (ledger entry 20). An earlier version of this string asserted the stronger claim;
        # it would have contradicted the bound artifact sitting next to it.
        "reading": ("EQUIVALENTE dentro de ±%.3f: seleccionar con un oraculo de RELEVANCIA "
                    "TOPICA no compra ni lo que compro balancear la cobertura. Esto descarta "
                    "el margen alcanzable por ranking topico, NO el margen de seleccion en "
                    "general -> leer junto a selection_bound.json (matriz, ledger entrada 20)."
                    % band) if p < alpha else
                   ("NO concluyente: el IC90 no cabe entero en ±%.3f, asi que estos datos NO "
                    "permiten afirmar equivalencia (ni significancia). Un nulo aqui NO "
                    "justifica el gasto en nube." % band),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verifier", default="hhem", choices=["small", "base", "hhem"])
    ap.add_argument("--exp-dir", default=str(EXP_DIR))
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)

    res = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))
    cfgs = res["configs"]
    by_arm = {c.get("scenario", k.split(" | ")[0]): {r["query_id"]: r for r in c["results"]}
              for k, c in cfgs.items()}
    if BASELINE_ARM not in by_arm:
        sys.exit(f"no {BASELINE_ARM} arm in {exp_dir}/results.json")
    base = by_arm[BASELINE_ARM]

    rows_path = (exp_dir / "faithfulness_rows__hhem.json" if args.verifier == "hhem"
                 else exp_dir / f"faithfulness_rows__{args.verifier}__vb_agree.json")
    rows = (json.loads(rows_path.read_text(encoding="utf-8"))["configs"]
            if rows_path.exists() else {})
    claims_path = exp_dir / "claims_extraction.json"
    claims = (json.loads(claims_path.read_text(encoding="utf-8"))["configs"]
              if claims_path.exists() else {})

    def arm_cfg_name(arm):
        for k, c in cfgs.items():
            if c.get("scenario", k.split(" | ")[0]) == arm:
                return k
        return None

    def genuine_claims(arm, qid):
        c = claims.get(arm_cfg_name(arm) or "", {}).get(qid)
        if not c:
            return set()
        return {cl.strip().lower() for cl, a in zip(c["claims"], c["artifact"]) if not a}

    # ---------------------------------------------------- 1. answer divergence
    divergence = []
    for arm, per_q in by_arm.items():
        if arm == BASELINE_ARM:
            continue
        j5, jt, ident, cov, n = [], [], 0, [], 0
        for qid, r in per_q.items():
            b = base.get(qid)
            if not b:
                continue
            n += 1
            a_txt, b_txt = (r.get("answer") or ""), (b.get("answer") or "")
            j5.append(jac(word_ngrams(a_txt), word_ngrams(b_txt)))
            jt.append(jac(tokens(a_txt), tokens(b_txt)))
            ident += int(a_txt == b_txt)
            bc, ac = genuine_claims(BASELINE_ARM, qid), genuine_claims(arm, qid)
            if bc:
                cov.append(len(bc & ac) / len(bc))
        divergence.append({
            "arm": arm, "n_paired": n,
            "jaccard_5gram_vs_baseline": round(float(np.mean(j5)), 4) if j5 else None,
            "token_jaccard_vs_baseline": round(float(np.mean(jt)), 4) if jt else None,
            "identical_answer_rate": round(ident / n, 4) if n else None,
            "baseline_claim_reappearance": round(float(np.mean(cov)), 4) if cov else None,
        })

    # ------------------------------------- 1b. pre-registered equivalence (oracle arm)
    equivalence = None
    if TOST_ARM in by_arm and rows:
        b_name = arm_cfg_name(BASELINE_ARM)
        a_name = arm_cfg_name(TOST_ARM)
        diffs = []
        for qid in by_arm[TOST_ARM]:
            fb = (rows.get(b_name, {}).get(qid) or {}).get("faithfulness")
            fa = (rows.get(a_name, {}).get(qid) or {}).get("faithfulness")
            if fb is not None and fa is not None:
                diffs.append(float(fa) - float(fb))
        equivalence = tost(diffs)
        if equivalence:
            equivalence["arm"] = TOST_ARM
            equivalence["band_source"] = ("exp17 HHEM effect of provider balancing (+0.081); "
                                          "pre-existing and blind to this contrast")

    # ------------------------------------------------- 2. truncation split
    split = None
    if "final_top_k_10" in by_arm and rows:
        trunc_qids = {qid for qid, r in by_arm["final_top_k_10"].items()
                      if (r.get("tokens") or {}).get("input", 0) >= CTX_LIMIT}
        b_name, a_name = arm_cfg_name(BASELINE_ARM), arm_cfg_name("final_top_k_10")
        groups = {"untruncated": [], "truncated": []}
        for qid in by_arm["final_top_k_10"]:
            fb = (rows.get(b_name, {}).get(qid) or {}).get("faithfulness")
            fa = (rows.get(a_name, {}).get(qid) or {}).get("faithfulness")
            if fb is None or fa is None:
                continue
            groups["truncated" if qid in trunc_qids else "untruncated"].append(
                (float(fb), float(fa)))
        split = {"context_limit": CTX_LIMIT, "n_truncated": len(trunc_qids), "groups": {}}
        for g, pairs in groups.items():
            if not pairs:
                split["groups"][g] = {"n": 0}
                continue
            b = np.array([p[0] for p in pairs])
            a = np.array([p[1] for p in pairs])
            split["groups"][g] = {
                "n": len(pairs),
                "baseline_mean": round(float(b.mean()), 4),
                "top10_mean": round(float(a.mean()), 4),
                "mean_diff": round(float((a - b).mean()), 4),
            }
        split["note"] = ("Only `untruncated` is a clean read on whether more evidence helps. "
                         "The truncated group mixes more evidence with cut-off evidence and "
                         "must not be averaged into a single number.")

    payload = {
        "experiment_id": exp_dir.name, "verifier": args.verifier,
        "baseline_arm": BASELINE_ARM,
        "divergence_note": (
            "Answer divergence vs baseline. LOW divergence on `evidence_swapped` means the "
            "generator does not track the evidence it is given -- which would explain the "
            "retrieval null directly. Faithfulness alone cannot distinguish that from the "
            "model correctly following wrong evidence, since both score low."),
        "answer_divergence": divergence,
        "preregistered_equivalence": equivalence,
        "truncation_split_final_top_k_10": split,
        "generated_by": "scripts/compute_exp18_diagnosis.py",
    }
    (exp_dir / f"diagnosis__{args.verifier}.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")

    L = [f"# exp18 — diagnóstico ({args.verifier})", "",
         "## Divergencia de respuesta vs baseline_repro", "",
         "| Brazo | n | jaccard 5-gram | jaccard tokens | idénticas | claims del baseline que reaparecen |",
         "|---|---|---|---|---|---|"]
    for d in divergence:
        L.append(f"| {d['arm']} | {d['n_paired']} | {d['jaccard_5gram_vs_baseline']} | "
                 f"{d['token_jaccard_vs_baseline']} | {d['identical_answer_rate']} | "
                 f"{d['baseline_claim_reappearance']} |")
    L += ["", "**Lectura de `evidence_swapped`:** solape ALTO con el baseline ⇒ el generador no "
              "sigue la evidencia (el techo es del generador, y el nulo de recuperación queda "
              "explicado). Solape BAJO ⇒ sí la sigue, y el techo hay que buscarlo en su capacidad "
              "de anclar evidencia buena.", ""]
    if equivalence:
        L += [f"## Equivalencia pre-registrada — `{TOST_ARM}` (TOST, banda ±{TOST_BAND})", "",
              f"n={equivalence['n']} · Δ={equivalence['mean_diff']} · "
              f"IC90=[{equivalence['ci90'][0]}, {equivalence['ci90'][1]}] · "
              f"p_TOST={equivalence['p_tost']} · "
              f"**{'EQUIVALENTE' if equivalence['equivalent'] else 'NO concluyente'}**", "",
              equivalence["reading"], "",
              "Banda anclada en el efecto HHEM de exp17 (+0,081): preexistente y ciega a este "
              "contraste, no un umbral ajustado hasta que algo pase.", ""]
    if split:
        L += ["## `final_top_k_10` partido por truncamiento observado", "",
              f"{split['n_truncated']} queries alcanzan el límite de {CTX_LIMIT} tokens.", "",
              "| Grupo | n | baseline | top-10 | Δ |", "|---|---|---|---|---|"]
        for g, s in split["groups"].items():
            if s.get("n"):
                L.append(f"| {g} | {s['n']} | {s['baseline_mean']} | {s['top10_mean']} | "
                         f"{s['mean_diff']} |")
        L += ["", split["note"]]
    (exp_dir / f"diagnosis__{args.verifier}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
