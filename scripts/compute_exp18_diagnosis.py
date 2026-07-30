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


def word_ngrams(text, n=5):
    w = re.findall(r"\w+", text.lower())
    return set(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) if len(w) >= n else set()


def tokens(text):
    return set(re.findall(r"\w+", text.lower()))


def jac(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


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
