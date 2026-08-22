"""exp18 — why are 759 claims supported by NO chunk in the k=50 pool?

The tension this script attacks: exp18's `evidence_swapped` arm proves the generator DOES use
the context (answers collapse when the evidence is swapped), yet 759 of the claims the baseline
asserted are not supported by any chunk in the retrieved pool. Four explanations, and they have
very different consequences for the thesis:

  (a) LEGITIMATE SYNTHESIS  — the claim follows from several chunks jointly, none alone.
  (b) PARAMETRIC KNOWLEDGE  — the model answers from training, not from the evidence.
  (c) HALLUCINATION         — no source, in the pool or plausibly in the model.
  (d) VERIFIER FAILURE      — the claim IS supported and HHEM missed it.

PRE-REGISTERED BEFORE LOOKING AT ANY OUTPUT (ledger entry 22). Two parts, and only the first
is decidable from data:

--- PART 1: the decidable test -------------------------------------------------------------
If (b) drives a meaningful share, grounding should DEGRADE across the answer of a model that has
just said the context does not cover the question -- the "...However, I can outline general
steps..." tail, written from memory rather than from the evidence.

  H1 (v1, WITHDRAWN — degenerate by construction, never computed): "among decline-prefixed
  answers, claims after the decline marker are less grounded than claims before it". Not
  computable: `pure_decline` is DEFINED as a marker inside the first OPENING_WINDOW=300 chars,
  so essentially no claim precedes it and the "before" stratum is empty. The script's own guard
  ("a lopsided split must be visible") caught this and reported nothing rather than a spurious
  number. The marker-offset distribution is emitted below as the evidence for the withdrawal.

  H1 (v2, DECLARED before any position data was inspected): difference-in-differences on the
  WITHIN-ANSWER position gradient.

      gradient(q)  = unsupportable_rate(second half of claims)
                   - unsupportable_rate(first half of claims)
      H1: mean gradient is MORE POSITIVE for decline-prefixed answers than for `answered` ones.

  Why this shape. The within-query split controls for query difficulty (a hard query makes all
  its claims hard). The `answered` arm controls for a generic "models drift as they write"
  effect, which would raise the gradient everywhere and is not evidence of anything about
  declining. Only the DIFFERENCE between the two is the parametric-fallback signature.

  Estimator: cluster bootstrap over queries, seed 42, 95 % CI on the DiD. Queries with < 4
  genuine claims are excluded (a 3-claim answer has no meaningful halves) and the exclusion
  count is reported. DESCRIPTIVE -- enters no BH family, because it is a mechanism probe, not
  an arm contrast.

--- PART 2: the stratification for the human gold ------------------------------------------
(a) vs (c) CANNOT be separated by a grounding model -- that is the whole reason the human gold
exists. So this script does NOT emit verdicts; it emits CANDIDATE strata plus the stratified
sample to annotate. Rule, declared in advance:

  d_threshold_artifact : best_over_pool  > tau - 0.1      (the bound is threshold-sensitive here)
  a_synthesis_cand     : else, >= 2 chunks over SOFT_TAU  (distributed partial support)
  b_parametric_cand    : else, query opened with a decline marker
  c_unattributed_cand  : everything else (low everywhere, and the model did not flag the gap)

`*_cand` is not a claim about truth. Naming them as verdicts would be exactly the overclaim the
phase keeps catching.

Usage: python scripts/compute_exp18_unsupported_taxonomy.py [--suffix _v2] [--tau 0.5]
       [--out-suffix _v2]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
Writes experiments/results/exp18_evidence_ceiling/unsupported_taxonomy<out-suffix>.{json,md}
       + output/audit/unsupported_claims_sample<out-suffix>.csv
       (stratified, for the human gold)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compute_faithfulness_metrics import classify_response  # noqa: E402
from src.utils.signed_evidence import guard_write  # noqa: E402

EXP_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
AUDIT_DIR = PROJECT_ROOT / "output/audit"
BASELINE_KEY = "baseline_repro | granite4.1-8b"
SOFT_TAU = 0.3          # "partially supports" — deliberately well below tau
NEAR_TAU_BAND = 0.1     # same band the bound already reports
SEED = 42
SAMPLE_PER_STRATUM = 10


def decline_marker_offset(answer, cfm):
    """Char offset of the first refusal marker, or None. Used only for the position split."""
    low = answer.lower()
    hits = [m.start() for rx in cfm._refusal_markers() for m in [rx.search(low)] if m]
    return min(hits) if hits else None


def _gradients(position, klass_wanted):
    """Within-answer gradient per query: unsupportable rate (2nd half - 1st half)."""
    out = []
    for q, v in position.items():
        if v["klass"] != klass_wanted or v["n1"] < 2 or v["n2"] < 2:
            continue
        out.append((v["u2"] / v["n2"]) - (v["u1"] / v["n1"]))
    return np.array(out, dtype=float)


def did_bootstrap(position, n_boot=10000, seed=SEED):
    """Difference-in-differences on the position gradient, resampling QUERIES in each arm."""
    rng = np.random.default_rng(seed)
    g_dec = _gradients(position, "pure_decline")
    g_ans = _gradients(position, "answered")
    if len(g_dec) < 5 or len(g_ans) < 5:
        return {"n_decline": int(len(g_dec)), "n_answered": int(len(g_ans)),
                "computable": False,
                "why": "fewer than 5 queries with two usable halves in an arm"}
    obs = float(g_dec.mean() - g_ans.mean())
    boot = np.array([float(g_dec[rng.integers(0, len(g_dec), len(g_dec))].mean()
                           - g_ans[rng.integers(0, len(g_ans), len(g_ans))].mean())
                     for _ in range(n_boot)])
    return {
        "computable": True,
        "n_decline": int(len(g_dec)), "n_answered": int(len(g_ans)),
        "gradient_decline_prefix": round(float(g_dec.mean()), 4),
        "gradient_answered": round(float(g_ans.mean()), 4),
        "did": round(obs, 4),
        "boot95": [round(float(np.percentile(boot, 2.5)), 4),
                   round(float(np.percentile(boot, 97.5)), 4)],
        "n_excluded_too_few_claims": None,   # filled by the caller
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2", help="which selection_scores<suffix> to read")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for every output artifact (for example, _v2)")
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()

    scores_dir = EXP_DIR / f"selection_scores{args.suffix}"
    index_path = EXP_DIR / f"selection_scores{args.suffix}_index.json"
    if not index_path.exists():
        sys.exit(f"{index_path.name} not found. Run compute_exp18_selection_bound.py "
                 f"--out-suffix {args.suffix} first: this analysis needs the claim x chunk "
                 f"matrix, and re-deriving it costs a full HHEM pass.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    results = {r["query_id"]: r for r in json.loads(
        (EXP_DIR / "results.json").read_text(encoding="utf-8"))["configs"][BASELINE_KEY]["results"]}

    import scripts.compute_faithfulness_metrics as cfm

    strata = {"d_threshold_artifact": [], "a_synthesis_cand": [],
              "b_parametric_cand": [], "c_unattributed_cand": []}
    position = {}
    excluded_few_claims = []
    marker_offsets = []
    n_claims_total = 0

    for qid, meta in sorted(index.items()):
        S = np.load(scores_dir / f"{qid}.npy")
        claims = meta["claims"]
        assert S.shape[0] == len(claims), f"{qid}: index/matrix out of step"
        answer = results[qid].get("answer") or ""
        klass = classify_response(answer)
        marker = decline_marker_offset(answer, cfm) if klass == "pure_decline" else None
        if marker is not None:
            marker_offsets.append(marker)

        best = S.max(axis=1)
        n_soft = (S > SOFT_TAU).sum(axis=1)
        unsupportable = best <= args.tau
        n_claims_total += len(claims)

        # --- Part 1: within-answer position gradient (claims arrive in answer order)
        if klass in ("pure_decline", "answered") and len(claims) >= 4:
            h = len(claims) // 2
            position[qid] = {
                "klass": klass, "n_claims": len(claims),
                "n1": h, "n2": len(claims) - h,
                "u1": int(unsupportable[:h].sum()), "u2": int(unsupportable[h:].sum()),
                "marker_offset": marker,
            }
        elif klass in ("pure_decline", "answered"):
            excluded_few_claims.append(qid)

        # --- Part 2: candidate strata for the unsupportable claims
        for i in np.nonzero(unsupportable)[0]:
            if best[i] > args.tau - NEAR_TAU_BAND:
                k = "d_threshold_artifact"
            elif n_soft[i] >= 2:
                k = "a_synthesis_cand"
            elif klass == "pure_decline":
                k = "b_parametric_cand"
            else:
                k = "c_unattributed_cand"
            strata[k].append({
                "query_id": qid, "claim_idx": int(i), "claim": claims[i],
                "best_over_pool": round(float(best[i]), 4),
                "n_chunks_over_soft_tau": int(n_soft[i]),
                "decline_class": klass,
            })

    total_unsup = sum(len(v) for v in strata.values())
    pos_stats = did_bootstrap(position)
    pos_stats["n_excluded_too_few_claims"] = len(excluded_few_claims)

    out = {
        "experiment_id": "exp18_evidence_ceiling",
        "analysis": "why 759 claims are supported by no chunk in the k=50 pool",
        "tau": args.tau, "soft_tau": SOFT_TAU, "near_tau_band": NEAR_TAU_BAND,
        "n_queries": len(index), "n_genuine_claims": n_claims_total,
        "n_unsupportable": total_unsup,
        "preregistration": "rule and H1 declared in this script's docstring before any output "
                           "was inspected; DESCRIPTIVE, enters no BH family",
        "h1_v1_withdrawn": {
            "hypothesis": "claims after the decline marker vs before it",
            "status": "WITHDRAWN, never computed — degenerate by construction",
            "why": (f"`pure_decline` is DEFINED as a marker inside the first {cfm.OPENING_WINDOW} "
                    f"chars, so the 'before' stratum is empty. Observed marker offsets over "
                    f"{len(marker_offsets)} decline-prefixed answers: "
                    f"p50={int(np.median(marker_offsets)) if marker_offsets else None}, "
                    f"max={max(marker_offsets) if marker_offsets else None} chars."),
        },
        "h1_position_gradient_did": {
            "hypothesis": "the within-answer unsupportable-rate gradient (2nd half - 1st half) "
                          "is MORE POSITIVE for decline-prefixed answers than for `answered` "
                          "ones — the signature of answering from memory after signalling a gap",
            "estimator": "difference-in-differences; cluster bootstrap over queries, seed 42; "
                         "queries with <4 genuine claims excluded",
            **pos_stats,
        },
        "strata": {k: {"n": len(v), "frac": round(len(v) / total_unsup, 4) if total_unsup else None,
                       "mean_best_over_pool": round(float(np.mean([c["best_over_pool"] for c in v])), 4)
                       if v else None}
                   for k, v in strata.items()},
        "strata_are_candidates_not_verdicts": (
            "(a) synthesis and (c) hallucination cannot be separated by a grounding model -- that "
            "is what the human gold is for. These are strata for annotation. A verdict here would "
            "be the overclaim this phase keeps catching."),
        "generated_by": "scripts/compute_exp18_unsupported_taxonomy.py",
    }
    taxonomy_json = guard_write(EXP_DIR / f"unsupported_taxonomy{args.out_suffix}.json")
    taxonomy_json.write_text(
        json.dumps({**out, "claims_by_stratum": strata}, indent=1, ensure_ascii=False),
        encoding="utf-8")

    # stratified sample for the human gold (proportional floor, capped per stratum)
    rng = np.random.default_rng(SEED)
    sample = []
    for k, v in strata.items():
        if not v:
            continue
        take = min(SAMPLE_PER_STRATUM, len(v))
        for i in rng.choice(len(v), take, replace=False):
            sample.append({"stratum": k, **v[int(i)], "human_verdict": "",
                           "human_notes": ""})
    sample_csv = guard_write(AUDIT_DIR / f"unsupported_claims_sample{args.out_suffix}.csv")
    with sample_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)

    h = out["h1_position_gradient_did"]
    L = [f"# exp18 — por que {total_unsup} claims no los soporta ningun chunk del pool", "",
         f"Sobre {out['n_queries']} queries y {n_claims_total} claims genuinos. "
         f"tau={args.tau}, soft_tau={SOFT_TAU}. **Pre-registrado** en el docstring del script "
         f"antes de mirar salida; DESCRIPTIVO, fuera de la familia BH.", "",
         "## H1 — ¿el modelo se desancla al avanzar, tras señalar que le falta contexto?", "",
         "**H1 v1 retirada, nunca computada:** «claims despues del marcador vs antes» es "
         "degenerada por construccion — " + out["h1_v1_withdrawn"]["why"], ""]
    if h.get("computable"):
        L += [f"Diferencia-en-diferencias sobre el gradiente intra-respuesta "
              f"(2.ª mitad − 1.ª mitad de los claims). {h['n_decline']} respuestas con prefijo "
              f"de declinacion vs {h['n_answered']} `answered`; "
              f"{h['n_excluded_too_few_claims']} excluidas por <4 claims.", "",
              "| grupo | gradiente |", "|---|---|",
              f"| `answered` (control) | {h['gradient_answered']} |",
              f"| **prefijo de declinacion** | **{h['gradient_decline_prefix']}** |",
              f"| **DiD** | **{h['did']}** (IC95 {h['boot95'][0]} a {h['boot95'][1]}) |", "",
              "IC95 que cruza 0 = el gradiente no distingue a los dos grupos, y la hipotesis "
              "de memoria parametrica **no** queda apoyada por esta via.", ""]
    else:
        L += [f"No computable: {h.get('why')} "
              f"(decline={h.get('n_decline')}, answered={h.get('n_answered')}).", ""]
    L += ["## Estratos candidatos (NO son veredictos)", "",
          "| estrato | n | % | mejor score medio |", "|---|---|---|---|"]
    for k, s in out["strata"].items():
        L.append(f"| `{k}` | {s['n']} | {(s['frac'] or 0)*100:.1f} % | {s['mean_best_over_pool']} |")
    L += ["", out["strata_are_candidates_not_verdicts"], "",
          f"Muestra estratificada para anotacion humana: "
          f"`output/audit/unsupported_claims_sample.csv` ({len(sample)} claims)."]
    taxonomy_md = guard_write(EXP_DIR / f"unsupported_taxonomy{args.out_suffix}.md")
    taxonomy_md.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
