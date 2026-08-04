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
If (b) drives a meaningful share, unsupportable claims should concentrate in the text the model
writes AFTER its own decline marker -- the "...However, I can outline general steps..." tail,
where it has explicitly said the context does not cover the question and answers anyway.

  H1 (directional, declared): among `decline_prefix` queries, the unsupportable rate is HIGHER
  for claims positioned after the decline marker than for claims before it.

  Estimator: paired within query (each query contributes both strata, so query-level confounds
  cancel), 95 % CI by cluster bootstrap over queries, seed 42. This is DESCRIPTIVE -- it enters
  no BH family, because it is a mechanism probe, not an arm contrast.

  Guard against the obvious artifact: a claim before the marker is usually a restatement of the
  question, so `not_a_claim` artifacts are already excluded (only genuine claims are counted),
  and the per-stratum claim counts are reported so a lopsided split is visible.

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
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
Writes experiments/results/exp18_evidence_ceiling/unsupported_taxonomy.{json,md}
       + output/audit/unsupported_claims_sample.csv (stratified, for the human gold)
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


def cluster_bootstrap_diff(per_query, n_boot=10000, seed=SEED):
    """Paired within-query difference (after - before), resampling QUERIES."""
    rng = np.random.default_rng(seed)
    qs = [q for q, v in per_query.items() if v["n_before"] and v["n_after"]]
    if not qs:
        return {"n_queries": 0}
    arr = np.array([[per_query[q]["unsup_after"], per_query[q]["n_after"],
                     per_query[q]["unsup_before"], per_query[q]["n_before"]] for q in qs],
                   dtype=float)

    def micro(a):
        return (a[:, 0].sum() / a[:, 1].sum()) - (a[:, 2].sum() / a[:, 3].sum())

    obs = micro(arr)
    boot = np.array([micro(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(n_boot)])
    return {
        "n_queries": len(qs),
        "rate_after": round(float(arr[:, 0].sum() / arr[:, 1].sum()), 4),
        "rate_before": round(float(arr[:, 2].sum() / arr[:, 3].sum()), 4),
        "diff_after_minus_before": round(float(obs), 4),
        "boot95": [round(float(np.percentile(boot, 2.5)), 4),
                   round(float(np.percentile(boot, 97.5)), 4)],
        "n_claims_after": int(arr[:, 1].sum()), "n_claims_before": int(arr[:, 3].sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2", help="which selection_scores<suffix> to read")
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
    n_claims_total = 0

    for qid, meta in sorted(index.items()):
        S = np.load(scores_dir / f"{qid}.npy")
        claims = meta["claims"]
        assert S.shape[0] == len(claims), f"{qid}: index/matrix out of step"
        answer = results[qid].get("answer") or ""
        klass = classify_response(answer)
        marker = decline_marker_offset(answer, cfm) if klass == "pure_decline" else None

        best = S.max(axis=1)
        n_soft = (S > SOFT_TAU).sum(axis=1)
        unsupportable = best <= args.tau
        n_claims_total += len(claims)

        # --- Part 1: position split, only meaningful on decline-prefixed answers
        if marker is not None:
            before = after = ub = ua = 0
            cursor = 0
            for i, c in enumerate(claims):
                # claims come out of _extract_claims in answer order; advance a cursor so a
                # repeated sentence is located at its own occurrence, not the first one.
                pos = answer.find(c[:60], cursor)
                if pos >= 0:
                    cursor = pos + 1
                else:
                    pos = answer.find(c[:30])
                if pos < 0:
                    continue                     # unlocatable; excluded from the split
                if pos < marker:
                    before += 1
                    ub += int(unsupportable[i])
                else:
                    after += 1
                    ua += int(unsupportable[i])
            position[qid] = {"n_before": before, "n_after": after,
                             "unsup_before": ub, "unsup_after": ua}

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
    pos_stats = cluster_bootstrap_diff(position)

    out = {
        "experiment_id": "exp18_evidence_ceiling",
        "analysis": "why 759 claims are supported by no chunk in the k=50 pool",
        "tau": args.tau, "soft_tau": SOFT_TAU, "near_tau_band": NEAR_TAU_BAND,
        "n_queries": len(index), "n_genuine_claims": n_claims_total,
        "n_unsupportable": total_unsup,
        "preregistration": "rule and H1 declared in this script's docstring before any output "
                           "was inspected; DESCRIPTIVE, enters no BH family",
        "h1_position_split": {
            "hypothesis": "among decline-prefixed answers, claims written AFTER the decline "
                          "marker are less grounded than claims written before it",
            "estimator": "paired within query; 95% CI by cluster bootstrap over queries, seed 42",
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
    (EXP_DIR / "unsupported_taxonomy.json").write_text(
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
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with (AUDIT_DIR / "unsupported_claims_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        w.writeheader()
        w.writerows(sample)

    h = out["h1_position_split"]
    L = [f"# exp18 — por que {total_unsup} claims no los soporta ningun chunk del pool", "",
         f"Sobre {out['n_queries']} queries y {n_claims_total} claims genuinos. "
         f"tau={args.tau}, soft_tau={SOFT_TAU}. **Pre-registrado** en el docstring del script "
         f"antes de mirar salida; DESCRIPTIVO, fuera de la familia BH.", "",
         "## H1 — ¿el modelo se desancla DESPUES de su propio marcador de declinacion?", ""]
    if h.get("n_queries"):
        L += [f"Pareado dentro de cada query, {h['n_queries']} queries con ambos estratos "
              f"({h['n_claims_before']} claims antes / {h['n_claims_after']} despues).", "",
              "| estrato | tasa no-soportable |", "|---|---|",
              f"| antes del marcador | {h['rate_before']} |",
              f"| **despues del marcador** | **{h['rate_after']}** |",
              f"| **diferencia** | **{h['diff_after_minus_before']}** (IC95 "
              f"{h['boot95'][0]} a {h['boot95'][1]}) |", ""]
    else:
        L += ["Sin queries con ambos estratos: el split no es computable.", ""]
    L += ["## Estratos candidatos (NO son veredictos)", "",
          "| estrato | n | % | mejor score medio |", "|---|---|---|---|"]
    for k, s in out["strata"].items():
        L.append(f"| `{k}` | {s['n']} | {(s['frac'] or 0)*100:.1f} % | {s['mean_best_over_pool']} |")
    L += ["", out["strata_are_candidates_not_verdicts"], "",
          f"Muestra estratificada para anotacion humana: "
          f"`output/audit/unsupported_claims_sample.csv` ({len(sample)} claims)."]
    (EXP_DIR / "unsupported_taxonomy.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
