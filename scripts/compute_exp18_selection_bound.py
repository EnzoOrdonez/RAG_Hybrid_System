"""exp18 — how much faithfulness could ANY evidence selection buy? (bracketed, not guessed)

The `oracle_evidence` arm selects with bge-reranker-large, which ranks by TOPICAL RELEVANCE
to the query. The metric asks something else: are the claims the model DECIDED TO ASSERT
supported by the given chunks. A chunk can be maximally relevant and still not contain the
specific fact the model asserted. So that arm measures the ceiling of relevance-optimal
selection, which is a LOWER bound on the selection ceiling -- and its null, alone, cannot
support "there is no selection headroom", which is the claim the cloud spend rests on.

This closes that gap. For each query it takes the claims the BASELINE ALREADY WROTE and
scores them against ALL 50 pool chunks, then reports two quantities that bracket the true
k=5 optimum:

  upper_bound       claims supportable by ANY pool chunk (max over all 50 > tau), / genuine.
                    A STRICT UPPER BOUND: no 5-chunk selection can support a claim that no
                    chunk in the pool supports. Needs no combinatorics.
  achievable_k5     greedy max-coverage over the pool with k=5, floored at the baseline's own
                    subset. A CONSTRUCTIVE LOWER BOUND on the k=5 optimum: this selection
                    exists and achieves this number.

Why both. Choosing the best 5 of 50 to maximise covered claims is max-coverage, NP-hard;
greedy only guarantees (1-1/e)~0.63 of the optimum, so greedy alone would UNDERSTATE the
ceiling -- and understating the selection ceiling is precisely the bias that argues for
spending on cloud. Reporting the bracket avoids claiming either direction.

CIRCULAR BY CONSTRUCTION, and declared as such: the selection is chosen using the answer's
own claims, so this is NOT an effect estimate, NOT an experimental arm, and NEVER enters the
BH family or the TOST. Its value is logical, not inferential: no non-circular selector can
beat `upper_bound`. Read it as:

  upper_bound ~ baseline      selection is genuinely exhausted -> the ceiling is elsewhere
                              (capacity / instrument), and the cloud diagnostic is justified
  achievable_k5 >> baseline   headroom EXISTS and is reachable with 5 chunks; the oracle's
                              null then means relevance ranking cannot find it, and the
                              missing piece is a grounding-guided selector -- a LOCAL method,
                              no spend required

Scoring mirrors rescore_grounding_exp15.py exactly (HHEM-2.1, premise truncated to 1500
chars, batch 16, supported iff max chunk p > tau) so the numbers are directly comparable to
every other faithfulness figure in the phase.

Usage: python scripts/compute_exp18_selection_bound.py [--tau 0.5] [--max-queries N]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import (  # noqa: E402
    HallucinationDetector, classify_artifact)
from scripts.rescore_grounding_exp15 import load_hhem  # noqa: E402

EXP_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
BASELINE_ARM = "baseline_repro"
FINAL_K = 5
PREMISE_CHARS = 1500   # == rescore_grounding_exp15
BATCH = 16             # T5 on 6 GB


def greedy_cover(cov, k):
    """Greedy max-coverage: pick k chunks covering the most still-uncovered claims.

    cov[j] = set of claim indices chunk j supports. Greedy is (1-1/e)-optimal, which is why
    the result is reported as a LOWER bound and floored at the baseline's own subset.
    """
    chosen, covered = [], set()
    for _ in range(min(k, len(cov))):
        best, gain = None, -1
        for j in range(len(cov)):
            if j in chosen:
                continue
            g = len(cov[j] - covered)
            if g > gain:
                best, gain = j, g
        if best is None or gain <= 0:
            break
        chosen.append(best)
        covered |= cov[best]
    return chosen, covered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="smoke mode; writes to a __smokeN file that cannot overwrite the real one")
    ap.add_argument("--out-suffix", default="",
                    help="write selection_bound<suffix>.json instead of overwriting the committed one")
    args = ap.parse_args()
    suffix = f"__smoke{args.max_queries}" if args.max_queries else args.out_suffix

    ids_doc = json.loads((EXP_DIR / "retrieval_ids.json").read_text(encoding="utf-8"))["ids"]
    results = json.loads((EXP_DIR / "results.json").read_text(encoding="utf-8"))["configs"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))
    base_cfg = next(k for k, c in results.items() if c.get("scenario") == BASELINE_ARM)
    rows = results[base_cfg]["results"]
    if args.max_queries:
        rows = rows[: args.max_queries]

    det = HallucinationDetector(use_nli=False)
    print(f"loading HHEM-2.1 ... ({len(rows)} queries)", flush=True)
    model = load_hhem()

    # The claim x chunk score matrix is the expensive part of this script and, until now, it
    # was thrown away after the aggregates were taken. Everything downstream that asks a
    # finer question than "how many claims clear tau" -- the unsupported-claim taxonomy, the
    # exp19 selector probe, the gold sample -- needs the matrix itself, and re-deriving it
    # costs another full HHEM pass. It is persisted per query so a resumed run keeps it too.
    scores_dir = EXP_DIR / f"selection_scores{suffix}"
    scores_dir.mkdir(exist_ok=True)
    index_path = EXP_DIR / f"selection_scores{suffix}_index.json"
    score_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}

    part = EXP_DIR / f"selection_bound{suffix}.partial.json"
    per_query = json.loads(part.read_text(encoding="utf-8")) if part.exists() else {}
    # A query counts as done only if BOTH its aggregates and its matrix are on disk. Resuming
    # on the aggregates alone would silently produce an index with holes -- the same shape of
    # defect as the pass_N failure: a work list derived from the convenient artifact.
    per_query = {q: v for q, v in per_query.items()
                 if v.get("genuine") in (0, None) or q in score_index}
    if per_query:
        print(f"resuming: {len(per_query)} queries done", flush=True)

    t0 = time.time()
    for n, r in enumerate(rows, 1):
        qid = r["query_id"]
        if qid in per_query:
            continue
        answer = r.get("answer") or ""
        claims = det._extract_claims(answer) if answer.strip() else []
        genuine = [c for c in claims if not classify_artifact(c)]
        pool_ids = ids_doc[qid].get("pool_ids")
        if not pool_ids:
            sys.exit(
                f"{qid}: retrieval_ids.json has no `pool_ids`. The bound MUST be computed "
                f"over the full k=50 candidate pool -- a bound over a subset of the pool is "
                f"not a bound over the pool, and it would understate the selection ceiling. "
                f"Re-run: python scripts/build_exp18_evidence_arms.py --queries all")
        pool_ids = [c for c in pool_ids if c in chunk_map]
        if not genuine or not pool_ids:
            per_query[qid] = {"genuine": 0, "baseline_faith": None,
                              "upper_bound": None, "achievable_k5": None, "n_pool": len(pool_ids)}
            continue

        texts = [chunk_map[c]["text"][:PREMISE_CHARS] for c in pool_ids]
        pairs = [(t, cl) for cl in genuine for t in texts]
        scores = []
        import torch
        for i in range(0, len(pairs), BATCH):
            with torch.no_grad():
                scores.extend(float(x) for x in model.predict(pairs[i:i + BATCH]))
        S = np.array(scores).reshape(len(genuine), len(pool_ids))   # claim x chunk
        np.save(scores_dir / f"{qid}.npy", S.astype(np.float32))
        score_index[qid] = {"pool_ids": pool_ids, "claims": genuine,
                            "shape": [len(genuine), len(pool_ids)]}

        sup = S > args.tau
        # strict upper bound: any chunk in the pool may support the claim
        upper = int(sup.any(axis=1).sum()) / len(genuine)
        # constructive lower bound on the k=5 optimum
        cov = [set(np.nonzero(sup[:, j])[0].tolist()) for j in range(len(pool_ids))]
        _, covered = greedy_cover(cov, FINAL_K)
        base_idx = [pool_ids.index(c) for c in ids_doc[qid]["baseline_repro_ids"]
                    if c in pool_ids]
        base_cov = set(np.nonzero(sup[:, base_idx].any(axis=1))[0].tolist()) if base_idx else set()
        achievable = max(len(covered), len(base_cov)) / len(genuine)

        # Diagnostics for the claims NO pool chunk supports. Without these, "the pool cannot
        # support it" is indistinguishable from "it sits just under tau": a max of 0.02 means
        # the claim is genuinely absent from the retrieved evidence, a max of 0.49 means the
        # bound is an artefact of where the threshold was put.
        best_over_pool = S.max(axis=1)
        unsup = best_over_pool[~sup.any(axis=1)]
        per_query[qid] = {
            "genuine": len(genuine), "n_pool": len(pool_ids),
            "baseline_faith": round(len(base_cov) / len(genuine), 4),
            "upper_bound": round(upper, 4),
            "achievable_k5": round(achievable, 4),
            "n_unsupportable": int(len(unsup)),
            "unsupportable_best_score": [round(float(x), 4) for x in unsup],
            "near_tau_unsupportable": int((unsup > args.tau - 0.1).sum()),
        }
        if n % 10 == 0:
            part.write_text(json.dumps(per_query), encoding="utf-8")
            index_path.write_text(json.dumps(score_index), encoding="utf-8")
            print(f"  [{n}/{len(rows)}] {qid} base={per_query[qid]['baseline_faith']} "
                  f"k5={per_query[qid]['achievable_k5']} upper={per_query[qid]['upper_bound']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    index_path.write_text(json.dumps(score_index), encoding="utf-8")
    # Record what was actually persisted, and refuse to publish an index with holes.
    scored_with_claims = {q for q, v in per_query.items() if v.get("genuine")}
    missing = sorted(scored_with_claims - set(score_index))
    if missing:
        sys.exit(f"INCOMPLETE score index: {len(missing)} queries have aggregates but no "
                 f"matrix on disk ({missing[:5]}...). Delete "
                 f"{part.name} and re-run so the two stay in step.")

    vals = [v for v in per_query.values() if v.get("upper_bound") is not None]
    all_unsup = [s for v in vals for s in v["unsupportable_best_score"]]
    near = sum(v["near_tau_unsupportable"] for v in vals)
    # validity: the bracket must hold in EVERY query, by construction
    bad = [q for q, v in per_query.items() if v.get("upper_bound") is not None
           and not (v["baseline_faith"] <= v["achievable_k5"] + 1e-9 <= v["upper_bound"] + 1e-9)]
    out = {
        "experiment_id": "exp18_evidence_ceiling", "tau": args.tau,
        "n_queries_scored": len(vals),
        "baseline_mean": round(float(np.mean([v["baseline_faith"] for v in vals])), 4),
        "achievable_k5_mean": round(float(np.mean([v["achievable_k5"] for v in vals])), 4),
        "upper_bound_mean": round(float(np.mean([v["upper_bound"] for v in vals])), 4),
        "bracket_violations": bad,
        "unsupportable_claims": {
            "n": len(all_unsup),
            "best_score_over_pool": {
                "mean": round(float(np.mean(all_unsup)), 4) if all_unsup else None,
                "p50": round(float(np.median(all_unsup)), 4) if all_unsup else None,
                "p90": round(float(np.percentile(all_unsup, 90)), 4) if all_unsup else None,
            },
            "n_within_0.1_of_tau": near,
            "why": ("Claims no pool chunk supports. If their best score is far below tau, the "
                    "claim is genuinely absent from the retrieved evidence and no selection "
                    "could ever ground it. If it clusters just under tau, the bound is an "
                    "artefact of the threshold, not of the evidence."),
        },
        "status": "CIRCULAR BY CONSTRUCTION — upper/achievable are bounds, not effect "
                  "estimates. Never enter the BH family or the TOST.",
        "reading": ("upper_bound ~ baseline => selection genuinely exhausted (ceiling is "
                    "capacity or instrument); achievable_k5 >> baseline => headroom exists "
                    "and is reachable with 5 chunks, so the oracle's null means relevance "
                    "ranking cannot find it and the missing piece is a grounding-guided "
                    "selector, which is LOCAL."),
        # Added 2026-08-04 (ledger entry 22), before exp19 is designed around this number.
        "not_a_target": (
            "This bound holds the ANSWER FIXED and only re-picks chunks under it. A real "
            "selector changes the answer, hence both numerator and denominator: exp18's own "
            "oracle arm -- a mild selection change -- already shows baseline_claim_reappearance "
            "0.0132, i.e. ~99 % of claims differ. So the gap achievable_k5 - baseline does NOT "
            "bound, and is not a target for, any non-circular selector; reporting a 'fraction "
            "of the gap recovered' would be a fabricated quantity. What the bound does "
            "establish is factual and enough to motivate exp19: the k=50 pool supports "
            "achievable_k5 of the claims the baseline actually wrote while the 5 selected "
            "chunks support only baseline_mean."),
        "score_matrix": {
            "dir": scores_dir.name,
            "index": index_path.name,
            "layout": "one float32 .npy per query, shape [genuine_claims x pool_chunks]; the "
                      "index gives the claim texts and the pool_ids in matching order",
            "n_queries": len(score_index),
        },
        "per_query": per_query,
        "generated_by": "scripts/compute_exp18_selection_bound.py",
    }
    (EXP_DIR / f"selection_bound{suffix}.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    part.unlink(missing_ok=True)

    L = [f"# exp18 — cota de seleccion (HHEM tau {args.tau}, n={len(vals)})", "",
         "**Circular por construccion**: la seleccion se elige usando los claims que la propia "
         "respuesta escribio. Son COTAS, no estimaciones de efecto. No entran en la familia BH "
         "ni en el TOST.", "",
         "| | media |", "|---|---|",
         f"| baseline (su propio top-5) | {out['baseline_mean']} |",
         f"| **alcanzable con k=5** (greedy, suelo=baseline) | **{out['achievable_k5_mean']}** |",
         f"| **cota superior** (cualquier chunk del pool) | **{out['upper_bound_mean']}** |", "",
         f"Violaciones del bracket: **{len(bad)}** (debe ser 0; "
         f"baseline <= alcanzable <= cota por construccion).", "",
         f"**Claims que NINGUN chunk del pool soporta:** {len(all_unsup)}. "
         f"Su mejor score sobre el pool: media {out['unsupportable_claims']['best_score_over_pool']['mean']}, "
         f"p50 {out['unsupportable_claims']['best_score_over_pool']['p50']}, "
         f"p90 {out['unsupportable_claims']['best_score_over_pool']['p90']} (tau {args.tau}). "
         f"A menos de 0,1 del umbral: **{near}**.", "",
         "Si esos scores estan muy por debajo de tau, el claim sencillamente NO esta en la "
         "evidencia recuperada y ninguna seleccion podria anclarlo; si se agolpan justo bajo "
         "tau, la cota es artefacto del umbral y no de la evidencia.", "", out["reading"]]
    (EXP_DIR / f"selection_bound{suffix}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if bad:
        sys.exit(f"\nBRACKET VIOLADO en {len(bad)} queries -> hay un bug, no interpretar")


if __name__ == "__main__":
    main()
