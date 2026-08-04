"""exp19a — offline probe: can a CLAIM-conditioned reranker find the grounding chunks?

The gate that decides whether exp19b (a generation run, hours of Enzo's GPU) is worth starting.

THE QUESTION. exp18 established that the k=50 pool supports 58.3 % of the claims the baseline
actually wrote while the 5 chunks selected by topical relevance support only 45.5 %. Relevance
to the QUERY is not the same as support for the CLAIM. exp19's selector is built on that gap:
draft an answer with the current top-5, extract the draft's claims, then re-rank the pool by
(claim, chunk) instead of (query, chunk), and regenerate from the new top-5.

This script measures the middle step alone, with ZERO generation:

    does ms-marco-L12 scored over (claim, chunk) pairs retrieve the chunks that ground those
    claims better than the same model scored over (query, chunk)?

WHAT THIS IS AND IS NOT. The claims used here are the baseline answer's, which is exactly the
draft the deployed selector would condition on — so the SELECTOR side carries no leak. The
EVALUATION side is circular: it asks whether the selection covers the chunks that ground those
same claims, while exp19b's final answer will assert different claims. So this is a measurement
of RETRIEVAL EFFECTIVENESS for the selector mechanism, never an estimate of the downstream
faithfulness effect, and it enters no BH family and no TOST.

GATE, DECLARED BEFORE RUNNING (one-sided, and that asymmetry is the point):
  - FAIL (claim-ranking does NOT beat query-ranking on recall of supporting chunks) => exp19b
    is dead. Under conditions this favourable the mechanism cannot find the evidence, so it
    will not find it with a real draft either. Do not spend the GPU.
  - PASS => proves only that the mechanism is not dead. It does NOT predict a faithfulness
    gain, because the answer changes when the evidence changes.

VERIFIER HYGIENE. The selector uses ms-marco-MiniLM-L-12-v2, the PRODUCTION reranker. No
faithfulness verifier enters the loop, so NLI-small, NLI-base and HHEM all stay clean
evaluators for exp19b. `bge-reranker-large` is deliberately NOT used: it is the independent
oracle for retrieval metrics and must not become part of the method.

SANITY CHECK, mandatory before any number is read: ranking the pool by (query, chunk) with this
same model must reproduce exp18's own top-5 for the baseline arm. If it does not, the harness is
not scoring what exp18 scored and every comparison below is meaningless.

Usage: python scripts/compute_exp19a_selector_probe.py [--suffix _v2] [--tau 0.5]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
Writes experiments/results/exp19a_selector_probe/probe.{json,md}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP18 = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
OUT_DIR = PROJECT_ROOT / "experiments/results/exp19a_selector_probe"
CHUNK_MAP = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
RERANKER = PROJECT_ROOT / "data/models/ms-marco-MiniLM-L-12-v2"
FINAL_K = 5
BATCH = 64
SEED = 42


def load_reranker():
    """The PRODUCTION cross-encoder. Not bge-reranker-large: that stays the independent oracle."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(str(RERANKER), max_length=512)


def select_by_claims(R, k):
    """Greedy over claims: repeatedly take the chunk that best serves the claims still unserved.

    Mean-over-claims would concentrate on chunks that are mildly relevant to everything, which
    is the failure mode of query-level ranking that this whole experiment exists to escape.
    Serving each claim by its own best chunk is what "grounding-guided" has to mean.
    """
    n_claims = R.shape[0]
    chosen, best_so_far = [], np.full(n_claims, -np.inf)
    for _ in range(min(k, R.shape[1])):
        gains = np.maximum(R, best_so_far[:, None]).sum(axis=0) - best_so_far.sum()
        gains[chosen] = -np.inf
        j = int(np.argmax(gains))
        if not np.isfinite(gains[j]):
            break
        chosen.append(j)
        best_so_far = np.maximum(best_so_far, R[:, j])
    return chosen


def faith_of(sup, idx):
    """Fixed-answer faithfulness of a selection: claims with >=1 selected chunk over tau."""
    if not idx:
        return 0.0
    return float(sup[:, idx].any(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-queries", type=int, default=None, help="smoke mode")
    args = ap.parse_args()

    scores_dir = EXP18 / f"selection_scores{args.suffix}"
    index_path = EXP18 / f"selection_scores{args.suffix}_index.json"
    if not index_path.exists():
        sys.exit(f"{index_path.name} not found — run compute_exp18_selection_bound.py "
                 f"--out-suffix {args.suffix} first (this probe needs the HHEM claim x chunk "
                 f"matrix as its ground truth).")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    ids_doc = json.loads((EXP18 / "retrieval_ids.json").read_text(encoding="utf-8"))["ids"]
    chunk_map = json.loads(CHUNK_MAP.read_text(encoding="utf-8"))

    qids = sorted(index)[: args.max_queries] if args.max_queries else sorted(index)
    print(f"loading ms-marco-L12 ... ({len(qids)} queries)", flush=True)
    model = load_reranker()

    rows, sanity = [], {"n_checked": 0, "exact_top5": 0, "mean_overlap": []}
    for n, qid in enumerate(qids, 1):
        meta = index[qid]
        pool_ids, claims = meta["pool_ids"], meta["claims"]
        S = np.load(scores_dir / f"{qid}.npy")
        sup = S > args.tau
        texts = [chunk_map[c]["text"] for c in pool_ids]
        question = ids_doc[qid]["question"]

        # (query, chunk) — the production ranking, and the sanity check
        q_scores = np.array(model.predict([(question, t) for t in texts], batch_size=BATCH))
        q_top = [int(j) for j in np.argsort(-q_scores)[:FINAL_K]]
        base_ids = [c for c in ids_doc[qid]["baseline_repro_ids"] if c in pool_ids]
        base_idx = [pool_ids.index(c) for c in base_ids]
        overlap = len(set(q_top) & set(base_idx))
        sanity["n_checked"] += 1
        sanity["exact_top5"] += int(set(q_top) == set(base_idx))
        sanity["mean_overlap"].append(overlap)

        # (claim, chunk) — the selector under test
        pairs = [(c, t) for c in claims for t in texts]
        cs = []
        for i in range(0, len(pairs), BATCH):
            cs.extend(float(x) for x in model.predict(pairs[i:i + BATCH], batch_size=BATCH))
        R = np.array(cs).reshape(len(claims), len(pool_ids))
        c_top = select_by_claims(R, FINAL_K)

        # ground truth: chunks that actually support at least one claim (HHEM)
        supporting = set(np.nonzero(sup.any(axis=0))[0].tolist())
        rec = (lambda idx: len(set(idx) & supporting) / len(supporting)) if supporting else (lambda idx: None)
        rows.append({
            "qid": qid, "n_claims": len(claims), "n_pool": len(pool_ids),
            "n_supporting_chunks": len(supporting),
            "recall5_query_rank": rec(q_top), "recall5_claim_rank": rec(c_top),
            "recall5_baseline": rec(base_idx),
            "faith_query_rank": faith_of(sup, q_top),
            "faith_claim_rank": faith_of(sup, c_top),
            "faith_baseline": faith_of(sup, base_idx),
            "top5_overlap_with_baseline": overlap,
        })
        if n % 10 == 0:
            print(f"  [{n}/{len(qids)}] {qid} recall5 q={rows[-1]['recall5_query_rank']} "
                  f"claim={rows[-1]['recall5_claim_rank']}", flush=True)

    def mean(field):
        xs = [r[field] for r in rows if r[field] is not None]
        return round(float(np.mean(xs)), 4) if xs else None

    rng = np.random.default_rng(SEED)
    paired = np.array([[r["recall5_claim_rank"], r["recall5_query_rank"]] for r in rows
                       if r["recall5_claim_rank"] is not None])
    diff = float(np.mean(paired[:, 0] - paired[:, 1])) if len(paired) else 0.0
    boot = np.array([float(np.mean(d[:, 0] - d[:, 1])) for d in
                     (paired[rng.integers(0, len(paired), len(paired))] for _ in range(10000))]) \
        if len(paired) else np.array([0.0])

    mean_ov = round(float(np.mean(sanity["mean_overlap"])), 3)
    sanity_ok = mean_ov >= 4.0
    passed = sanity_ok and diff > 0 and float(np.percentile(boot, 2.5)) > 0

    out = {
        "experiment_id": "exp19a_selector_probe", "tau": args.tau, "final_k": FINAL_K,
        "n_queries": len(rows),
        "selector": "ms-marco-MiniLM-L-12-v2 over (claim, chunk); greedy max over claims",
        "verifier_hygiene": ("no faithfulness verifier is used to SELECT, so small/base/HHEM stay "
                             "clean evaluators for exp19b; bge-reranker-large is untouched and "
                             "remains the independent retrieval oracle"),
        "sanity_check": {
            "what": "(query, chunk) ranking with this model must reproduce exp18's baseline top-5",
            "mean_overlap_at_5": mean_ov, "exact_match_rate":
                round(sanity["exact_top5"] / max(1, sanity["n_checked"]), 4),
            "passed": sanity_ok,
            "why": ("below ~4/5 mean overlap the harness is not scoring what exp18 scored and "
                    "every comparison here is meaningless"),
        },
        "recall_of_supporting_chunks@5": {
            "baseline_selection": mean("recall5_baseline"),
            "query_rank": mean("recall5_query_rank"),
            "claim_rank": mean("recall5_claim_rank"),
            "paired_diff_claim_minus_query": round(diff, 4),
            "boot95": [round(float(np.percentile(boot, 2.5)), 4),
                       round(float(np.percentile(boot, 97.5)), 4)],
        },
        "fixed_answer_faithfulness": {
            "baseline_selection": mean("faith_baseline"),
            "query_rank": mean("faith_query_rank"),
            "claim_rank": mean("faith_claim_rank"),
            "caveat": ("CIRCULAR: computed against the very claims the baseline wrote. A real "
                       "selector changes the answer. Never an effect estimate."),
        },
        "gate": {
            "result": "PASS" if passed else "FAIL",
            "rule": ("declared before running: FAIL => exp19b is dead, the mechanism cannot find "
                     "the evidence under conditions this favourable. PASS => the mechanism is not "
                     "dead; it does NOT predict a faithfulness gain."),
        },
        "not_in_any_family": "descriptive probe; enters no BH family and no TOST",
        "per_query": rows, "generated_by": "scripts/compute_exp19a_selector_probe.py",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "probe.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    r5, ff = out["recall_of_supporting_chunks@5"], out["fixed_answer_faithfulness"]
    L = [f"# exp19a — sonda offline del selector (n={len(rows)}, tau {args.tau})", "",
         "Pregunta: reordenar por `(claim, chunk)` con ms-marco-L12, ¿encuentra los chunks que "
         "anclan mejor que reordenar por `(query, chunk)`? **Cero generacion.**", "",
         f"**Sanity check:** solape medio con el top-5 real de exp18 = **{mean_ov}/5** "
         f"({'OK' if sanity_ok else 'FALLA — no leer nada de abajo'}).", "",
         "| seleccion | recall@5 de chunks que anclan | fidelidad a respuesta fija |",
         "|---|---|---|",
         f"| baseline (top-5 de exp18) | {r5['baseline_selection']} | {ff['baseline_selection']} |",
         f"| rerank por query | {r5['query_rank']} | {ff['query_rank']} |",
         f"| **rerank por claim** | **{r5['claim_rank']}** | **{ff['claim_rank']}** |",
         "",
         f"Diferencia pareada (claim − query): **{r5['paired_diff_claim_minus_query']}** "
         f"(IC95 {r5['boot95'][0]} a {r5['boot95'][1]}).", "",
         f"## COMPUERTA: **{out['gate']['result']}**", "", out["gate"]["rule"], "",
         ff["caveat"], "", out["verifier_hygiene"]]
    (OUT_DIR / "probe.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
