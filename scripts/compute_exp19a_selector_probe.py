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
    chosen, best_so_far = [], np.full(R.shape[0], -np.inf)
    for _ in range(min(k, R.shape[1])):
        # Total coverage if chunk j were added. Ranking by coverage is equivalent to ranking by
        # gain (they differ by a constant) and avoids the -inf arithmetic that an explicit gain
        # needs on the first step, where `best_so_far` is still -inf everywhere.
        coverage = np.maximum(R, best_so_far[:, None]).sum(axis=0)
        coverage[chosen] = -np.inf
        j = int(np.argmax(coverage))
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
    bound = json.loads((EXP18 / f"selection_bound{args.suffix}.json")
                       .read_text(encoding="utf-8"))["per_query"]

    qids = sorted(index)[: args.max_queries] if args.max_queries else sorted(index)

    # Checkpoint/resume: this environment kills long jobs, and it has already killed this one.
    # Keyed by query, so a resumed run redoes at most the query in flight.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"__smoke{args.max_queries}" if args.max_queries else ""
    ckpt = OUT_DIR / f"probe{suffix}.partial.json"
    done = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    todo = [q for q in qids if q not in done]
    if done:
        print(f"resuming: {len(done)} queries done, {len(todo)} to go", flush=True)

    print(f"loading ms-marco-L12 ... ({len(todo)} queries)", flush=True)
    model = load_reranker()

    for n, qid in enumerate(todo, 1):
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

        # (claim, chunk) — the selector under test
        pairs = [(c, t) for c in claims for t in texts]
        cs = []
        for i in range(0, len(pairs), BATCH):
            cs.extend(float(x) for x in model.predict(pairs[i:i + BATCH], batch_size=BATCH))
        R = np.array(cs).reshape(len(claims), len(pool_ids))
        c_top = select_by_claims(R, FINAL_K)

        # Chunk-level recall is kept only as a diagnostic: it SATURATES. Most chunks in the pool
        # support some claim (q001: 38 of 50), so recall@5 is ~5/|supporting| for any selection
        # of five supporting chunks and says nothing about whether the RIGHT claims got covered.
        supporting = set(np.nonzero(sup.any(axis=0))[0].tolist())
        rec = (lambda idx: len(set(idx) & supporting) / len(supporting)) if supporting else (lambda idx: None)
        done[qid] = ({
            "qid": qid, "n_claims": len(claims), "n_pool": len(pool_ids),
            "n_supporting_chunks": len(supporting),
            # PRIMARY: claim coverage of the fixed answer, with the bound as ceiling
            "faith_baseline": faith_of(sup, base_idx),
            "faith_query_rank": faith_of(sup, q_top),
            "faith_claim_rank": faith_of(sup, c_top),
            "achievable_k5": bound.get(qid, {}).get("achievable_k5"),
            # diagnostic only
            "recall5_query_rank": rec(q_top), "recall5_claim_rank": rec(c_top),
            "recall5_baseline": rec(base_idx),
            "top5_overlap_with_baseline": overlap,
        })
        ckpt.write_text(json.dumps(done), encoding="utf-8")
        if n % 10 == 0:
            print(f"  [{n}/{len(todo)}] {qid} faith base={done[qid]['faith_baseline']:.3f} "
                  f"claim={done[qid]['faith_claim_rank']:.3f}", flush=True)

    # Derive the work list from what is on disk, and refuse to publish a partial result.
    rows = [done[q] for q in qids if q in done]
    if len(rows) != len(qids):
        sys.exit(f"INCOMPLETE: {len(qids) - len(rows)} of {len(qids)} queries unscored. "
                 f"Re-run to resume from {ckpt.name}; a gate read off a partial sample is worse "
                 f"than no gate.")
    sanity = {"n_checked": len(rows),
              "exact_top5": sum(1 for r in rows if r["top5_overlap_with_baseline"] == FINAL_K),
              "mean_overlap": [r["top5_overlap_with_baseline"] for r in rows]}

    def mean(field):
        xs = [r[field] for r in rows if r[field] is not None]
        return round(float(np.mean(xs)), 4) if xs else None

    rng = np.random.default_rng(SEED)
    paired = np.array([[r["faith_claim_rank"], r["faith_baseline"], r["achievable_k5"]]
                       for r in rows if r["achievable_k5"] is not None])
    diff = float(np.mean(paired[:, 0] - paired[:, 1])) if len(paired) else 0.0
    boot = np.array([float(np.mean(d[:, 0] - d[:, 1])) for d in
                     (paired[rng.integers(0, len(paired), len(paired))] for _ in range(10000))]) \
        if len(paired) else np.array([0.0])
    headroom = float(np.mean(paired[:, 2] - paired[:, 1])) if len(paired) else 0.0

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
        "fixed_answer_claim_coverage": {
            "_primary": True,
            "baseline_selection": mean("faith_baseline"),
            "query_rank": mean("faith_query_rank"),
            "claim_rank": mean("faith_claim_rank"),
            "achievable_k5_ceiling": mean("achievable_k5"),
            "paired_diff_claim_minus_baseline": round(diff, 4),
            "boot95": [round(float(np.percentile(boot, 2.5)), 4),
                       round(float(np.percentile(boot, 97.5)), 4)],
            "headroom_available": round(headroom, 4),
            "frac_of_headroom_closed": round(diff / headroom, 4) if headroom > 0 else None,
            "note_query_rank_equals_baseline": (
                "the pool is stored in production-reranked order, so re-ranking it by (query, "
                "chunk) returns indices 0-4 = the baseline's own top-5. `query_rank` is therefore "
                "a harness check, not a second comparator; the contrast that matters is "
                "claim_rank vs baseline."),
            "caveat": ("CIRCULAR: computed against the very claims the baseline wrote, so this is "
                       "a statement about RETRIEVAL under a fixed answer. `frac_of_headroom_closed` "
                       "is well defined ONLY inside that fixed-answer world -- it is NOT the "
                       "fraction of exp18's +0.128 that exp19b would deliver, because a real "
                       "selector changes the answer. Never an effect estimate."),
        },
        "recall_of_supporting_chunks@5_DIAGNOSTIC": {
            "baseline_selection": mean("recall5_baseline"),
            "query_rank": mean("recall5_query_rank"),
            "claim_rank": mean("recall5_claim_rank"),
            "why_not_primary": ("it saturates: most pool chunks support SOME claim (q001: 38 of "
                                "50), so recall@5 is ~5/|supporting| for any five supporting "
                                "chunks and is blind to whether the right claims got covered"),
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
    (OUT_DIR / f"probe{suffix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    ckpt.unlink(missing_ok=True)

    ff = out["fixed_answer_claim_coverage"]
    r5 = out["recall_of_supporting_chunks@5_DIAGNOSTIC"]
    L = [f"# exp19a — sonda offline del selector (n={len(rows)}, tau {args.tau})", "",
         "Pregunta: reordenar por `(claim, chunk)` con ms-marco-L12, ¿encuentra los chunks que "
         "anclan los claims mejor que el ranking de produccion? **Cero generacion.**", "",
         f"**Sanity check:** solape medio con el top-5 real de exp18 = **{mean_ov}/5** "
         f"({'OK' if sanity_ok else 'FALLA — no leer nada de abajo'}).", "",
         "## Primaria — cobertura de claims a respuesta fija", "",
         "| seleccion | cobertura de claims |", "|---|---|",
         f"| baseline (top-5 de exp18) | {ff['baseline_selection']} |",
         f"| rerank por query (control del harness) | {ff['query_rank']} |",
         f"| **rerank por claim** | **{ff['claim_rank']}** |",
         f"| cota alcanzable k=5 (techo) | {ff['achievable_k5_ceiling']} |", "",
         f"Diferencia pareada (claim − baseline): **{ff['paired_diff_claim_minus_baseline']}** "
         f"(IC95 {ff['boot95'][0]} a {ff['boot95'][1]}). Margen disponible: "
         f"{ff['headroom_available']}. Fraccion del margen cerrada: "
         f"**{ff['frac_of_headroom_closed']}**.", "",
         ff["note_query_rank_equals_baseline"], "",
         f"## COMPUERTA: **{out['gate']['result']}**", "", out["gate"]["rule"], "",
         ff["caveat"], "",
         f"Diagnostico (no primaria): recall@5 de chunks que anclan — baseline "
         f"{r5['baseline_selection']}, claim {r5['claim_rank']}. " + r5["why_not_primary"], "",
         out["verifier_hygiene"]]
    (OUT_DIR / f"probe{suffix}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
