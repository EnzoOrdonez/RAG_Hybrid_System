"""exp18 — build the four evidence arms that locate the faithfulness ceiling.

The summer phase answered "what does faithfulness respond to?" (evidence SELECTION:
Tier 3 + exp17) but never answered "how high could it go?". Ablation can only remove
components; it cannot measure the ceiling with ideal evidence. That gap is what makes
the cloud spend undecidable, so this runs first and gates everything after it.

Four arms, all from the SAME hybrid candidate pool (k=50) so only the SELECTION differs.
Ids are built for all 194 queries; the RUNNER decides the scale per arm (see ARM_SCALE in
run_exp18_ceiling.py: oracle and top-10 at 194 because their nulls carry decisions, swap at
60 because it expects a large effect):

  baseline_repro   rerank(pool)[:5] with ms-marco-L12          anchor
  oracle_evidence  top-5 by bge-reranker-large over the pool   ceiling of SELECTION
  evidence_swapped top-5 of a DIFFERENT query, same query_type ceiling of ATTENTION
  final_top_k_10   rerank(pool)[:10]                           ceiling of QUANTITY

ANTI-CIRCULARITY (hard rule, Flag 17): the selection oracle is bge-reranker-large, which
is INDEPENDENT of the verifiers that score faithfulness (NLI small/base, HHEM). Selecting
evidence with the instrument that then measures grounding would manufacture the result.

Why `evidence_swapped` is the sharpest arm: if answers barely change when the context is
replaced by another query's evidence, the generator is not reading the context, and no
retrieval improvement could ever have moved faithfulness -- which would explain the whole
0/12 null at once. Pairing is deterministic (seed 42), within query_type, and a derangement
(no query keeps its own evidence).

Why `final_top_k_10` is a boundary probe, not a clean quantity test: at k=5 the granite
4096-token window never binds (0/60 on the subset, 2/194 directly observed in exp12), but at
k=10 it binds on a large minority. So this arm confounds "more evidence" with "truncated
evidence" BY CONSTRUCTION. The estimated prompt size per query is recorded here, and the
runner records the OBSERVED tokens.input, so the analysis can split truncated from
untruncated instead of averaging over the confound. Testing quantity cleanly needs a bigger
window, i.e. the cloud.

Resumable: retrieval is deterministic, so the per-query checkpoint resumes exactly. The
environment has killed long jobs twice; without this the whole oracle pass would be lost.

Out: experiments/results/exp18_evidence_ceiling/{retrieval_ids.json, retrieval_report.md}
Usage: python scripts/build_exp18_evidence_arms.py [--queries all|subset] [--no-oracle]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""
import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp18_build")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS = PROJECT_ROOT / "data" / "models"
SUBSET = PROJECT_ROOT / "data" / "evaluation" / "summer_subset.json"
QUERIES = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
EXP11 = PROJECT_ROOT / "experiments/results/exp11_retrieval194_fullrerank/results.json"
OUT_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
POOL_K = 50
FINAL_K = 5
BIG_K = 10
SEED = 42
CHECKPOINT_EVERY = 10
# chars -> input tokens, fitted on exp12 granite/hibrido (n=192 untruncated, R^2=0.922)
TOK_A, TOK_B = 0.2228, 261
CTX_LIMIT = 4096


def patch_offline_model_paths():
    from src.embedding import embedding_manager as EM
    from src.reranking import cross_encoder_reranker as RR
    EM.MODEL_CONFIGS["bge-large"]["full_name"] = str(MODELS / "bge-large-en-v1.5")
    RR.CROSS_ENCODER_MODELS["ms-marco-mini-12"]["full_name"] = str(MODELS / "ms-marco-MiniLM-L-12-v2")


def load_all_queries():
    q = json.loads(QUERIES.read_text(encoding="utf-8"))
    return q.get("queries", q) if isinstance(q, dict) else q


def subset_ids():
    d = json.loads(SUBSET.read_text(encoding="utf-8"))
    return set(d.get("query_ids") or d.get("ids")
               or [q["query_id"] for q in d.get("queries", [])])


def load_subset():
    ids = subset_ids()
    return [r for r in load_all_queries() if r["query_id"] in ids]


def deranged_pairing(items, key, rng):
    """Map each query to a DIFFERENT query of the same key (a derangement per group).

    A rotation inside each group is the simplest derangement and is fully deterministic
    after the seeded shuffle. Groups of size 1 cannot be deranged within type and fall
    back to a global partner, recorded so the analysis can exclude them if wanted.
    """
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it["query_id"])
    mapping, singletons = {}, []
    for k, qids in sorted(groups.items()):
        qids = sorted(qids)
        rng.shuffle(qids)
        if len(qids) == 1:
            singletons.append(qids[0])
            continue
        for i, q in enumerate(qids):
            mapping[q] = qids[(i + 1) % len(qids)]
    if singletons:  # rotate the leftovers among themselves / against the whole set
        pool = sorted(mapping) or sorted(singletons)
        for q in singletons:
            partner = next(p for p in pool if p != q)
            mapping[q] = partner
    return mapping, singletons


def est_tokens(ids, chunk_map):
    chars = sum(len((chunk_map.get(c) or {}).get("text", "")) for c in ids)
    return TOK_A * chars + TOK_B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the bge-reranker-large arm (SLOW)")
    ap.add_argument("--queries", default="all", choices=["all", "subset"],
                    help="'all' = the 194-query set (default; the oracle and top-10 arms "
                         "need it for power), 'subset' = the 60-query summer subset")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="smoke mode: stop after N queries; writes to retrieval_ids__smokeN.json "
                         "so it can never overwrite the real build")
    args = ap.parse_args()
    patch_offline_model_paths()
    from src.pipeline.rag_pipeline import load_hybrid_index
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.reranking.cross_encoder_reranker import CrossEncoderReranker
    from src.retrieval.query_processor import QueryProcessor
    from src.pipeline.pipeline_config import PROPOSED_HYBRID as CFG

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    subset = load_all_queries() if args.queries == "all" else load_subset()
    in_subset = subset_ids()
    if args.max_queries:
        subset = subset[: args.max_queries]
    suffix = f"__smoke{args.max_queries}" if args.max_queries else ""
    logger.info("queries: %d (%s)%s", len(subset), args.queries,
                " [SMOKE]" if suffix else "")

    exp11 = {r["query_id"]: r["retrieved_ids"] for r in
             json.loads(EXP11.read_text(encoding="utf-8"))["configs"]
             ["RAG Hibrido Propuesto"]["results"]}

    logger.info("loading hybrid index (bge-large) + reranker ...")
    index = load_hybrid_index(embedding_model="bge-large", chunking_strategy="adaptive",
                              chunk_size=500)
    chunk_map = index.chunk_map
    qp = QueryProcessor()
    hybrid = HybridRetriever(index, query_processor=qp, reranker=None,
                             fusion_method=CFG.fusion_method or "rrf", alpha=CFG.alpha,
                             rrf_k=CFG.rrf_k)
    reranker = CrossEncoderReranker(model_name="ms-marco-mini-12")
    oracle = None
    if not args.no_oracle:
        from sentence_transformers import CrossEncoder
        logger.info("loading INDEPENDENT oracle bge-reranker-large ...")
        oracle = CrossEncoder(str(MODELS / "bge-reranker-large"), max_length=512)

    # Group the swap by the ROUTING type (QueryProcessor), not the dataset label. They
    # disagree on 5/60 queries, and it is the routing type that picks the prompt template:
    # pairing by dataset label could hand a cross_cloud-routed query single-provider
    # evidence, so the cross-cloud template would ask to compare providers that are not in
    # the context -- a template/context mismatch confounded with the swap itself.
    routing_type = {it["query_id"]: qp.process(it["question"]).query_type for it in subset}
    swap_of, singletons = deranged_pairing(
        subset, lambda it: routing_type[it["query_id"]], rng)

    # Per-query checkpoint. Retrieval is deterministic, so resuming is exact -- and the
    # environment has killed long jobs twice, which would otherwise throw away the whole
    # oracle pass. Written every CHECKPOINT_EVERY queries and removed on success.
    part_path = OUT_DIR / f"retrieval_ids{suffix}.partial.json"
    pools, ids_out, rows = {}, {}, []
    overlaps = []
    done_qids = set()
    if part_path.exists():
        prev = json.loads(part_path.read_text(encoding="utf-8"))
        ids_out = prev["ids"]
        rows = prev["per_query"]
        overlaps = [r["exp11_overlap@5"] for r in rows if r["exp11_overlap@5"] is not None]
        done_qids = set(ids_out)
        logger.info("resuming: %d queries already built", len(done_qids))

    for n, item in enumerate(subset, 1):
        qid, q = item["query_id"], item["question"]
        if qid in done_qids:
            continue
        cands = hybrid.search(q, top_k=POOL_K, top_k_candidates=POOL_K,
                              use_reranker=False, use_expansion=False)
        reranked = reranker.rerank(q, cands, top_k=POOL_K)
        pool_ids = [r.chunk_id for r in reranked]
        pools[qid] = pool_ids

        base_ids = pool_ids[:FINAL_K]
        big_ids = pool_ids[:BIG_K]
        oracle_ids = None
        if oracle is not None:
            # FULL chunk text, capped by the model's max_length=512 exactly like the
            # production reranker does (D12). exp17's NDCG probe pre-truncated to 800
            # chars, which is harmless for a descriptive metric but NOT here: this arm
            # measures the CEILING of selection, so handicapping the oracle would
            # understate the headroom -- and understating it argues for the cloud. Bias
            # in a decision-relevant direction is the one kind we cannot afford.
            texts = [(chunk_map.get(c) or {}).get("text", "") for c in pool_ids]
            logits = oracle.predict([[q, t] for t in texts], show_progress_bar=False,
                                    batch_size=16)
            order = sorted(range(len(pool_ids)), key=lambda i: -float(logits[i]))
            oracle_ids = [pool_ids[i] for i in order[:FINAL_K]]

        # self-validation: the anchor must reproduce the signed exp11 hybrid top-5
        ov = len(set(base_ids) & set(exp11.get(qid, []))) if qid in exp11 else None
        if ov is not None:
            overlaps.append(ov)

        ids_out[qid] = {"question": q,
                        "dataset_query_type": item["query_type"],
                        "routing_query_type": routing_type[qid],
                        "in_summer_subset": qid in in_subset,
                        # Full reranked candidate pool. Needed by
                        # compute_exp18_selection_bound.py: an upper bound computed over a
                        # subset of the pool is not an upper bound over the pool, and
                        # understating the selection ceiling is exactly the bias that
                        # argues for spending on cloud.
                        "pool_ids": pool_ids,
                        "baseline_repro_ids": base_ids,
                        "final_top_k_10_ids": big_ids,
                        "oracle_evidence_ids": oracle_ids,
                        "swap_partner": swap_of[qid]}
        rows.append({"qid": qid, "query_type": routing_type[qid],
                     "exp11_overlap@5": ov,
                     "oracle_overlap_with_baseline": (
                         len(set(oracle_ids) & set(base_ids)) if oracle_ids else None),
                     "swap_partner": swap_of[qid],
                     "est_tokens_k5": round(est_tokens(base_ids, chunk_map)),
                     "est_tokens_k10": round(est_tokens(big_ids, chunk_map))})
        logger.info("[%d/%d] %s ovlp11=%s oracle∩base=%s", n, len(subset), qid, ov,
                    rows[-1]["oracle_overlap_with_baseline"])
        if len(rows) % CHECKPOINT_EVERY == 0:
            part_path.write_text(json.dumps({"ids": ids_out, "per_query": rows}),
                                 encoding="utf-8")

    # evidence_swapped resolves only after every pool is built
    for qid in ids_out:
        ids_out[qid]["evidence_swapped_ids"] = ids_out[ids_out[qid]["swap_partner"]][
            "baseline_repro_ids"]

    trunc10 = [r["qid"] for r in rows if r["est_tokens_k10"] > CTX_LIMIT]
    trunc5 = [r["qid"] for r in rows if r["est_tokens_k5"] > CTX_LIMIT]
    oracle_ov = [r["oracle_overlap_with_baseline"] for r in rows
                 if r["oracle_overlap_with_baseline"] is not None]

    payload = {
        "experiment_id": "exp18_evidence_ceiling",
        "source": ("test_queries.json (194 q)" if args.queries == "all"
                   else "summer_subset.json (60 q)"),
        "n_queries": len(rows),
        "n_in_summer_subset": sum(1 for r in ids_out.values() if r["in_summer_subset"]),
        "arm_scale": {
            "baseline_repro": "all", "oracle_evidence": "all", "final_top_k_10": "all",
            "evidence_swapped": "subset",
            "why": ("oracle_evidence is the only arm whose NULL must be believed, so it needs "
                    "power for the pre-registered TOST; final_top_k_10 is analysed split by "
                    "truncation and both strata need n (at 60 the split was 24/36); "
                    "evidence_swapped expects a large effect, 60 suffices."),
        },
        "pool_k": POOL_K, "final_k": FINAL_K, "big_k": BIG_K, "seed": SEED,
        "retrieval": "PROPOSED_HYBRID (hybrid rrf, k=50) + ms-marco-L12 rerank",
        "oracle": ("bge-reranker-large — INDEPENDENT of the NLI/HHEM verifiers that score "
                   "faithfulness (anti-circularity, Flag 17)"),
        "swap_rule": ("derangement within the QueryProcessor ROUTING type (not the dataset "
                      "label; they disagree on 5/60 and routing picks the prompt template), "
                      "seed 42; no query keeps its own evidence"),
        "swap_singleton_types": singletons,
        "self_validation_baseline_vs_exp11_mean_overlap@5": (
            round(float(np.mean(overlaps)), 3) if overlaps else None),
        "oracle_vs_baseline_mean_overlap@5": (
            round(float(np.mean(oracle_ov)), 3) if oracle_ov else None),
        "context_budget": {
            "token_model": f"tokens ~= {TOK_A}*chars + {TOK_B} (fit on exp12, R^2 0.922)",
            "limit": CTX_LIMIT,
            "n_truncated_k5": len(trunc5), "n_truncated_k10": len(trunc10),
            "note": ("k=5 never binds the 4096 window; k=10 binds on a large minority, so "
                     "final_top_k_10 confounds quantity with truncation BY CONSTRUCTION. "
                     "Analyse it split by truncation, and read a clean quantity test as "
                     "requiring a bigger window (cloud)."),
            "truncated_k10_qids": trunc10,
        },
        "per_query": rows, "ids": ids_out,
        "generated_by": "scripts/build_exp18_evidence_arms.py",
    }
    part_path.unlink(missing_ok=True)
    (OUT_DIR / f"retrieval_ids{suffix}.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")

    L = [f"# exp18 — brazos de evidencia ({len(rows)} q)", "",
         f"Auto-validación: baseline_repro vs exp11 híbrido firmado, solape@5 medio = "
         f"**{payload['self_validation_baseline_vs_exp11_mean_overlap@5']}/5** (5.0 = réplica exacta).", "",
         f"Oráculo independiente (bge-reranker-large) vs baseline: solape@5 medio "
         f"**{payload['oracle_vs_baseline_mean_overlap@5']}/5** — cuánto margen de selección "
         f"queda por encima del reranker de producción.", "",
         f"Presupuesto de contexto (límite {CTX_LIMIT} tok): k=5 trunca **{len(trunc5)}/{len(rows)}**, "
         f"k=10 trunca **{len(trunc10)}/{len(rows)}**. El brazo top-10 se analiza partido por "
         f"truncamiento; no es un test limpio de cantidad.", "",
         "| qid | tipo | ovlp11@5 | oracle∩base | swap | tok k5 | tok k10 |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['qid']} | {r['query_type']} | {r['exp11_overlap@5']} | "
                 f"{r['oracle_overlap_with_baseline']} | {r['swap_partner']} | "
                 f"{r['est_tokens_k5']} | {r['est_tokens_k10']} |")
    (OUT_DIR / f"retrieval_report{suffix}.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"\nwrote {OUT_DIR/f'retrieval_ids{suffix}.json'} + retrieval_report{suffix}.md")


if __name__ == "__main__":
    main()
