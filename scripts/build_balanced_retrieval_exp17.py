"""exp17 — provider-balanced retrieval for the 25 cross-cloud comparative queries.

Diagnosis (2026-07-24): only 2/25 cross-cloud queries retrieve ALL wanted providers in
the top-5 (23/25 miss a provider entirely -> the comparison is ungroundable), despite
NDCG@5 ~0.85. This is a content-selection failure, the axis Tier 3 flagged as the one that
moves faithfulness. exp13's lexical expansion never addressed it.

This builds two top-5 id lists per query from the SAME hybrid candidate pool (so only the
selection differs, isolating the balancing variable):
  baseline: rerank(pool)[:5]         -> should match exp13 exp_off (self-validation)
  balanced: ceil(5/|P|) per wanted provider P, taken in reranked order -> covers all providers

Replicates exp13 exactly: PROPOSED_HYBRID (hybrid RRF, retrieval_top_k=50, final_top_k=5),
CrossEncoderReranker ms-marco-MiniLM-L-12-v2. Models are loaded from data/models (the HF-id
config is monkeypatched in-memory to the local path so it runs offline; no file edits).

Also reports provider coverage (baseline vs balanced) and an independent-oracle NDCG@5
(bge-reranker-large, sigmoid-graded, within-pool ideal) to measure the coverage/relevance
trade-off. Retrieval is deterministic (no H5).

Out: experiments/results/exp17_crosscloud_balanced/{retrieval_ids.json, retrieval_report.md}
Usage: python scripts/build_balanced_retrieval_exp17.py
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp17_build")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS = PROJECT_ROOT / "data" / "models"
SUBSET = PROJECT_ROOT / "data" / "evaluation" / "cross_cloud_subset.json"
EXP13 = PROJECT_ROOT / "experiments/results/exp13_expansion/results.json"
OUT_DIR = PROJECT_ROOT / "experiments/results/exp17_crosscloud_balanced"
POOL_K = 50           # = PROPOSED_HYBRID.retrieval_top_k (exp13)
FINAL_K = 5           # = final_top_k


def patch_offline_model_paths():
    """Point the bge-large embedder + ms-marco reranker + bge-reranker oracle at
    data/models (in-memory), so SentenceTransformer/CrossEncoder resolve offline."""
    from src.embedding import embedding_manager as EM
    from src.reranking import cross_encoder_reranker as RR
    EM.MODEL_CONFIGS["bge-large"]["full_name"] = str(MODELS / "bge-large-en-v1.5")
    RR.CROSS_ENCODER_MODELS["ms-marco-mini-12"]["full_name"] = str(MODELS / "ms-marco-MiniLM-L-12-v2")


def balance(reranked_ids, provider_of, wanted, k=FINAL_K):
    """Select k ids covering each wanted provider, taken in reranked order.
    quota = ceil(k/|wanted|) per provider; then fill any remainder from the best
    remaining ids (any provider). Preserves reranked order within/across picks."""
    quota = math.ceil(k / len(wanted))
    per = {p: 0 for p in wanted}
    picked, pickset = [], set()
    for cid in reranked_ids:
        p = provider_of(cid)
        if p in per and per[p] < quota and len(picked) < k:
            picked.append(cid); pickset.add(cid); per[p] += 1
    if len(picked) < k:  # fill remainder from best remaining
        for cid in reranked_ids:
            if cid not in pickset:
                picked.append(cid); pickset.add(cid)
                if len(picked) == k:
                    break
    return picked


def ndcg_at_k(sel_ids, rel, k=FINAL_K):
    """Graded NDCG@k: DCG of sel_ids order over rel{} vs ideal (best-k by rel)."""
    def dcg(ids):
        return sum(rel.get(c, 0.0) / math.log2(i + 2) for i, c in enumerate(ids[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return (dcg(sel_ids) / idcg) if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-oracle", action="store_true",
                    help="also compute the independent-oracle NDCG@5 (bge-reranker-large; SLOW, "
                         "~1min/query). Off by default: the ids + coverage + self-validation (the "
                         "pilot's core) don't need it.")
    args = ap.parse_args()
    patch_offline_model_paths()
    from src.pipeline.rag_pipeline import load_hybrid_index
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.reranking.cross_encoder_reranker import CrossEncoderReranker
    from src.retrieval.query_processor import QueryProcessor
    from src.pipeline.pipeline_config import PROPOSED_HYBRID as CFG

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    exp13 = {r["query_id"]: r["retrieved_ids"]
             for r in json.loads(EXP13.read_text(encoding="utf-8"))["configs"]["exp_off | granite4.1-8b"]["results"]}

    logger.info("loading hybrid index (bge-large) + reranker ...")
    index = load_hybrid_index(embedding_model="bge-large", chunking_strategy="adaptive", chunk_size=500)
    chunk_map = index.chunk_map
    qp = QueryProcessor()
    hybrid = HybridRetriever(index, query_processor=qp, reranker=None,
                             fusion_method=CFG.fusion_method or "rrf", alpha=CFG.alpha, rrf_k=CFG.rrf_k)
    reranker = CrossEncoderReranker(model_name="ms-marco-mini-12")
    oracle = None
    if args.with_oracle:
        from sentence_transformers import CrossEncoder
        logger.info("loading oracle bge-reranker-large ...")
        oracle = CrossEncoder(str(MODELS / "bge-reranker-large"), max_length=512)

    def prov(cid):
        return (chunk_map.get(cid) or {}).get("cloud_provider")

    ids_out, rows = {}, []
    cov_base = cov_bal = 0
    ndcg_base, ndcg_bal, overlaps = [], [], []
    for n, item in enumerate(subset, 1):
        qid, q = item["query_id"], item["question"]
        wanted = list(dict.fromkeys(item["cloud_providers"]))  # unique, ordered
        cands = hybrid.search(q, top_k=POOL_K, top_k_candidates=POOL_K,
                              use_reranker=False, use_expansion=False)
        reranked = reranker.rerank(q, cands, top_k=POOL_K)
        pool_ids = [r.chunk_id for r in reranked]
        base_ids = pool_ids[:FINAL_K]
        bal_ids = balance(pool_ids, prov, wanted)

        nb = nl = None
        if oracle is not None:
            # oracle relevance over the pool (independent bge-reranker, sigmoid-graded); text
            # truncated to 800 chars to keep the cross-encoder fast.
            texts = [(chunk_map.get(c) or {}).get("text", "")[:800] for c in pool_ids]
            logits = oracle.predict([[q, t] for t in texts], show_progress_bar=False, batch_size=16)
            rel = {c: float(1 / (1 + math.exp(-float(s)))) for c, s in zip(pool_ids, logits)}
            nb, nl = ndcg_at_k(base_ids, rel), ndcg_at_k(bal_ids, rel)
            ndcg_base.append(nb); ndcg_bal.append(nl)

        wanted_in_pool = {p for p in wanted if any(prov(c) == p for c in pool_ids)}
        base_cov = wanted_in_pool <= {prov(c) for c in base_ids}
        bal_cov = wanted_in_pool <= {prov(c) for c in bal_ids}
        cov_base += base_cov; cov_bal += bal_cov
        ov = len(set(base_ids) & set(exp13.get(qid, []))) if qid in exp13 else None
        if ov is not None:
            overlaps.append(ov)

        ids_out[qid] = {"question": q, "wanted_providers": wanted,
                        "baseline_ids": base_ids, "balanced_ids": bal_ids}
        rows.append({"qid": qid, "wanted": wanted,
                     "base_prov": [prov(c) for c in base_ids],
                     "bal_prov": [prov(c) for c in bal_ids],
                     "base_cov": bool(base_cov), "bal_cov": bool(bal_cov),
                     "ndcg_base": round(nb, 4) if nb is not None else None,
                     "ndcg_bal": round(nl, 4) if nl is not None else None,
                     "exp13_overlap": ov})
        logger.info("[%d/25] %s wanted=%s base_cov=%s bal_cov=%s ovlp13=%s",
                    n, qid, "+".join(wanted), bool(base_cov), bool(bal_cov), ov)

    payload = {"experiment_id": "exp17_crosscloud_balanced",
               "source": "cross_cloud_subset.json (25 q)", "pool_k": POOL_K, "final_k": FINAL_K,
               "retrieval": "PROPOSED_HYBRID (hybrid rrf) + ms-marco-L12 rerank (replicates exp13 exp_off)",
               "oracle": "bge-reranker-large, sigmoid-graded NDCG@5 within pool",
               "self_validation_baseline_vs_exp13_mean_overlap@5": round(float(np.mean(overlaps)), 3) if overlaps else None,
               "coverage": {"baseline": f"{cov_base}/25", "balanced": f"{cov_bal}/25"},
               "ndcg@5_mean": {"baseline": round(float(np.mean(ndcg_base)), 4) if ndcg_base else None,
                               "balanced": round(float(np.mean(ndcg_bal)), 4) if ndcg_bal else None},
               "per_query": rows, "ids": ids_out,
               "generated_by": "scripts/build_balanced_retrieval_exp17.py"}
    (OUT_DIR / "retrieval_ids.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    L = ["# exp17 — provider-balanced retrieval (25 cross-cloud q)", "",
         f"Self-validation: baseline top-5 vs exp13 exp_off mean overlap@5 = "
         f"**{payload['self_validation_baseline_vs_exp13_mean_overlap@5']}/5** (5.0 = perfect replication).", "",
         f"**Provider coverage (all wanted providers present in top-5):** baseline {cov_base}/25 -> "
         f"balanced **{cov_bal}/25**.",
         f"**Oracle NDCG@5 (bge-reranker, within-pool):** baseline {payload['ndcg@5_mean']['baseline']} -> "
         f"balanced {payload['ndcg@5_mean']['balanced']} (trade-off).", "",
         "| qid | wanted | base providers | bal providers | base_cov | bal_cov | ndcg_b | ndcg_bal | ovlp13 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['qid']} | {'+'.join(r['wanted'])} | {','.join(str(x) for x in r['base_prov'])} | "
                 f"{','.join(str(x) for x in r['bal_prov'])} | {r['base_cov']} | {r['bal_cov']} | "
                 f"{r['ndcg_base']} | {r['ndcg_bal']} | {r['exp13_overlap']} |")
    (OUT_DIR / "retrieval_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:9]))
    print(f"\nwrote {OUT_DIR/'retrieval_ids.json'} + retrieval_report.md")


if __name__ == "__main__":
    main()
