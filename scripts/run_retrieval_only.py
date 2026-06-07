"""
Retrieval-only benchmark over the REBUILT corpus (2697 docs / 24481 chunks)
with the 194-query depurated evaluation set.

NO LLM generation (no Ollama), NO NLI verification. Only the retrieval +
reranking stages are executed, replicating EXACTLY the exp8 configuration by
reusing the same retriever / reranker objects and the same call sequence that
``RAGPipeline.query()`` performs in steps 2-3 (retrieval) and 3 (reranking).

Systems (identical to exp8 / pipeline_config.py):
  - "RAG Lexico (BM25)"      BM25 top-50  -> top-5            (no expansion)
  - "RAG Semantico (Dense)"  Dense top-50 -> top-5
  - "RAG Hibrido Propuesto"  BM25(50)+Dense(50) -> RRF(k=60)
                             -> cross-encoder ms-marco-MiniLM-L-12-v2 -> top-5

Faithful-replication notes (this script does NOT alter exp8 behaviour):
  * Hybrid uses fusion="rrf"; HybridRetriever.search passes the RAW query to
    HybridIndex.search_hybrid for rrf (see hybrid_retriever.py line ~51:
    `processed.bm25_query if fusion == "linear" else query`). Therefore the
    BM25 sub-search inside the hybrid is NOT term-expanded in exp8 either.
  * D12 fix: the separately-applied cross-encoder reranker now scores the FULL
    chunk text (previously HybridRetriever truncated chunk_text to 200 chars
    before the reranker saw it). exp11 re-measures hybrid retrieval with the
    corrected reranker (reranker applied after search, reranker=None inside
    HybridRetriever — mirroring RAGPipeline).
  * The hybrid PRE-rerank RRF order is ALSO recorded as a separate system
    ("RAG Hibrido (pre-rerank RRF)") so retrieval metrics can evidence that
    hybrid >= dense >= lexical does not depend solely on the shared
    cross-encoder (oracle-circularity mitigation).

Output: experiments/results/<exp_id>/results.json in the SAME schema as
experiments/results/exp8/results.json, so scripts/compute_retrieval_metrics.py
consumes it unmodified (generation/hallucination fields are present but empty).

Usage:
  python scripts/run_retrieval_only.py                 # all 194 queries
  python scripts/run_retrieval_only.py --max-queries 5 # smoke test
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import set_all_seeds
from src.pipeline.pipeline_config import (
    BASELINE_LEXICAL,
    BASELINE_SEMANTIC,
    PROPOSED_HYBRID,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_retrieval_only")

QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
SEED = 42


def load_queries(max_queries=None):
    """Load the depurated evaluation queries (file order, deterministic)."""
    data = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if max_queries is not None:
        data = data[:max_queries]
    return data


def build_systems(index):
    """Instantiate retrievers + reranker exactly as RAGPipeline._build_* does."""
    from src.retrieval.query_processor import QueryProcessor
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.reranking.cross_encoder_reranker import CrossEncoderReranker

    # Shared query processor (QueryProcessor() is what the pipeline builds for
    # both the expansion path and the per-retriever default).
    qp = QueryProcessor()

    bm25 = BM25Retriever(index, query_processor=qp)
    dense = DenseRetriever(index, query_processor=qp)
    hybrid = HybridRetriever(
        index,
        query_processor=qp,
        reranker=None,  # reranking handled separately, exactly like the pipeline
        fusion_method=PROPOSED_HYBRID.fusion_method or "rrf",
        alpha=PROPOSED_HYBRID.alpha,
        rrf_k=PROPOSED_HYBRID.rrf_k,
    )

    # PROPOSED_HYBRID.reranker == "cross-encoder/ms-marco-MiniLM-L-12-v2"
    # -> short name "ms-marco-mini-12" (see RAGPipeline._build_reranker map).
    reranker = CrossEncoderReranker(model_name="ms-marco-mini-12")

    return qp, bm25, dense, hybrid, reranker


def main():
    parser = argparse.ArgumentParser(description="Retrieval-only benchmark (no LLM/NLI)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of queries (smoke test). Default: all.")
    parser.add_argument("--exp-id", default="exp11_retrieval194_fullrerank",
                        help="Experiment id / output dir under experiments/results/")
    args = parser.parse_args()

    t0 = time.perf_counter()
    set_all_seeds(SEED)

    timings = {}

    # --- Load queries ---
    queries = load_queries(args.max_queries)
    logger.info("Loaded %d queries from %s", len(queries), QUERIES_PATH.name)

    # --- Load index (FAISS + BM25 + chunk_map) and embedding model ---
    logger.info("Loading hybrid index (bge-large / adaptive / 500) ...")
    t = time.perf_counter()
    from src.pipeline.rag_pipeline import load_hybrid_index
    index = load_hybrid_index(
        embedding_model="bge-large",
        chunking_strategy="adaptive",
        chunk_size=500,
    )
    timings["index_and_embed_model_load_s"] = time.perf_counter() - t
    logger.info("Index loaded in %.1fs (chunks=%d)",
                timings["index_and_embed_model_load_s"], len(index.chunk_map))

    # --- Build systems ---
    qp, bm25, dense, hybrid, reranker = build_systems(index)

    # Force cross-encoder load now so the cost is attributed to "model load"
    # (and any download surfaces immediately, before the loop).
    logger.info("Loading cross-encoder reranker (ms-marco-MiniLM-L-12-v2) ...")
    t = time.perf_counter()
    _ = reranker.model  # triggers lazy load
    timings["reranker_load_s"] = time.perf_counter() - t
    logger.info("Reranker loaded in %.1fs", timings["reranker_load_s"])

    # --- Config-driven parameters (faithful to exp8) ---
    cfg_bm25, cfg_dense, cfg_hyb = BASELINE_LEXICAL, BASELINE_SEMANTIC, PROPOSED_HYBRID
    name_bm25, name_dense, name_hyb = cfg_bm25.name, cfg_dense.name, cfg_hyb.name
    name_hyb_pre = "RAG Hibrido (pre-rerank RRF)"

    results = {name_bm25: [], name_dense: [], name_hyb: [], name_hyb_pre: []}
    stage_s = {"bm25_search": 0.0, "dense_search": 0.0,
               "hybrid_search": 0.0, "hybrid_rerank": 0.0}

    loop_t = time.perf_counter()
    for i, q in enumerate(queries):
        qid = q["query_id"]
        question = q["question"]

        # ---- BM25 (steps 2+3 of RAGPipeline.query, bm25 branch) ----
        t = time.perf_counter()
        cands = bm25.search(
            question,
            top_k=cfg_bm25.retrieval_top_k,
            use_expansion=cfg_bm25.query_expansion,  # False
        )
        reranked = cands[: cfg_bm25.final_top_k]
        stage_s["bm25_search"] += time.perf_counter() - t
        ids_bm25 = [r.chunk_id for r in reranked]

        # ---- Dense ----
        t = time.perf_counter()
        cands = dense.search(question, top_k=cfg_dense.retrieval_top_k)
        reranked = cands[: cfg_dense.final_top_k]
        stage_s["dense_search"] += time.perf_counter() - t
        ids_dense = [r.chunk_id for r in reranked]

        # ---- Hybrid (RRF fusion -> cross-encoder rerank) ----
        t = time.perf_counter()
        cands = hybrid.search(
            question,
            top_k=cfg_hyb.retrieval_top_k,
            top_k_candidates=cfg_hyb.retrieval_top_k,
            use_reranker=False,  # reranker applied separately, like the pipeline
        )
        stage_s["hybrid_search"] += time.perf_counter() - t
        # Pre-rerank RRF order (before the cross-encoder) — recorded as a
        # separate system so retrieval metrics can show the hybrid ordering does
        # NOT depend solely on the shared cross-encoder (oracle-circularity).
        ids_hyb_pre = [r.chunk_id for r in cands[:cfg_hyb.final_top_k]]
        if cands:
            t = time.perf_counter()
            reranked = reranker.rerank(question, cands, top_k=cfg_hyb.final_top_k)
            stage_s["hybrid_rerank"] += time.perf_counter() - t
        else:
            reranked = []
        ids_hyb = [r.chunk_id for r in reranked]

        ts = datetime.now().isoformat()
        for cfg_name, rids in (
            (name_bm25, ids_bm25),
            (name_dense, ids_dense),
            (name_hyb, ids_hyb),
            (name_hyb_pre, ids_hyb_pre),
        ):
            results[cfg_name].append({
                "query_id": qid,
                "config_name": cfg_name,
                "question": question,
                "answer": "",
                "retrieved_ids": rids,
                "relevant_ids": [],
                "retrieval_metrics": {},
                "generation_metrics": {},
                "hallucination_metrics": {},
                "latency": {},
                "error": None,
                "timestamp": ts,
            })

        if (i + 1) % 25 == 0 or (i + 1) == len(queries):
            logger.info("  [%d/%d] processed", i + 1, len(queries))

    timings["retrieval_loop_s"] = time.perf_counter() - loop_t
    timings.update({f"{k}_s": v for k, v in stage_s.items()})

    # --- Write results.json (exp8 schema) ---
    exp_dir = PROJECT_ROOT / "experiments" / "results" / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_path = exp_dir / "results.json"

    timings["total_s"] = time.perf_counter() - t0
    payload = {
        "experiment_id": args.exp_id,
        "name": "Retrieval-only (BM25 / Dense / Hybrid) — rebuilt corpus",
        "description": (
            "Retrieval+reranking only (no LLM, no NLI) over the rebuilt corpus "
            "(2697 docs / 24481 chunks) with the 194-query depurated eval set. "
            "Replicates exp8 retrieval config (RAGPipeline.query steps 2-3)."
        ),
        "hypothesis": "Hybrid (pre- and post-rerank) >= Dense >= BM25 across independent oracles.",
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "corpus": {"docs": 2697, "chunks": len(index.chunk_map)},
        "num_queries": len(queries),
        "timings_seconds": timings,
        "configs": {},
    }
    for cfg_name, qresults in results.items():
        payload["configs"][cfg_name] = {
            "total_queries": len(qresults),
            "errors": 0,
            "results": qresults,
        }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    logger.info("Wrote %s", out_path)

    # --- Console summary ---
    print("\n" + "=" * 64)
    print(f"RETRIEVAL-ONLY RUN — {args.exp_id}")
    print("=" * 64)
    print(f"Queries processed : {len(queries)}")
    for cfg_name in (name_bm25, name_dense, name_hyb, name_hyb_pre):
        rs = results[cfg_name]
        empties = sum(1 for r in rs if not r["retrieved_ids"])
        avg_k = sum(len(r["retrieved_ids"]) for r in rs) / max(1, len(rs))
        print(f"  {cfg_name:<26} n={len(rs):<4} avg_retrieved={avg_k:.2f} empty={empties}")
    print("-" * 64)
    print("Timings (s):")
    for k in ("index_and_embed_model_load_s", "reranker_load_s",
              "bm25_search_s", "dense_search_s",
              "hybrid_search_s", "hybrid_rerank_s",
              "retrieval_loop_s", "total_s"):
        if k in timings:
            print(f"  {k:<34} {timings[k]:8.2f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
