"""
Build blind annotation pool for human relevance judgments.

For each query in data/evaluation/gold_queries_50.json, unions top-k results from
BM25 + Dense + Hybrid (no reranker, no query expansion, no terminology normalization
-- raw retrieval only so judgments are portable across ablations). Deduplicates per
query, randomizes per-query chunk order with a deterministic seed derived from the
query_id, and writes:

  data/evaluation/annotation_pool_full.json     (full metadata for audit)
  data/evaluation/annotation_enzo.csv           (blind export, no source_systems)
  data/evaluation/annotation_advisor.csv        (blind export)
  data/evaluation/annotation_classmate.csv      (blind export)

The three CSVs share identical row content and row order so that annotator
agreement can be measured directly (Cohen's kappa / Krippendorff's alpha).

Do NOT use the cross-encoder reranker or any LLM here. The pool is
retrieval-only. The reranker is part of the Hybrid *pipeline* under evaluation;
mixing it into the annotation pool would re-introduce the Flag-17 circularity
that this work is meant to break.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


GOLD_QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "gold_queries_50.json"
OUTPUT_POOL_PATH = PROJECT_ROOT / "data" / "evaluation" / "annotation_pool_full.json"
OUTPUT_CSV_PATHS = {
    "enzo": PROJECT_ROOT / "data" / "evaluation" / "annotation_enzo.csv",
    "advisor": PROJECT_ROOT / "data" / "evaluation" / "annotation_advisor.csv",
    "classmate": PROJECT_ROOT / "data" / "evaluation" / "annotation_classmate.csv",
}
INDICES_DIR = PROJECT_ROOT / "data" / "indices"
EMBEDDING_MODEL = "bge-large"
CHUNK_STRATEGY = "adaptive"
CHUNK_SIZE = 500
TOP_K_DEFAULT = 10
PREVIEW_CHARS = 800


def _whitespace_collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _preview(text: str, n: int = PREVIEW_CHARS) -> str:
    return _whitespace_collapse(text)[:n]


def _seed_for_query(query_id: str) -> int:
    # Deterministic, platform-independent seed (hash() differs across runs).
    h = 0
    for ch in query_id:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def load_hybrid_index():
    from src.embedding.embedding_manager import EmbeddingManager
    from src.embedding.index.hybrid_index import HybridIndex

    embedding_manager = EmbeddingManager(
        model_name=EMBEDDING_MODEL,
        cache_dir=str(PROJECT_ROOT / "data" / "embeddings"),
        batch_size=64,
    )
    hybrid_index = HybridIndex(
        embedding_manager=embedding_manager,
        indices_dir=str(INDICES_DIR),
    )
    hybrid_index.load(chunk_strategy=CHUNK_STRATEGY, chunk_size=CHUNK_SIZE)
    logger.info("Loaded index: chunks=%d", len(hybrid_index.chunk_map))
    return hybrid_index


def build_retrievers(hybrid_index):
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.query_processor import QueryProcessor

    qp = QueryProcessor()
    bm25 = BM25Retriever(hybrid_index, query_processor=qp)
    dense = DenseRetriever(hybrid_index, query_processor=qp)
    hybrid = HybridRetriever(
        hybrid_index,
        query_processor=qp,
        reranker=None,
        fusion_method="rrf",
        alpha=0.5,
        rrf_k=60,
    )
    return bm25, dense, hybrid


def _retrieve_raw(retriever, name: str, question: str, top_k: int) -> List:
    common = dict(
        top_k=top_k,
        enable_query_expansion=False,
        enable_terminology_normalization=False,
    )
    if name == "bm25":
        return retriever.search(question, use_expansion=False, **common)
    if name == "dense":
        return retriever.search(question, **common)
    if name == "hybrid":
        return retriever.search(question, use_reranker=False, **common)
    raise ValueError(f"Unknown retriever name: {name}")


def build_pool_for_query(
    query: Dict,
    bm25,
    dense,
    hybrid,
    hybrid_index,
    top_k: int,
) -> Dict:
    question = query["question"]
    qid = query["query_id"]

    # Run 3 retrievers, collect rank per system
    ranks: Dict[str, Dict[str, int]] = {}  # chunk_id -> {bm25:rank, dense:rank, hybrid:rank}
    for name, retriever in (("bm25", bm25), ("dense", dense), ("hybrid", hybrid)):
        results = _retrieve_raw(retriever, name, question, top_k)
        for i, r in enumerate(results):
            cid = r.chunk_id
            ranks.setdefault(cid, {})[name] = i + 1  # 1-indexed rank

    # Randomize order deterministically per query
    chunk_ids = sorted(ranks.keys())  # stable base order before shuffle
    rng = random.Random(_seed_for_query(qid))
    rng.shuffle(chunk_ids)

    chunks_out = []
    for pos, cid in enumerate(chunk_ids):
        chunk_data = hybrid_index.get_chunk(cid) or {}
        text = chunk_data.get("text", "")
        source_systems = sorted(ranks[cid].keys())
        rank_dict = OrderedDict([
            ("bm25", ranks[cid].get("bm25")),
            ("dense", ranks[cid].get("dense")),
            ("hybrid", ranks[cid].get("hybrid")),
        ])
        chunks_out.append(OrderedDict([
            ("chunk_id", cid),
            ("source_systems", source_systems),
            ("ranks", rank_dict),
            ("randomized_position", pos),
            ("cloud_provider", chunk_data.get("cloud_provider", "")),
            ("service_name", chunk_data.get("service_name", "")),
            ("doc_type", chunk_data.get("doc_type", "")),
            ("heading_path", chunk_data.get("heading_path", "")),
            ("text_preview", _preview(text)),
            ("full_text", text),
        ]))

    return OrderedDict([
        ("query_id", qid),
        ("query_text", question),
        ("cloud_providers", query.get("cloud_providers", [])),
        ("sampling_stratum", query.get("sampling_stratum")),
        ("n_chunks", len(chunks_out)),
        ("chunks", chunks_out),
    ])


def write_csvs(pool: List[Dict]) -> None:
    rows = []
    for entry in pool:
        for chunk in entry["chunks"]:
            rows.append(OrderedDict([
                ("query_id", entry["query_id"]),
                ("query_text", entry["query_text"]),
                ("chunk_id", chunk["chunk_id"]),
                ("chunk_text_preview", chunk["text_preview"]),
                ("relevance", ""),
                ("notes", ""),
            ]))
    # Sort by query_id then by randomized position within the pool entry
    # (already in randomized order inside each entry, but sort keeps CSVs stable)
    rows.sort(key=lambda r: (r["query_id"],))
    # We cannot rely on dict insertion to preserve per-query order after the
    # first sort, so re-sort with a compound key using a helper list.
    # The pool is already sorted within each query; regenerate preserving order.
    rows = []
    for entry in sorted(pool, key=lambda e: e["query_id"]):
        for chunk in entry["chunks"]:
            rows.append(OrderedDict([
                ("query_id", entry["query_id"]),
                ("query_text", entry["query_text"]),
                ("chunk_id", chunk["chunk_id"]),
                ("chunk_text_preview", chunk["text_preview"]),
                ("relevance", ""),
                ("notes", ""),
            ]))

    for annotator, path in OUTPUT_CSV_PATHS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["query_id", "query_text", "chunk_id", "chunk_text_preview", "relevance", "notes"],
                quoting=csv.QUOTE_ALL,
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        logger.info("Wrote %s: %d rows", path.name, len(rows))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    args = ap.parse_args()

    if not GOLD_QUERIES_PATH.exists():
        logger.error("Gold queries missing: %s", GOLD_QUERIES_PATH)
        return 1

    payload = json.loads(GOLD_QUERIES_PATH.read_text(encoding="utf-8"))
    queries = payload.get("queries", payload)
    logger.info("Loaded %d gold queries", len(queries))

    hybrid_index = load_hybrid_index()
    bm25, dense, hybrid = build_retrievers(hybrid_index)

    pool = []
    for i, q in enumerate(queries, start=1):
        logger.info("[%d/%d] %s", i, len(queries), q["query_id"])
        pool.append(build_pool_for_query(q, bm25, dense, hybrid, hybrid_index, args.top_k))

    OUTPUT_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = OrderedDict([
        ("generated_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ("source_queries", str(GOLD_QUERIES_PATH.relative_to(PROJECT_ROOT))),
        ("retrievers", ["bm25", "dense", "hybrid(rrf,alpha=0.5,rrf_k=60)"]),
        ("retriever_flags", {
            "enable_query_expansion": False,
            "enable_terminology_normalization": False,
            "use_reranker": False,
        }),
        ("top_k_per_retriever", args.top_k),
        ("per_query_shuffle_seed", "polynomial hash of query_id (see _seed_for_query in scripts/build_annotation_pool.py)"),
        ("total_queries", len(pool)),
        ("avg_chunks_per_query", round(sum(e["n_chunks"] for e in pool) / max(1, len(pool)), 2)),
        ("min_chunks_per_query", min((e["n_chunks"] for e in pool), default=0)),
        ("max_chunks_per_query", max((e["n_chunks"] for e in pool), default=0)),
    ])
    out = OrderedDict([("metadata", meta), ("queries", pool)])
    with open(OUTPUT_POOL_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %s (%d queries, avg %.1f chunks)", OUTPUT_POOL_PATH, len(pool), meta["avg_chunks_per_query"])

    write_csvs(pool)

    return 0


if __name__ == "__main__":
    sys.exit(main())
