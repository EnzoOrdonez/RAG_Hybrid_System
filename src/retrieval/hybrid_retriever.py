"""
Hybrid Retriever - Experimental System: Combined lexical + semantic + reranking.
"""

import logging
import time
from typing import Dict, List, Optional

from src.retrieval.bm25_retriever import RetrievalResult
from src.retrieval.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Experimental System: Hybrid retrieval with fusion and optional reranking."""

    def __init__(
        self,
        hybrid_index,
        query_processor: Optional[QueryProcessor] = None,
        reranker=None,
        fusion_method: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        self.index = hybrid_index
        self.query_processor = query_processor or QueryProcessor()
        self.reranker = reranker
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        top_k_candidates: int = 50,
        fusion: Optional[str] = None,
        alpha: Optional[float] = None,
        use_reranker: bool = True,
        use_expansion: bool = False,
    ) -> List[RetrievalResult]:
        """Search with hybrid fusion and optional reranking.

        use_expansion (D11): when True and fusion='rrf', the BM25 leg uses the
        term-expanded query (processed.bm25_query) while the dense leg keeps the
        raw query. Default False so the main experiments (paper §4.4) are
        unchanged; expansion is evaluated only in the cross-cloud experiment.
        """
        start = time.time()

        fusion = fusion or self.fusion_method
        alpha = alpha if alpha is not None else self.alpha
        processed = self.query_processor.process(query)

        # D11: route the (optionally) expanded query to the BM25 leg only.
        if fusion == "linear":
            hybrid_query = processed.bm25_query  # backward-compatible behavior
            bm25_query = None  # bm25 leg reuses hybrid_query
        else:  # rrf (default)
            hybrid_query = query
            bm25_query = processed.bm25_query if use_expansion else None

        if use_expansion and bm25_query is not None and bm25_query != query:
            logger.info(
                "D11 expansion ON (%s): bm25_query='%s'", fusion, bm25_query[:160]
            )

        # Hybrid search with fusion
        results_raw = self.index.search_hybrid(
            query=hybrid_query,
            top_k=top_k_candidates,
            fusion=fusion,
            alpha=alpha,
            top_k_candidates=top_k_candidates,
            rrf_k=self.rrf_k,
            bm25_query=bm25_query,
        )

        # Apply provider filter
        if processed.provider_filter:
            results_raw = [
                (cid, score) for cid, score in results_raw
                if self._get_provider(cid) in processed.provider_filter
            ]

        # Build RetrievalResult objects
        candidates = []
        for chunk_id, score in results_raw:
            chunk_data = self.index.get_chunk(chunk_id) or {}
            candidates.append(RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                retrieval_method=f"hybrid_{fusion}",
                chunk_text=chunk_data.get("text", ""),
                cloud_provider=chunk_data.get("cloud_provider", ""),
                service_name=chunk_data.get("service_name", ""),
                doc_type=chunk_data.get("doc_type", ""),
                heading_path=chunk_data.get("heading_path", ""),
            ))

        # Optional reranking
        if use_reranker and self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # D12 fix: do NOT truncate chunk_text here. When reranking is applied
        # externally (the pipeline calls search(use_reranker=False) and then
        # reranks separately), the cross-encoder scores c.chunk_text; the old
        # [:200] truncation starved it to ~40 tokens. Keep the full text — the
        # cross-encoder caps the query+passage pair at its own max_length (512).
        # Downstream context is re-fetched from the index (rag_pipeline) and the
        # benchmark output only persists chunk_id, so nothing relies on a short
        # chunk_text.
        elapsed = time.time() - start
        logger.info(
            "Hybrid search (%s, alpha=%.2f): query='%s', results=%d, time=%.3fs",
            fusion, alpha, query[:50], len(candidates), elapsed,
        )
        return candidates

    def grid_search_alpha(
        self,
        query: str,
        alpha_values: List[float] = None,
        fusion_methods: List[str] = None,
        top_k: int = 5,
        top_k_candidates: int = 50,
    ) -> Dict:
        """Run grid search over alpha and fusion methods."""
        alpha_values = alpha_values or [0.3, 0.5, 0.7]
        fusion_methods = fusion_methods or ["linear", "rrf"]

        results = {}
        for fusion in fusion_methods:
            for alpha in alpha_values:
                key = f"{fusion}_alpha_{alpha}"
                res = self.search(
                    query, top_k=top_k, top_k_candidates=top_k_candidates,
                    fusion=fusion, alpha=alpha, use_reranker=False,
                )
                results[key] = {
                    "fusion": fusion,
                    "alpha": alpha,
                    "results": res,
                    "top_ids": [r.chunk_id for r in res],
                }
        return results

    def _get_provider(self, chunk_id: str) -> str:
        chunk = self.index.get_chunk(chunk_id)
        return chunk.get("cloud_provider", "") if chunk else ""
