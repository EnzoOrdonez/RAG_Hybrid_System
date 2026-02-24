"""
Dense Retriever - Control System 2: Pure semantic retrieval.
"""

import logging
import time
from typing import List, Optional

from src.retrieval.bm25_retriever import RetrievalResult
from src.retrieval.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Control System 2: Pure dense (semantic) retrieval."""

    def __init__(self, hybrid_index, query_processor: Optional[QueryProcessor] = None):
        self.index = hybrid_index
        self.query_processor = query_processor or QueryProcessor()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Search using dense embeddings only."""
        start = time.time()

        processed = self.query_processor.process(query)

        # Dense search (semantic query = original, no expansion)
        results_raw = self.index.search_dense(processed.semantic_query, top_k=top_k * 2)

        # Apply provider filter
        if processed.provider_filter:
            results_raw = [
                (cid, score) for cid, score in results_raw
                if self._get_provider(cid) in processed.provider_filter
            ]

        results_raw = results_raw[:top_k]

        # Build results
        results = []
        for chunk_id, score in results_raw:
            chunk_data = self.index.get_chunk(chunk_id) or {}
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                retrieval_method="dense",
                chunk_text=chunk_data.get("text", "")[:200],
                cloud_provider=chunk_data.get("cloud_provider", ""),
                service_name=chunk_data.get("service_name", ""),
                doc_type=chunk_data.get("doc_type", ""),
                heading_path=chunk_data.get("heading_path", ""),
            ))

        elapsed = time.time() - start
        logger.info("Dense search: query='%s', results=%d, time=%.3fs", query[:50], len(results), elapsed)
        return results

    def _get_provider(self, chunk_id: str) -> str:
        chunk = self.index.get_chunk(chunk_id)
        return chunk.get("cloud_provider", "") if chunk else ""
