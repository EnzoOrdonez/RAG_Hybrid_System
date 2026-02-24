"""
BM25 Retriever - Control System 1: Pure lexical retrieval.
"""

import logging
import time
from typing import List, Optional

from pydantic import BaseModel

from src.retrieval.query_processor import QueryProcessor

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """A single retrieval result."""
    chunk_id: str
    score: float
    retrieval_method: str = "bm25"
    chunk_text: str = ""
    cloud_provider: str = ""
    service_name: str = ""
    doc_type: str = ""
    heading_path: str = ""


class BM25Retriever:
    """Control System 1: Pure BM25 lexical retrieval."""

    def __init__(self, hybrid_index, query_processor: Optional[QueryProcessor] = None):
        self.index = hybrid_index
        self.query_processor = query_processor or QueryProcessor()

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_expansion: bool = True,
    ) -> List[RetrievalResult]:
        """Search using BM25 only."""
        start = time.time()

        # Process query
        processed = self.query_processor.process(query)
        search_query = processed.bm25_query if use_expansion else query

        # Search BM25
        results_raw = self.index.search_bm25(search_query, top_k=top_k * 2)

        # Apply provider filter
        if processed.provider_filter:
            results_raw = [
                (cid, score) for cid, score in results_raw
                if self._get_provider(cid) in processed.provider_filter
            ]

        # Normalize scores to [0, 1]
        results_raw = results_raw[:top_k]
        if results_raw:
            max_score = max(s for _, s in results_raw) or 1.0
            results_raw = [(cid, s / max_score) for cid, s in results_raw]

        # Build results
        results = []
        for chunk_id, score in results_raw:
            chunk_data = self.index.get_chunk(chunk_id) or {}
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                retrieval_method="bm25",
                chunk_text=chunk_data.get("text", "")[:200],
                cloud_provider=chunk_data.get("cloud_provider", ""),
                service_name=chunk_data.get("service_name", ""),
                doc_type=chunk_data.get("doc_type", ""),
                heading_path=chunk_data.get("heading_path", ""),
            ))

        elapsed = time.time() - start
        logger.info("BM25 search: query='%s', results=%d, time=%.3fs", query[:50], len(results), elapsed)
        return results

    def _get_provider(self, chunk_id: str) -> str:
        chunk = self.index.get_chunk(chunk_id)
        return chunk.get("cloud_provider", "") if chunk else ""
