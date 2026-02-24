"""
No Reranker - Baseline for ablation study.
Simply returns top_k candidates without changing order.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class NoReranker:
    """Baseline: no reranking applied. For ablation study."""

    def rerank(
        self,
        query: str,
        candidates: List,
        top_k: int = 5,
    ) -> List:
        """Return first top_k candidates unchanged."""
        return candidates[:top_k]
