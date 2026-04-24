"""
Multidimensional Scorer - Combines cross-encoder score with
recency, source quality, and diversity (MMR).
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SOURCE_QUALITY = {
    "guide": 1.0,
    "api_reference": 1.0,
    "concept": 0.9,
    "task": 0.9,
    "tutorial": 0.8,
    "reference": 0.8,
    "glossary": 0.7,
    "faq": 0.6,
}

DEFAULT_WEIGHTS = {
    "cross_encoder": 0.6,
    "recency": 0.1,
    "source_quality": 0.1,
    "diversity": 0.2,
}


class MultidimensionalScorer:
    """Combines multiple scoring signals for final ranking."""

    def __init__(
        self,
        cross_encoder_reranker=None,
        embedding_manager=None,
        weights: Optional[Dict[str, float]] = None,
        mmr_lambda: float = 0.7,
    ):
        self.cross_encoder = cross_encoder_reranker
        self.embedding_manager = embedding_manager
        self.weights = weights or DEFAULT_WEIGHTS
        self.mmr_lambda = mmr_lambda

    def rerank(
        self,
        query: str,
        candidates: List,
        top_k: int = 5,
    ) -> List:
        """Rerank using multidimensional scoring."""
        if not candidates:
            return candidates

        weights = self.weights

        if self.cross_encoder:
            candidates = self.cross_encoder.rerank(query, candidates, top_k=len(candidates))

        ce_scores = [candidate.score for candidate in candidates]
        ce_min, ce_max = min(ce_scores), max(ce_scores)
        ce_range = ce_max - ce_min if ce_max != ce_min else 1.0

        for candidate in candidates:
            norm_ce = (candidate.score - ce_min) / ce_range
            recency = self._recency_score(candidate)
            quality = SOURCE_QUALITY.get(getattr(candidate, "doc_type", "guide"), 0.5)
            candidate.score = (
                weights["cross_encoder"] * norm_ce
                + weights["recency"] * recency
                + weights["source_quality"] * quality
            )

        if weights.get("diversity", 0) > 0:
            return self._mmr_rerank(query, candidates, top_k)

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[:top_k]

    def _recency_score(self, candidate) -> float:
        """Score based on document recency. More recent = higher score."""
        last_updated = getattr(candidate, "last_updated", "")
        if not last_updated:
            return 0.5

        try:
            normalized = str(last_updated).replace("Z", "+00:00")
            updated_at = datetime.fromisoformat(normalized)
            age_days = (datetime.now(updated_at.tzinfo) - updated_at).days
            return max(0.1, 1.0 / (1 + age_days / 365))
        except (ValueError, TypeError):
            return 0.5

    def _mmr_rerank(self, query: str, candidates: List, top_k: int) -> List:
        """Maximal Marginal Relevance for result diversity."""
        if not self.embedding_manager or len(candidates) <= 1:
            candidates.sort(key=lambda candidate: candidate.score, reverse=True)
            return candidates[:top_k]

        try:
            doc_texts = [candidate.chunk_text for candidate in candidates]
            doc_embeddings = self.embedding_manager.embed_documents(doc_texts, show_progress=False)

            selected = []
            remaining = list(range(len(candidates)))

            for _ in range(min(top_k, len(candidates))):
                best_idx = None
                best_mmr = -float("inf")

                for idx in remaining:
                    relevance = candidates[idx].score
                    max_similarity = 0.0
                    if selected:
                        for selected_idx in selected:
                            similarity = float(np.dot(doc_embeddings[idx], doc_embeddings[selected_idx]))
                            max_similarity = max(max_similarity, similarity)

                    mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_similarity
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)

            return [candidates[idx] for idx in selected]

        except Exception as exc:
            logger.warning("MMR failed, falling back to score sort: %s", exc)
            candidates.sort(key=lambda candidate: candidate.score, reverse=True)
            return candidates[:top_k]
