"""
Multidimensional Scorer - Combines cross-encoder score with
recency, source quality, and diversity (MMR).
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Source quality scores by doc_type
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

        w = self.weights

        # Step 1: Cross-encoder scores
        if self.cross_encoder:
            candidates = self.cross_encoder.rerank(query, candidates, top_k=len(candidates))

        # Normalize cross-encoder scores to [0, 1]
        ce_scores = [c.score for c in candidates]
        ce_min, ce_max = min(ce_scores), max(ce_scores)
        ce_range = ce_max - ce_min if ce_max != ce_min else 1.0

        for c in candidates:
            norm_ce = (c.score - ce_min) / ce_range

            # Step 2: Recency score
            recency = self._recency_score(c)

            # Step 3: Source quality
            quality = SOURCE_QUALITY.get(getattr(c, "doc_type", "guide"), 0.5)

            # Combined score (without diversity for now)
            c.score = (
                w["cross_encoder"] * norm_ce
                + w["recency"] * recency
                + w["source_quality"] * quality
            )

        # Step 4: MMR diversity re-ranking
        if w.get("diversity", 0) > 0:
            candidates = self._mmr_rerank(query, candidates, top_k)
        else:
            candidates.sort(key=lambda x: x.score, reverse=True)
            candidates = candidates[:top_k]

        return candidates

    def _recency_score(self, candidate) -> float:
        """Score based on document recency. More recent = higher score."""
        # Try to get last_updated from chunk data
        chunk_data = getattr(candidate, "_chunk_data", None)
        if chunk_data and chunk_data.get("last_updated"):
            try:
                dt = datetime.fromisoformat(str(chunk_data["last_updated"]))
                age_days = (datetime.now() - dt).days
                # Decay: 1.0 for today, 0.5 for 1 year old, 0.25 for 2 years
                return max(0.1, 1.0 / (1 + age_days / 365))
            except (ValueError, TypeError):
                pass
        return 0.5  # Default: neutral

    def _mmr_rerank(
        self, query: str, candidates: List, top_k: int
    ) -> List:
        """Maximal Marginal Relevance for result diversity.
        MMR = lambda * sim(query, doc) - (1-lambda) * max(sim(doc, selected))
        """
        if not self.embedding_manager or len(candidates) <= 1:
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:top_k]

        try:
            # Get embeddings for diversity computation
            query_emb = np.array(self.embedding_manager.embed_query(query))
            doc_texts = [c.chunk_text for c in candidates]
            doc_embs = self.embedding_manager.embed_documents(doc_texts, show_progress=False)

            selected = []
            remaining = list(range(len(candidates)))

            for _ in range(min(top_k, len(candidates))):
                best_idx = None
                best_mmr = -float("inf")

                for idx in remaining:
                    # Relevance to query
                    relevance = candidates[idx].score

                    # Max similarity to already selected
                    max_sim = 0.0
                    if selected:
                        for sel_idx in selected:
                            sim = float(np.dot(doc_embs[idx], doc_embs[sel_idx]))
                            max_sim = max(max_sim, sim)

                    mmr = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = idx

                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.remove(best_idx)

            return [candidates[i] for i in selected]

        except Exception as e:
            logger.warning("MMR failed, falling back to score sort: %s", e)
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:top_k]
