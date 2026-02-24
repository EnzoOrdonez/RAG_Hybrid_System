"""
Cross-Encoder Reranker - Scores query-passage pairs directly.
Supports multiple cross-encoder models.
"""

import logging
import time
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODELS = {
    "ms-marco-mini-6": {
        "full_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "description": "Fast baseline cross-encoder",
    },
    "ms-marco-mini-12": {
        "full_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "description": "Better quality cross-encoder",
    },
    "bge-reranker-large": {
        "full_name": "BAAI/bge-reranker-large",
        "description": "State-of-the-art reranker",
    },
}


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "ms-marco-mini-6",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        if model_name not in CROSS_ENCODER_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(CROSS_ENCODER_MODELS.keys())}"
            )
        self.model_name = model_name
        self.config = CROSS_ENCODER_MODELS[model_name]
        self.batch_size = batch_size

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        from sentence_transformers import CrossEncoder

        full_name = self.config["full_name"]
        logger.info("Loading cross-encoder: %s on %s", full_name, self.device)
        start = time.time()
        model = CrossEncoder(full_name, device=self.device)
        elapsed = time.time() - start
        logger.info("Cross-encoder loaded in %.1fs", elapsed)
        return model

    def rerank(
        self,
        query: str,
        candidates: List,
        top_k: int = 5,
    ) -> List:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: The search query
            candidates: List of RetrievalResult objects
            top_k: Number of results to return

        Returns:
            Reranked list of RetrievalResult
        """
        if not candidates:
            return candidates

        start = time.time()

        # Create query-passage pairs
        # Use full chunk text (not truncated) for reranking
        pairs = []
        for c in candidates:
            text = c.chunk_text if c.chunk_text else ""
            pairs.append([query, text])

        # Score all pairs in batch
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Update scores and sort
        for i, c in enumerate(candidates):
            c.score = float(scores[i])
            c.retrieval_method += "+rerank"

        candidates.sort(key=lambda x: x.score, reverse=True)

        elapsed = time.time() - start
        logger.info(
            "Reranked %d candidates in %.3fs (model=%s)",
            len(candidates), elapsed, self.model_name,
        )

        return candidates[:top_k]
