"""
Hybrid Index - Maintains both FAISS and BM25 indices in sync.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.embedding.embedding_manager import EmbeddingManager
from src.embedding.index.bm25_index import BM25Index
from src.embedding.index.faiss_index import FaissIndex

logger = logging.getLogger(__name__)


class HybridIndex:
    """Wrapper that maintains synchronized FAISS + BM25 indices."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        indices_dir: str = "data/indices",
    ):
        self.embedding_manager = embedding_manager
        self.faiss_index = FaissIndex(dimension=embedding_manager.get_dimension())
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)
        self.indices_dir = Path(indices_dir)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_map: Dict[str, dict] = {}  # chunk_id -> chunk data

    def build(
        self,
        chunks: List[dict],
        chunk_strategy: str = "adaptive",
        chunk_size: int = 500,
        force_reembed: bool = False,
    ):
        """Build both indices from chunks.

        Args:
            chunks: List of chunk dicts (must have 'chunk_id' and 'text')
            chunk_strategy: Name of chunking strategy
            chunk_size: Chunk size config
            force_reembed: Force re-embedding even if cache exists
        """
        texts = [c["text"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        # Store chunk map for later retrieval
        self.chunk_map = {c["chunk_id"]: c for c in chunks}

        logger.info("Building hybrid index: %d chunks", len(chunks))

        # Build dense (FAISS) index
        logger.info("Building dense index...")
        embeddings, cached_ids = self.embedding_manager.embed_and_cache(
            texts, chunk_ids, chunk_strategy, chunk_size, force=force_reembed
        )
        self.faiss_index.build_index(embeddings, cached_ids)

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25_index.build_index(texts, chunk_ids)

        # Verify sync
        assert set(self.faiss_index.chunk_ids) == set(self.bm25_index.chunk_ids), (
            "FAISS and BM25 indices have different chunk IDs!"
        )
        logger.info(
            "Hybrid index built: %d chunks in both indices",
            len(chunk_ids),
        )

    def search_dense(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """Search using dense (FAISS) retrieval only."""
        query_embedding = self.embedding_manager.embed_query(query)
        ids, scores = self.faiss_index.search(
            np.array(query_embedding), top_k
        )
        return list(zip(ids, scores))

    def search_bm25(
        self, query: str, top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """Search using BM25 lexical retrieval only."""
        ids, scores = self.bm25_index.search(query, top_k)
        return list(zip(ids, scores))

    def search_hybrid(
        self,
        query: str,
        top_k: int = 20,
        fusion: str = "rrf",
        alpha: float = 0.5,
        top_k_candidates: int = 50,
        rrf_k: int = 60,
    ) -> List[Tuple[str, float]]:
        """Search using hybrid fusion of BM25 + dense.

        Args:
            query: Search query
            top_k: Number of final results
            fusion: Fusion method ('linear', 'rrf')
            alpha: Weight for BM25 in linear fusion (1-alpha for dense)
            top_k_candidates: Candidates to retrieve from each system
            rrf_k: Constant for RRF formula
        """
        # Get candidates from both systems
        bm25_results = self.search_bm25(query, top_k=top_k_candidates)
        dense_results = self.search_dense(query, top_k=top_k_candidates)

        if fusion == "linear":
            return self._linear_fusion(bm25_results, dense_results, alpha, top_k)
        elif fusion == "rrf":
            return self._rrf_fusion(bm25_results, dense_results, rrf_k, top_k)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

    def _linear_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        alpha: float,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Linear fusion: score = alpha * norm(bm25) + (1-alpha) * norm(dense)."""
        # Min-max normalize scores
        bm25_scores = self._normalize_scores(bm25_results)
        dense_scores = self._normalize_scores(dense_results)

        # Combine
        all_ids = set(s[0] for s in bm25_scores) | set(s[0] for s in dense_scores)
        bm25_dict = dict(bm25_scores)
        dense_dict = dict(dense_scores)

        combined = []
        for cid in all_ids:
            score = alpha * bm25_dict.get(cid, 0.0) + (1 - alpha) * dense_dict.get(cid, 0.0)
            combined.append((cid, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
        rrf_k: int,
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion: score = sum(1/(k + rank_i))."""
        rrf_scores: Dict[str, float] = {}

        for rank, (cid, _) in enumerate(bm25_results):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, (cid, _) in enumerate(dense_results):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        combined = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    @staticmethod
    def _normalize_scores(
        results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return results
        scores = [s for _, s in results]
        min_s = min(scores)
        max_s = max(scores)
        rng = max_s - min_s
        if rng == 0:
            return [(cid, 1.0) for cid, _ in results]
        return [(cid, (s - min_s) / rng) for cid, s in results]

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Retrieve chunk data by ID."""
        return self.chunk_map.get(chunk_id)

    # ----------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------

    def _index_prefix(self, strategy: str, size: int) -> str:
        model = self.embedding_manager.get_model_name()
        return f"{model}_{strategy}_{size}"

    def save(self, chunk_strategy: str = "adaptive", chunk_size: int = 500):
        """Save all indices to disk."""
        prefix = self._index_prefix(chunk_strategy, chunk_size)
        self.faiss_index.save(
            str(self.indices_dir / f"faiss_{prefix}.index")
        )
        self.bm25_index.save(
            str(self.indices_dir / f"bm25_{chunk_strategy}_{chunk_size}.pkl")
        )
        # Save chunk map
        map_path = self.indices_dir / f"chunk_map_{prefix}.json"
        map_path.write_text(
            json.dumps(self.chunk_map, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Saved hybrid index: %s", prefix)

    def load(self, chunk_strategy: str = "adaptive", chunk_size: int = 500):
        """Load all indices from disk."""
        prefix = self._index_prefix(chunk_strategy, chunk_size)
        self.faiss_index.load(
            str(self.indices_dir / f"faiss_{prefix}.index")
        )
        self.bm25_index.load(
            str(self.indices_dir / f"bm25_{chunk_strategy}_{chunk_size}.pkl")
        )
        map_path = self.indices_dir / f"chunk_map_{prefix}.json"
        if map_path.exists():
            self.chunk_map = json.loads(map_path.read_text(encoding="utf-8"))
        logger.info("Loaded hybrid index: %s", prefix)

    def get_stats(self) -> Dict:
        return {
            "faiss": self.faiss_index.get_stats(),
            "bm25": self.bm25_index.get_stats(),
            "chunk_map_size": len(self.chunk_map),
        }
