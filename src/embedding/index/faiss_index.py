"""
FAISS Index for dense vector search.
Supports exact (FlatIP) and approximate (IVFFlat) search.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS-based dense vector index."""

    def __init__(self, dimension: int = 1024, use_gpu: bool = False):
        self.dimension = dimension
        self.use_gpu = use_gpu
        self.index: Optional[faiss.Index] = None
        self.chunk_ids: List[str] = []
        self.index_type: str = ""

    def build_index(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
        nlist: int = 100,
        nprobe: int = 10,
    ) -> "FaissIndex":
        """Build FAISS index from embeddings.

        Uses FlatIP for <50K vectors, IVFFlat for larger sets.
        Embeddings should already be L2-normalized for cosine similarity via IP.
        """
        assert embeddings.shape[0] == len(chunk_ids), (
            f"Mismatch: {embeddings.shape[0]} embeddings vs {len(chunk_ids)} IDs"
        )
        assert embeddings.shape[1] == self.dimension, (
            f"Dimension mismatch: got {embeddings.shape[1]}, expected {self.dimension}"
        )

        n = embeddings.shape[0]
        embeddings = embeddings.astype(np.float32)

        # Ensure L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        start = time.time()

        if n < 50000:
            # Exact inner product search
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index_type = "FlatIP"
        else:
            # Approximate search with IVF
            nlist = min(nlist, n // 10)  # nlist should be << n
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.nprobe = nprobe
            self.index.train(embeddings)
            self.index_type = f"IVFFlat(nlist={nlist}, nprobe={nprobe})"

        self.index.add(embeddings)
        self.chunk_ids = list(chunk_ids)

        elapsed = time.time() - start
        logger.info(
            "Built FAISS index: type=%s, vectors=%d, dim=%d, time=%.1fs",
            self.index_type, n, self.dimension, elapsed,
        )
        return self

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 50,
    ) -> Tuple[List[str], List[float]]:
        """Search for nearest neighbors.

        Returns:
            Tuple of (chunk_ids, scores)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        result_ids = []
        result_scores = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunk_ids):
                result_ids.append(self.chunk_ids[idx])
                result_scores.append(float(score))

        return result_ids, result_scores

    def save(self, path: str):
        """Save index and mapping to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))

        mapping_path = path.with_suffix(".mapping.json")
        mapping = {
            "chunk_ids": self.chunk_ids,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "total_vectors": len(self.chunk_ids),
        }
        mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
        logger.info("Saved FAISS index to %s (%d vectors)", path, len(self.chunk_ids))

    def load(self, path: str) -> "FaissIndex":
        """Load index and mapping from disk."""
        path = Path(path)
        self.index = faiss.read_index(str(path))

        mapping_path = path.with_suffix(".mapping.json")
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        self.chunk_ids = mapping["chunk_ids"]
        self.dimension = mapping["dimension"]
        self.index_type = mapping.get("index_type", "unknown")

        logger.info("Loaded FAISS index from %s (%d vectors)", path, len(self.chunk_ids))
        return self

    def get_stats(self) -> Dict:
        return {
            "type": self.index_type,
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
        }
