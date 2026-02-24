"""
Unified Embedding Manager.
Supports multiple embedding models with GPU acceleration,
batch processing, and disk caching.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "full_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_seq_length": 256,
        "query_prefix": "",
        "document_prefix": "",
        "description": "Lightweight baseline, fast inference",
    },
    "bge-large": {
        "full_name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "max_seq_length": 512,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
        "description": "State-of-the-art, primary candidate",
    },
    "e5-large": {
        "full_name": "intfloat/e5-large-v2",
        "dimension": 1024,
        "max_seq_length": 512,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "description": "Competitive alternative to BGE",
    },
    "instructor-large": {
        "full_name": "hkunlp/instructor-large",
        "dimension": 768,
        "max_seq_length": 512,
        "query_prefix": "",
        "document_prefix": "",
        "query_instruction": "Represent the cloud computing question for retrieving relevant documentation: ",
        "document_instruction": "Represent the cloud computing documentation for retrieval: ",
        "description": "Task-specific with custom instructions",
    },
}


class EmbeddingManager:
    """Manages embedding models with caching and GPU support."""

    def __init__(
        self,
        model_name: str = "bge-large",
        cache_dir: str = "data/embeddings",
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(MODEL_CONFIGS.keys())}"
            )

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(
            "EmbeddingManager: model=%s, device=%s, dim=%d",
            model_name, self.device, self.config["dimension"],
        )

        self._model = None

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load the sentence-transformer model."""
        from sentence_transformers import SentenceTransformer

        full_name = self.config["full_name"]
        logger.info("Loading embedding model: %s ...", full_name)
        start = time.time()

        if self.model_name == "instructor-large":
            try:
                from InstructorEmbedding import INSTRUCTOR
                model = INSTRUCTOR(full_name, device=self.device)
            except ImportError:
                logger.warning(
                    "InstructorEmbedding not installed, using SentenceTransformer fallback"
                )
                model = SentenceTransformer(full_name, device=self.device)
        else:
            model = SentenceTransformer(full_name, device=self.device)

        elapsed = time.time() - start
        logger.info("Model loaded in %.1fs on %s", elapsed, self.device)
        return model

    def get_dimension(self) -> int:
        return self.config["dimension"]

    def get_model_name(self) -> str:
        return self.model_name

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with appropriate prefix."""
        prefix = self.config.get("query_prefix", "")
        if self.model_name == "instructor-large":
            instruction = self.config.get("query_instruction", "")
            prefixed = instruction + text if instruction else text
        else:
            prefixed = prefix + text if prefix else text

        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_documents(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed multiple documents with batching and progress bar."""
        prefix = self.config.get("document_prefix", "")
        if self.model_name == "instructor-large":
            instruction = self.config.get("document_instruction", "")
            prefixed = [instruction + t if instruction else t for t in texts]
        elif prefix:
            prefixed = [prefix + t for t in texts]
        else:
            prefixed = texts

        all_embeddings = []
        total_batches = (len(prefixed) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(prefixed), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc=f"Embedding ({self.model_name})",
                unit="batch",
            )

        for i in iterator:
            batch = prefixed[i : i + self.batch_size]
            # Use smaller encode batch for GPU memory efficiency
            encode_batch = 32 if self.device == "cuda" else self.batch_size
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=min(len(batch), encode_batch),
            )
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.info(
            "Embedded %d documents -> shape %s", len(texts), embeddings.shape
        )
        return embeddings

    # ----------------------------------------------------------
    # Caching
    # ----------------------------------------------------------

    def get_cache_path(
        self, chunk_strategy: str, chunk_size: int
    ) -> Path:
        """Get the cache file path for embeddings."""
        return self.cache_dir / f"{self.model_name}_{chunk_strategy}_{chunk_size}.npy"

    def get_ids_cache_path(
        self, chunk_strategy: str, chunk_size: int
    ) -> Path:
        """Get the cache file path for chunk IDs."""
        return self.cache_dir / f"{self.model_name}_{chunk_strategy}_{chunk_size}_ids.npy"

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[str],
        chunk_strategy: str,
        chunk_size: int,
    ):
        """Save embeddings and chunk IDs to disk."""
        emb_path = self.get_cache_path(chunk_strategy, chunk_size)
        ids_path = self.get_ids_cache_path(chunk_strategy, chunk_size)
        np.save(emb_path, embeddings)
        np.save(ids_path, np.array(chunk_ids))
        logger.info(
            "Saved embeddings (%s) to %s", embeddings.shape, emb_path
        )

    def load_embeddings(
        self, chunk_strategy: str, chunk_size: int
    ) -> Optional[Tuple[np.ndarray, List[str]]]:
        """Load cached embeddings from disk if available."""
        emb_path = self.get_cache_path(chunk_strategy, chunk_size)
        ids_path = self.get_ids_cache_path(chunk_strategy, chunk_size)
        if emb_path.exists() and ids_path.exists():
            embeddings = np.load(emb_path)
            chunk_ids = np.load(ids_path).tolist()
            logger.info(
                "Loaded cached embeddings (%s) from %s",
                embeddings.shape, emb_path,
            )
            return embeddings, chunk_ids
        return None

    def embed_and_cache(
        self,
        texts: List[str],
        chunk_ids: List[str],
        chunk_strategy: str,
        chunk_size: int,
        force: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """Embed documents, using cache if available."""
        if not force:
            cached = self.load_embeddings(chunk_strategy, chunk_size)
            if cached is not None:
                return cached

        start = time.time()
        embeddings = self.embed_documents(texts)
        elapsed = time.time() - start
        logger.info(
            "Embedding took %.1fs (%.0f docs/sec)",
            elapsed, len(texts) / elapsed if elapsed > 0 else 0,
        )

        self.save_embeddings(embeddings, chunk_ids, chunk_strategy, chunk_size)
        return embeddings, chunk_ids

    def get_stats(self) -> Dict:
        """Return stats about cached embeddings."""
        stats = {"model": self.model_name, "dimension": self.get_dimension(), "cached": []}
        for npy_file in self.cache_dir.glob(f"{self.model_name}_*.npy"):
            if "_ids" in npy_file.name:
                continue
            size_mb = npy_file.stat().st_size / (1024 * 1024)
            emb = np.load(npy_file)
            stats["cached"].append({
                "file": npy_file.name,
                "shape": list(emb.shape),
                "size_mb": round(size_mb, 2),
            })
        return stats
