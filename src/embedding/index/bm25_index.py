"""
BM25 Index for lexical search.
Custom stopword list that preserves cloud-domain acronyms.
"""

import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Standard English stopwords MINUS technical terms
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "it", "its", "this", "that", "these", "those", "i", "we", "you",
    "he", "she", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "because", "as", "until", "while",
    "about", "between", "through", "during", "before", "after", "above",
    "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "also", "into",
}

# Technical terms to NEVER remove as stopwords
KEEP_TERMS = {
    "api", "sdk", "cli", "vpc", "iam", "ec2", "s3", "rds", "ecs", "eks",
    "aks", "gke", "gce", "gcs", "acr", "ecr", "alb", "nlb", "elb",
    "kms", "cdn", "dns", "vm", "k8s", "cncf", "rbac", "cidr", "tls",
    "ssl", "rest", "grpc", "yaml", "json", "http", "https", "tcp", "udp",
    "ip", "cpu", "gpu", "ram", "ssd", "hdd", "os", "io",
    "lambda", "fargate", "bedrock", "sagemaker",
    "azure", "aws", "gcp", "google", "amazon", "microsoft",
}


class BM25Index:
    """BM25-based lexical search index."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.index: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.corpus_tokens: List[List[str]] = []

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25: lowercase, split, remove stopwords."""
        text = text.lower()
        # Split on non-alphanumeric, keep hyphens in compound terms
        tokens = re.findall(r'[a-z0-9][-a-z0-9]*[a-z0-9]|[a-z0-9]', text)
        # Remove stopwords but keep technical terms
        filtered = [
            t for t in tokens
            if t not in STOPWORDS or t in KEEP_TERMS
        ]
        return filtered

    def build_index(
        self,
        texts: List[str],
        chunk_ids: List[str],
    ) -> "BM25Index":
        """Build BM25 index from text documents."""
        assert len(texts) == len(chunk_ids)

        start = time.time()
        self.chunk_ids = list(chunk_ids)
        self.corpus_tokens = [self.tokenize(t) for t in texts]

        self.index = BM25Okapi(
            self.corpus_tokens, k1=self.k1, b=self.b
        )

        elapsed = time.time() - start
        avg_len = sum(len(t) for t in self.corpus_tokens) / max(1, len(self.corpus_tokens))
        logger.info(
            "Built BM25 index: docs=%d, avg_tokens=%.0f, time=%.1fs",
            len(texts), avg_len, elapsed,
        )
        return self

    def search(
        self, query: str, top_k: int = 50
    ) -> Tuple[List[str], List[float]]:
        """Search with BM25.

        Returns:
            Tuple of (chunk_ids, scores)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return [], []

        scores = self.index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = scores.argsort()[::-1][:top_k]

        result_ids = []
        result_scores = []
        for idx in top_indices:
            if scores[idx] > 0:
                result_ids.append(self.chunk_ids[idx])
                result_scores.append(float(scores[idx]))

        return result_ids, result_scores

    def save(self, path: str):
        """Save BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "k1": self.k1,
            "b": self.b,
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("Saved BM25 index to %s (%d docs)", path, len(self.chunk_ids))

    def load(self, path: str) -> "BM25Index":
        """Load BM25 index from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.k1 = data["k1"]
        self.b = data["b"]
        self.chunk_ids = data["chunk_ids"]
        self.corpus_tokens = data["corpus_tokens"]
        self.index = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)

        logger.info("Loaded BM25 index from %s (%d docs)", path, len(self.chunk_ids))
        return self

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.chunk_ids),
            "k1": self.k1,
            "b": self.b,
            "avg_doc_length": (
                sum(len(t) for t in self.corpus_tokens) / max(1, len(self.corpus_tokens))
                if self.corpus_tokens else 0
            ),
        }
