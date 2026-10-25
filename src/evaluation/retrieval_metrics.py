"""
Retrieval Metrics for thesis evaluation.

Metrics: Recall@K, Precision@K, MRR, NDCG@K, MAP.
Evaluated at K = 5, 10, 20.

All functions receive:
  - retrieved_ids: List[str] -- chunk IDs retrieved (ordered by relevance)
  - relevant_ids: List[str] -- ground truth chunk IDs
  - k: int -- cutoff

Return float between 0.0 and 1.0.
"""

import logging
import math
from typing import Dict, List

logger = logging.getLogger(__name__)


def _validate_ids(retrieved_ids, relevant_ids, k=None):
    """Reject ambiguous ranks/denominators; preserve historical valid-input values."""
    if len(set(retrieved_ids)) != len(retrieved_ids) or len(set(relevant_ids)) != len(relevant_ids):
        raise ValueError("Retrieval metrics require unique retrieved and relevant IDs")
    if k is not None and (type(k) is not int or k <= 0):
        raise ValueError("k must be a positive integer")


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Fraction of relevant documents found in top-k results."""
    _validate_ids(retrieved_ids, relevant_ids, k)
    if not relevant_ids:
        logger.warning("recall_at_k: empty relevant_ids")
        return 0.0
    if not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = len(set(top_k) & set(relevant_ids))
    return found / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Relevant / returned results up to k (historical denominator: min(k, n))."""
    _validate_ids(retrieved_ids, relevant_ids, k)
    if not relevant_ids:
        logger.warning("precision_at_k: empty relevant_ids")
        return 0.0
    if not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    found = len(set(top_k) & set(relevant_ids))
    return found / len(top_k) if top_k else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    _validate_ids(retrieved_ids, relevant_ids)
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    _validate_ids(retrieved_ids, relevant_ids, k)
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # DCG
    dcg = 0.0
    for i, rid in enumerate(top_k):
        rel = 1.0 if rid in relevant_set else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG (all relevant docs at the top)
    ideal_k = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def map_score(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean Average Precision (for a single query = Average Precision)."""
    _validate_ids(retrieved_ids, relevant_ids)
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant_ids) if relevant_ids else 0.0


def compute_all_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """Compute all retrieval metrics for all K values."""
    if k_values is None:
        k_values = [5, 10, 20]

    results = {}
    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ids, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
    results["mrr"] = mrr(retrieved_ids, relevant_ids)
    results["map"] = map_score(retrieved_ids, relevant_ids)
    return results
