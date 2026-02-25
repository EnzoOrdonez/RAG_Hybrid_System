"""
Generation Metrics for thesis evaluation.

Metrics: Exact Match, F1 Token Overlap, ROUGE-L, BERTScore.

Error handling:
  - Empty texts -> return 0.0
  - BERTScore model unavailable -> fallback to F1 token with warning
  - Texts > 2000 tokens -> truncate before computing
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

MAX_TOKENS = 2000


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\w+", text.lower())


def _truncate(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Truncate text to max_tokens words."""
    tokens = text.split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return text


def exact_match(predicted: str, ground_truth: str) -> float:
    """1.0 if predicted contains the key information from ground_truth, else 0.0.

    Uses normalized string comparison: lowercased, stripped, key phrases.
    """
    if not predicted or not ground_truth:
        return 0.0
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # Exact string match
    if pred_lower == gt_lower:
        return 1.0

    # Check if ground truth key phrases appear in predicted
    gt_tokens = set(_tokenize(gt_lower))
    pred_tokens = set(_tokenize(pred_lower))

    if not gt_tokens:
        return 0.0

    # If 80%+ of ground truth tokens appear in predicted, count as match
    overlap = len(gt_tokens & pred_tokens) / len(gt_tokens)
    return 1.0 if overlap >= 0.8 else 0.0


def f1_token(predicted: str, ground_truth: str) -> float:
    """F1 score of token overlap between predicted and ground_truth."""
    if not predicted or not ground_truth:
        return 0.0

    predicted = _truncate(predicted)
    ground_truth = _truncate(ground_truth)

    pred_tokens = _tokenize(predicted)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)

    common = pred_set & gt_set
    if not common:
        return 0.0

    precision = len(common) / len(pred_set)
    recall = len(common) / len(gt_set)
    return 2 * precision * recall / (precision + recall)


def rouge_l(predicted: str, ground_truth: str) -> float:
    """ROUGE-L: Longest Common Subsequence ratio."""
    if not predicted or not ground_truth:
        return 0.0

    predicted = _truncate(predicted)
    ground_truth = _truncate(ground_truth)

    pred_tokens = _tokenize(predicted)
    gt_tokens = _tokenize(ground_truth)

    if not pred_tokens or not gt_tokens:
        return 0.0

    # LCS using dynamic programming
    m, n = len(pred_tokens), len(gt_tokens)
    # Limit to avoid memory issues
    if m > 500:
        pred_tokens = pred_tokens[:500]
        m = 500
    if n > 500:
        gt_tokens = gt_tokens[:500]
        n = 500

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def bert_score_metric(predicted: str, ground_truth: str) -> float:
    """BERTScore F1. Falls back to f1_token if model unavailable."""
    if not predicted or not ground_truth:
        return 0.0

    predicted = _truncate(predicted)
    ground_truth = _truncate(ground_truth)

    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            [predicted],
            [ground_truth],
            lang="en",
            verbose=False,
            rescale_with_baseline=True,
        )
        return float(F1[0])
    except ImportError:
        logger.warning("bert_score not installed, falling back to f1_token")
        return f1_token(predicted, ground_truth)
    except Exception as e:
        logger.warning("BERTScore failed (%s), falling back to f1_token", e)
        return f1_token(predicted, ground_truth)


def compute_all_generation_metrics(
    predicted: str, ground_truth: str, use_bert_score: bool = False
) -> Dict[str, float]:
    """Compute all generation metrics."""
    results = {
        "exact_match": exact_match(predicted, ground_truth),
        "f1_token": f1_token(predicted, ground_truth),
        "rouge_l": rouge_l(predicted, ground_truth),
    }
    if use_bert_score:
        results["bert_score"] = bert_score_metric(predicted, ground_truth)
    return results
