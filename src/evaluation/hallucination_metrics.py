"""
Hallucination Metrics for thesis evaluation.

Uses the HallucinationDetector from src/generation/hallucination_detector.py.
Includes: faithfulness, hallucination_rate, suggested_rubric.

RAGAS metrics are optional (if ragas library is installed).
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def compute_hallucination_metrics(
    response_text: str,
    retrieved_chunks: List[dict],
    detector=None,
) -> Dict[str, float]:
    """
    Compute hallucination metrics using the NLI-based detector.

    Args:
        response_text: Generated LLM response
        retrieved_chunks: List of chunk dicts used as evidence
        detector: Optional pre-loaded HallucinationDetector

    Returns:
        Dict with faithfulness, hallucination_rate, rubric, etc.
    """
    if not response_text or not retrieved_chunks:
        return {
            "faithfulness": 0.0,
            "hallucination_rate": 1.0,
            "suggested_rubric": 1,
            "total_claims": 0,
            "supported_claims": 0,
            "method": "none",
        }

    try:
        if detector is None:
            from src.generation.hallucination_detector import HallucinationDetector
            detector = HallucinationDetector()

        report = detector.check(response_text, retrieved_chunks)

        return {
            "faithfulness": report.faithfulness_score,
            "hallucination_rate": report.hallucination_rate,
            "suggested_rubric": report.suggested_rubric,
            "total_claims": report.total_claims,
            "supported_claims": report.supported_claims,
            "contradicted_claims": report.contradicted_claims,
            "unsupported_claims": report.unsupported_claims,
            "method": report.method,
            "processing_time_ms": report.processing_time_ms,
        }
    except Exception as e:
        logger.warning("Hallucination detection failed: %s", e)
        return {
            "faithfulness": 0.0,
            "hallucination_rate": 1.0,
            "suggested_rubric": 1,
            "total_claims": 0,
            "supported_claims": 0,
            "method": "error",
            "error": str(e),
        }


def compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute RAGAS metrics if the ragas library is available.

    Returns empty dict if ragas is not installed.
    """
    try:
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas import evaluate
        from datasets import Dataset

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            data["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data)
        metrics_to_eval = [faithfulness, answer_relevancy]
        if ground_truth:
            metrics_to_eval.extend([context_precision, context_recall])

        result = evaluate(dataset, metrics=metrics_to_eval)
        return dict(result)

    except ImportError:
        logger.debug("ragas not installed, skipping RAGAS metrics")
        return {}
    except Exception as e:
        logger.warning("RAGAS evaluation failed: %s", e)
        return {}
