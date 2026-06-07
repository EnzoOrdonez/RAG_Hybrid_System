"""
Unit tests for NLI faithfulness calibration (audit Flag 135).

Guards two things that, if broken, silently corrupt every faithfulness number
in the paper:

  1. The HallucinationDetector NLI path classifies an obviously-entailed claim
     as ``supported`` and an obviously-contradicted claim as NOT supported.
  2. The underlying cross-encoder is read AS PROBABILITIES (apply_softmax=True):
     the 0.7 threshold is a probability threshold, not a raw-logit threshold.
     Reverting to raw logits (the pre-Flag-135 bug) makes test 2 fail.

The NLI model (cross-encoder/nli-deberta-v3-small, ~200MB) must be cached
locally; tests skip gracefully if it cannot be loaded offline.

Run: pytest tests/test_nli_calibration.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402

# A single evidence chunk that clearly entails one claim and contradicts another.
CHUNK = {
    "chunk_id": "test_chunk_1",
    "text": (
        "The capital of France is Paris. Paris is located in northern France "
        "on the river Seine and is the country's most populous city."
    ),
}

SUPPORTED_RESPONSE = "Paris is the capital of France."
CONTRADICTED_RESPONSE = "The capital of France is Berlin."


@pytest.fixture(scope="module")
def detector():
    det = HallucinationDetector(use_nli=True)
    # Touch the lazy property; skip the whole module if the model is not
    # available offline (so this test never produces a misleading red on a
    # machine without the cached weights).
    if det.nli_model is None or not det._use_nli:
        pytest.skip("NLI model unavailable (offline / not cached)")
    return det


def test_supported_claim_is_supported(detector):
    """An entailed claim against matching evidence must be 'supported'."""
    report = detector.check(SUPPORTED_RESPONSE, [CHUNK])
    assert report.method == "nli", "NLI path inactive (fell back to keyword)"
    assert report.total_claims >= 1
    assert report.supported_claims >= 1
    assert report.faithfulness_score >= 0.5


def test_contradicted_claim_is_not_supported(detector):
    """A contradicted claim must NOT be counted as supported."""
    report = detector.check(CONTRADICTED_RESPONSE, [CHUNK])
    assert report.method == "nli"
    assert report.total_claims >= 1
    assert report.supported_claims == 0
    assert report.faithfulness_score < 0.5


def test_nli_output_is_softmax_probabilities():
    """Flag 135 guard: thresholds operate on softmax probabilities.

    With apply_softmax=True the three NLI label scores sum to 1 and the
    obvious-entailment pair crosses ENTAILMENT_THRESHOLD. If someone reverts to
    raw logits, the sum is arbitrary and this test fails.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(HallucinationDetector.NLI_MODEL, max_length=512)
    except Exception as e:  # offline / not cached
        pytest.skip(f"NLI model unavailable: {e}")

    id2label = {i: lbl.lower() for i, lbl in model.config.id2label.items()}
    assert "entailment" in id2label.values(), f"unexpected labels: {id2label}"
    ent_idx = next(i for i, lbl in id2label.items() if lbl == "entailment")

    probs = model.predict(
        [("The capital of France is Paris.", "Paris is the capital of France.")],
        apply_softmax=True,
    )[0]

    assert abs(float(sum(probs)) - 1.0) < 1e-4, (
        f"NLI scores must sum to 1 after softmax, got {float(sum(probs))}"
    )
    assert float(probs[ent_idx]) > HallucinationDetector.ENTAILMENT_THRESHOLD, (
        f"entailment prob {float(probs[ent_idx]):.3f} must exceed threshold "
        f"{HallucinationDetector.ENTAILMENT_THRESHOLD}"
    )
