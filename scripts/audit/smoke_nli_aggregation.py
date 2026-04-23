"""
Smoke test for Flag 138 fix in src/generation/hallucination_detector.py.

Validates that _nli_matching aggregates per claim using the
max-per-class rule (Honovich 2022 TRUE), not the previous
first-supported-wins short-circuit. Runs WITHOUT network access by
monkey-patching HallucinationDetector.nli_model with a scripted stub
that returns pre-canned scores.

Covers:
  1. Mixed signal: chunk A supports (ent=0.80), chunk B contradicts
     (contr=0.90). Must return "contradicted". Old code returned
     "supported" because of the `best_status != "supported"` guard.
  2. Consistent support: chunk A supports (ent=0.80), chunk B neutral.
     Must return "supported".
  3. Below threshold: all chunks below 0.7. Must return "unsupported".
  4. Tie between entailment and contradiction slightly favoring
     contradiction: max_ent=0.72, max_contr=0.80. Contradiction wins.

Run with:
    python scripts/audit/smoke_nli_aggregation.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402


class _ScriptedNLI:
    """Stand-in for sentence_transformers.CrossEncoder with canned output."""

    def __init__(self, scores_per_pair):
        # scores_per_pair: list of [contradiction, entailment, neutral]
        self._scores = scores_per_pair

    def predict(self, pairs, batch_size=32, show_progress_bar=False, apply_softmax=True):
        if len(pairs) != len(self._scores):
            raise AssertionError(
                f"stub expected {len(self._scores)} pairs, got {len(pairs)}"
            )
        return self._scores


def _run_case(name, scored_chunks, expected_status, expected_chunk_id):
    """scored_chunks: list of (chunk_id, [contr, ent, neutral]).

    Builds a detector with a scripted NLI and asserts the resulting
    ClaimDetail has the expected status + evidence chunk.
    """
    chunk_ids = [cid for cid, _ in scored_chunks]
    chunk_texts = [f"text for {cid}" for cid in chunk_ids]
    scores = [row for _, row in scored_chunks]

    det = HallucinationDetector(use_nli=True)
    # Bypass lazy loader with the scripted stub.
    det._nli_model = _ScriptedNLI(scores)
    det._nli_available = True

    details = det._nli_matching(
        claims=["the test claim"],
        chunk_texts=chunk_texts,
        chunk_ids=chunk_ids,
    )
    assert len(details) == 1
    d = details[0]
    assert d.status == expected_status, (
        f"{name}: status={d.status!r} expected={expected_status!r}"
    )
    assert d.evidence_chunk_id == expected_chunk_id, (
        f"{name}: evidence_chunk_id={d.evidence_chunk_id!r} "
        f"expected={expected_chunk_id!r}"
    )
    print(
        f"PASS: {name} — status={d.status}, "
        f"evidence={d.evidence_chunk_id}, nli_score={d.nli_score:.3f}"
    )


def main() -> None:
    # Case 1: chunk A supports (ent=0.80), chunk B contradicts (contr=0.90).
    # The OLD implementation returned "supported" on the first pass and
    # then ignored the later contradiction. New implementation must pick
    # contradiction because max_contr (0.90) > max_ent (0.80) AND
    # max_contr > threshold.
    _run_case(
        name="mixed signal: A supports, B contradicts -> contradicted",
        scored_chunks=[
            ("chunk_a", [0.10, 0.80, 0.10]),   # entailment strong
            ("chunk_b", [0.90, 0.05, 0.05]),   # contradiction stronger
        ],
        expected_status="contradicted",
        expected_chunk_id="chunk_b",
    )

    # Case 2: chunk A supports strongly, chunk B is neutral. Supported.
    _run_case(
        name="consistent support -> supported",
        scored_chunks=[
            ("chunk_a", [0.05, 0.80, 0.15]),
            ("chunk_b", [0.05, 0.30, 0.65]),
        ],
        expected_status="supported",
        expected_chunk_id="chunk_a",
    )

    # Case 3: everything below 0.7 threshold in both classes. Unsupported.
    _run_case(
        name="below-threshold entail and contr -> unsupported",
        scored_chunks=[
            ("chunk_a", [0.20, 0.60, 0.20]),
            ("chunk_b", [0.50, 0.10, 0.40]),
        ],
        expected_status="unsupported",
        # unsupported still records the chunk of the strongest signal
        expected_chunk_id="chunk_a",  # ent=0.60 > contr=0.50
    )

    # Case 4: both cross threshold but contradiction is higher.
    # Honovich rule requires max_ent > max_contr for "supported"; here
    # max_ent=0.72 < max_contr=0.80 so "contradicted" wins even though
    # entailment also crosses its threshold.
    _run_case(
        name="both cross threshold, contr higher -> contradicted",
        scored_chunks=[
            ("chunk_a", [0.20, 0.72, 0.08]),
            ("chunk_b", [0.80, 0.10, 0.10]),
        ],
        expected_status="contradicted",
        expected_chunk_id="chunk_b",
    )

    # Case 5: both cross threshold but entailment is higher. Supported.
    _run_case(
        name="both cross threshold, ent higher -> supported",
        scored_chunks=[
            ("chunk_a", [0.71, 0.10, 0.19]),
            ("chunk_b", [0.10, 0.85, 0.05]),
        ],
        expected_status="supported",
        expected_chunk_id="chunk_b",
    )

    print("\nAll smoke cases passed for Flag 138 aggregation rewrite.")


if __name__ == "__main__":
    main()
