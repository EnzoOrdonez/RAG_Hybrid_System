"""
Unit tests for `decide_nli_status` — the single rule every faithfulness number rests on.

It is shared byte-for-byte by the runtime detector and every offline re-score, so a
change here silently rewrites Tier A, exp16, exp17 and the whole v4 table at once. It had
no direct unit coverage: test_nli_calibration.py exercises the MODEL (are the scores
probabilities?), never the DECISION (what do we do with them?).

The asymmetry it encodes is deliberate and is the subject of ledger N8 / Tier 3:
  - `supported` carries a guard, `max_ent > max_contr`, on every variant.
  - `contradicted` under the legacy v0 has NO symmetric guard, which is why the
    instrument marked 22% of RANDOM text as contradicted (negative control) and masked
    the granite hybrid>lexical effect that HHEM reveals.
  - `vb_agree` (the variant used everywhere in the summer phase) adds one: at least TWO
    chunks must exceed the threshold before a claim is called contradicted.

Run: pytest tests/test_decide_nli_status.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.hallucination_detector import decide_nli_status  # noqa: E402

T = 0.7


def status(contr, ent, variant="vb_agree", margin=0.0):
    return decide_nli_status(contr, ent, T, T, variant=variant, margin=margin)[0]


# ------------------------------------------------------------------ supported
def test_supported_needs_entailment_over_threshold():
    assert status([0.1] * 5, [0.9, 0.1, 0.1, 0.1, 0.1]) == "supported"
    assert status([0.1] * 5, [0.69, 0.1, 0.1, 0.1, 0.1]) != "supported"


def test_supported_is_strict_not_inclusive_at_the_threshold():
    """`>` not `>=`: exactly 0.7 does NOT support."""
    assert status([0.1] * 5, [0.70, 0.1, 0.1, 0.1, 0.1]) != "supported"
    assert status([0.1] * 5, [0.7000001, 0.1, 0.1, 0.1, 0.1]) == "supported"


def test_supported_requires_beating_contradiction_on_every_variant():
    """The asymmetric guard: high entailment loses to higher contradiction."""
    for v in ("v0", "va_margin", "vb_agree"):
        assert status([0.95, 0.95, 0.1, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1, 0.1],
                      variant=v) != "supported", v


def test_supported_wins_over_contradiction_when_it_is_stronger():
    """Evaluated first: a claim entailed by one chunk and contradicted by two is
    supported, because the passages disagree and entailment is the stronger signal."""
    assert status([0.8, 0.8, 0.1, 0.1, 0.1], [0.95, 0.1, 0.1, 0.1, 0.1]) == "supported"


# -------------------------------------------------------------- vb_agree gate
def test_vb_agree_needs_two_chunks_over_the_threshold():
    """One lone contradicting chunk is not enough — the guard that defines the variant."""
    one = [0.95, 0.1, 0.1, 0.1, 0.1]
    assert status(one, [0.1] * 5, variant="vb_agree") == "unsupported"
    assert status(one, [0.1] * 5, variant="v0") == "contradicted"


def test_vb_agree_fires_with_exactly_two():
    assert status([0.95, 0.75, 0.1, 0.1, 0.1], [0.1] * 5, variant="vb_agree") == "contradicted"


def test_vb_agree_counts_strictly_above_the_threshold():
    """A second chunk sitting exactly at 0.7 does not count as agreement."""
    assert status([0.95, 0.70, 0.1, 0.1, 0.1], [0.1] * 5, variant="vb_agree") == "unsupported"


def test_vb_agree_is_stricter_than_v0_never_looser():
    """Property: anything vb_agree calls contradicted, v0 calls contradicted too."""
    cases = [
        ([0.95, 0.1, 0.1], [0.1, 0.1, 0.1]),
        ([0.95, 0.8, 0.1], [0.1, 0.1, 0.1]),
        ([0.71, 0.71, 0.71], [0.2, 0.2, 0.2]),
        ([0.4, 0.4, 0.4], [0.5, 0.5, 0.5]),
        ([0.9, 0.9, 0.9], [0.95, 0.1, 0.1]),
    ]
    for contr, ent in cases:
        if status(contr, ent, variant="vb_agree") == "contradicted":
            assert status(contr, ent, variant="v0") == "contradicted", (contr, ent)


# ------------------------------------------------------------ va_margin gate
def test_va_margin_requires_beating_entailment_by_the_margin():
    contr, ent = [0.8, 0.1, 0.1], [0.75, 0.1, 0.1]
    assert status(contr, ent, variant="va_margin", margin=0.0) == "contradicted"
    assert status(contr, ent, variant="va_margin", margin=0.2) == "unsupported"


# ----------------------------------------------------------------- edge cases
def test_empty_scores_are_unsupported_not_a_crash():
    """A response with no evidence chunks must not raise; it has nothing to ground on."""
    assert decide_nli_status([], [], T, T, variant="vb_agree") == ("unsupported", 0.0, -1)


def test_returned_index_points_at_the_chunk_that_decided():
    st, score, idx = decide_nli_status([0.1, 0.1, 0.2], [0.1, 0.92, 0.3], T, T,
                                       variant="vb_agree")
    assert (st, idx) == ("supported", 1) and score == pytest.approx(0.92)

    st, score, idx = decide_nli_status([0.1, 0.93, 0.85], [0.1, 0.1, 0.1], T, T,
                                       variant="vb_agree")
    assert (st, idx) == ("contradicted", 1) and score == pytest.approx(0.93)


def test_unsupported_surfaces_the_strongest_signal():
    """Neither gate fires: report whichever side is larger, for downstream triage."""
    st, score, idx = decide_nli_status([0.2, 0.3], [0.5, 0.4], T, T, variant="vb_agree")
    assert st == "unsupported" and score == pytest.approx(0.5) and idx == 0

    st, score, idx = decide_nli_status([0.5, 0.4], [0.2, 0.3], T, T, variant="vb_agree")
    assert st == "unsupported" and score == pytest.approx(0.5) and idx == 0


def test_every_variant_returns_one_of_the_three_labels():
    import itertools
    vals = [0.0, 0.5, 0.71, 1.0]
    labels = {"supported", "contradicted", "unsupported"}
    for variant in ("v0", "va_margin", "vb_agree"):
        for contr in itertools.product(vals, repeat=2):
            for ent in itertools.product(vals, repeat=2):
                assert status(list(contr), list(ent), variant=variant) in labels


def test_thresholds_are_honoured_when_not_the_default():
    """The sweep re-runs this rule at other operating points (exp15 Tier 0)."""
    contr, ent = [0.1, 0.1], [0.55, 0.1]
    assert decide_nli_status(contr, ent, 0.5, 0.5, variant="vb_agree")[0] == "supported"
    assert decide_nli_status(contr, ent, 0.7, 0.7, variant="vb_agree")[0] == "unsupported"
