"""Canonical historical v2 classifier, moved unchanged from the metrics script."""
import logging

logger = logging.getLogger(__name__)
CLASSIFIER_VERSION = "faithfulness_v2_28_patterns_300_chars"

# ---------------------------------------------------------------------------
# v2 decline classifier (analysis layer; ledger N5)
# ---------------------------------------------------------------------------
# Canonical runtime patterns (the 14 in response_formatter.DECLINE_PATTERNS)
# are imported lazily in classify_response so this module stays importable
# without src/ on path for legacy experiments.
#
# Extended markers: refusal variants observed in exp12 answers that the
# canonical list misses (validated 2026-06-11 against the 16 configs; they
# drive the v1 false-negative rate of ~18 % on mistral's "answered" rows).
EXTENDED_REFUSAL_PATTERNS = [
    r"there is no (information|mention|specific|detail)",
    r"does not (mention|cover|provide|contain|include|specify|describe|address)",
    r"do not (mention|cover|provide|contain|include|specify)",
    r"is not (covered|mentioned|available|present|described|addressed|included|provided|specified)",
    r"are not (covered|mentioned|available|described|included)",
    r"not (covered|mentioned|available|specified|detailed|explicitly stated) in",
    r"insufficient (information|context|detail)",
    r"not enough (information|detail|context)",
    r"i (was|am) (unable|not able) to (find|locate|provide)",
    r"could ?n[o']t find",
    r"cannot (provide|answer|determine|confirm|be determined)",
    r"no (specific |further |additional )?(information|details?|mention)\b",
    r"beyond the (scope|provided)",
    r"outside the (scope|provided)",
]

# Chars of the lowercased answer inspected for the PURE-decline call: a
# refusal marker inside this window means the model led with a refusal.
OPENING_WINDOW = 300

_REFUSAL_MARKERS = None  # canonical + extended, compiled lazily


def _refusal_markers():
    global _REFUSAL_MARKERS
    if _REFUSAL_MARKERS is None:
        import re
        try:
            from src.generation.response_formatter import DECLINE_PATTERNS
        except Exception:  # legacy runs without src on path
            DECLINE_PATTERNS = []
            logger.warning("response_formatter unavailable; v2 classifier uses extended patterns only")
        _REFUSAL_MARKERS = [re.compile(p) for p in
                            list(DECLINE_PATTERNS) + EXTENDED_REFUSAL_PATTERNS]
    return _REFUSAL_MARKERS


# The wire value stays `pure_decline` because it is serialised inside SIGNED evidence
# (exp12_matrix/faithfulness_metrics_v2..v4*.json, exp13_expansion, exp14_h5_replicas) and
# verify_v4_offline.py re-derives numbers from those files. Renaming it would desync code
# from evidence. What IS fixed here is the human-facing label, because the name lies.
DISPLAY_LABELS = {
    "pure_decline": "decline_prefix",
    "hedged_partial": "hedged_partial",
    "answered": "answered",
}


def classify_response(answer: str):
    """'pure_decline' | 'hedged_partial' | 'answered' (None for empty text).

    `pure_decline` means ONLY "a refusal marker appears in the first OPENING_WINDOW chars".
    It does NOT mean the model refused. Measured on exp18's baseline (defect #7, ledger
    entry 22): 84 of 89 rows so labelled go on to assert genuine claims (mean 5.6), and they
    carry the WORST unsupportable rate of the three classes (0.471 vs 0.330 for `answered`).
    The dominant shape is "I cannot find sufficient information ... However, I can outline
    general steps that are typically involved" -- a parametric-knowledge answer wearing a
    decline prefix. The rate of answers that truly assert nothing is 5/194 = 2.6 %.

    Read it as `decline_prefix` (see DISPLAY_LABELS) and pair it with `asserts_content`
    whenever a decision depends on whether the model actually said something.
    """
    if not answer or not answer.strip():
        return None
    low = answer.lower()
    opening = low[:OPENING_WINDOW]
    if any(rx.search(opening) for rx in _refusal_markers()):
        return "pure_decline"
    if any(rx.search(low) for rx in _refusal_markers()):
        return "hedged_partial"
    return "answered"


def asserts_content(genuine_claims) -> bool:
    """Did the answer actually assert something, regardless of how it opened?

    The content-based counterpart to classify_response, which only reads the prefix.
    `genuine_claims` is the row's genuine-claim count (total minus format artifacts).
    """
    return bool(genuine_claims)
