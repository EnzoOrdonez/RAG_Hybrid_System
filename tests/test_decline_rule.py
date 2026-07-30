"""
The guards and the faithfulness metric must share ONE definition of "declination".

They did not. `compute_exp16_guards.py` tested a single exact, case-sensitive substring
while `compute_faithfulness_metrics.py` ran `classify_response` over 28 case-insensitive
patterns. The two disagreed by 4-24 points on every arm of Tier A, exp16 and exp17, so
the declination rates quoted in the ledger were not the ones the metric saw. Both
verdicts survived the correction and got stronger, but a repo with two live definitions
of the same word is one edit away from a number that means nothing.

These tests pin the unification:
  1. the guards import the metric's own classifier, they do not re-implement it;
  2. `classify_response`'s three classes behave as documented, including the
     pure-vs-hedged boundary that carries the interpretation;
  3. the committed guards artifacts agree with the classifier.

Run: pytest tests/test_decline_rule.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GUARDS = PROJECT_ROOT / "scripts" / "compute_exp16_guards.py"
METRICS = PROJECT_ROOT / "scripts" / "compute_faithfulness_metrics.py"
RESULTS = PROJECT_ROOT / "experiments" / "results"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def cfm():
    return _load("cfm_t", METRICS)


@pytest.fixture(scope="module")
def guards():
    return _load("guards_t", GUARDS)


# ------------------------------------------------------------- one definition
def test_guards_use_the_metrics_classifier_object(guards, cfm):
    """Same behaviour, imported — not a second copy that can drift."""
    samples = [
        "I cannot find sufficient information to fully answer this question.",
        "EKS supports managed node groups and Fargate profiles.",
        "EKS supports node groups. However, the documentation does not mention pricing.",
        "",
    ]
    assert [guards.classify_response(s) for s in samples] == \
           [cfm.classify_response(s) for s in samples]


def test_guards_no_longer_hardcode_a_single_substring():
    """Source-level guard against re-introducing the narrow rule."""
    src = GUARDS.read_text(encoding="utf-8")
    assert 'DECLINE = "' not in src, "the single-substring decline rule is back"
    assert "classify_response" in src, "guards must use the canonical classifier"


def test_pattern_set_is_the_canonical_union(cfm):
    """28 markers = 14 canonical DECLINE_PATTERNS + 14 EXTENDED_REFUSAL_PATTERNS."""
    from src.generation.response_formatter import DECLINE_PATTERNS
    n = len(DECLINE_PATTERNS) + len(cfm.EXTENDED_REFUSAL_PATTERNS)
    assert len(cfm._refusal_markers()) == n
    assert n >= 28, f"the refusal marker set shrank to {n}"


# --------------------------------------------------------- classifier contract
def test_leading_refusal_is_pure_decline(cfm):
    assert cfm.classify_response(
        "Based on the provided documentation, I cannot find sufficient information "
        "to fully answer this question.") == "pure_decline"


def test_late_refusal_after_a_real_answer_is_hedged_not_pure(cfm):
    """The distinction that carries the interpretation: hedging is not abstaining.

    37/60 Tier A baseline answers carry a refusal phrase yet still assert claims and
    are scored normally, so collapsing these into 'declined' overstates abstention.
    """
    answer = ("Amazon EKS supports managed node groups, Fargate profiles and self-managed "
              "nodes. " + ("Cluster upgrades follow the documented version skew policy. " * 8)
              + "The documentation does not mention specific pricing tiers.")
    assert len(answer) > cfm.OPENING_WINDOW
    assert cfm.classify_response(answer) == "hedged_partial"


def test_clean_answer_is_answered(cfm):
    assert cfm.classify_response(
        "Amazon EKS supports managed node groups and Fargate profiles.") == "answered"


def test_empty_answer_is_none(cfm):
    assert cfm.classify_response("") is None
    assert cfm.classify_response("   ") is None


def test_matching_is_case_insensitive(cfm):
    """The old rule was case-sensitive; a capitalised refusal slipped through it."""
    for txt in ("I Cannot Find Sufficient Information to answer.",
                "I CANNOT FIND SUFFICIENT INFORMATION to answer."):
        assert cfm.classify_response(txt) == "pure_decline", txt


def test_extended_markers_catch_variants_the_canonical_list_misses(cfm):
    """These are why the two rules diverged by up to 24 points."""
    for txt in ("The provided context does not mention pricing tiers.",
                "There is no information about quotas in the documentation.",
                "This is beyond the scope of the provided documentation."):
        assert cfm.classify_response(txt) == "pure_decline", txt


# ------------------------------------------------------------------ artifacts
@pytest.mark.needs_artifacts
@pytest.mark.parametrize("exp", ["exp16_anchored_decoding", "exp17_crosscloud_balanced"])
def test_committed_guards_match_the_classifier(exp, cfm):
    """Recompute the class counts from results.json and demand the artifact agrees."""
    gpath = RESULTS / exp / "guards.json"
    rpath = RESULTS / exp / "results.json"
    if not gpath.exists() or not rpath.exists():
        pytest.skip(f"missing artifacts for {exp}")
    guards_doc = {a["scenario"]: a for a in json.loads(gpath.read_text(encoding="utf-8"))["arms"]}
    configs = json.loads(rpath.read_text(encoding="utf-8"))["configs"]

    for cname, c in configs.items():
        arm = c.get("scenario", cname.split(" | ")[0])
        counts = {"pure_decline": 0, "hedged_partial": 0, "answered": 0, "empty": 0}
        for r in c["results"]:
            counts[cfm.classify_response(r.get("answer") or "") or "empty"] += 1
        assert guards_doc[arm]["n_by_class"] == counts, f"{exp}/{arm}"


@pytest.mark.needs_artifacts
def test_guards_artifacts_declare_the_rule_they_used():
    """A future reader must be able to tell which definition produced the numbers."""
    for exp in ("exp16_anchored_decoding", "exp17_crosscloud_balanced"):
        p = RESULTS / exp / "guards.json"
        if not p.exists():
            pytest.skip(f"missing {p}")
        doc = json.loads(p.read_text(encoding="utf-8"))
        assert "classify_response" in doc.get("decline_rule", ""), exp
        assert "decline_phrase" not in doc, f"{exp} still carries the old single-phrase field"
