"""
Unit tests for the paired arm-vs-baseline harness (scripts/compute_tierA_arm_stats.py).

Guards the defect class that ledger entry 9 had to publicly retract: a BH family
declared in the artifact that does not match the family the p-values were actually
corrected over. The script is reused across experiments via --exp-dir/--baseline-arm
(Tier A: 4 contrasts/n=60/baseline_repro; exp16: 2 contrasts; exp17: 1 contrast/n=25/
baseline), so any metadata field that is hardcoded instead of derived silently
mislabels two of the three artifacts.

  1. bh_family / baseline / n_queries_subset are DERIVED from the run, not literals.
  2. The committed artifacts agree with their own contrast lists.
  3. pair() is decline-aware: None on either side drops the pair; vacuous 1.0 stays.

Run: pytest tests/test_arm_stats.py -v
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "experiments" / "results"
SCRIPT = PROJECT_ROOT / "scripts" / "compute_tierA_arm_stats.py"

# Reads committed arm_stats artifacts; skips cleanly when they are absent.
pytestmark = pytest.mark.needs_artifacts

# (experiment dir, expected baseline arm) — the three consumers of this script.
ARTIFACTS = [
    ("exp15_ablation_tierA", "baseline_repro"),
    ("exp16_anchored_decoding", "baseline_repro"),
    ("exp17_crosscloud_balanced", "baseline"),
]
VERIFIERS = ["small", "base", "hhem"]


@pytest.fixture(scope="module")
def mod():
    """Import the script as a module (it has no package path of its own)."""
    spec = importlib.util.spec_from_file_location("arm_stats", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(exp, verifier):
    p = RESULTS / exp / f"arm_stats__{verifier}.json"
    if not p.exists():
        pytest.skip(f"artifact not present: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 1. Declared BH family == family the p-values were actually corrected over
# --------------------------------------------------------------------------

@pytest.mark.parametrize("exp,baseline", ARTIFACTS)
@pytest.mark.parametrize("verifier", VERIFIERS)
def test_declared_bh_family_matches_contrast_count(exp, baseline, verifier):
    """The integer in bh_family must equal len(contrasts).

    This is the regression guard for the hardcoded
    '4 arm-vs-baseline_repro contrasts' that mislabelled exp16 (2) and exp17 (1).
    """
    doc = _load(exp, verifier)
    n_declared = int(re.match(r"\s*(\d+)", doc["bh_family"]).group(1))
    assert n_declared == len(doc["contrasts"]), (
        f"{exp}/{verifier}: bh_family declares {n_declared} contrasts but the "
        f"artifact carries {len(doc['contrasts'])} — the BH correction and the "
        f"declaration disagree."
    )


@pytest.mark.parametrize("exp,baseline", ARTIFACTS)
@pytest.mark.parametrize("verifier", VERIFIERS)
def test_baseline_and_n_are_derived_not_hardcoded(exp, baseline, verifier):
    """baseline / n_queries_subset must describe THIS run, not Tier A's."""
    doc = _load(exp, verifier)
    assert doc["baseline"] == baseline, (
        f"{exp}/{verifier}: baseline '{doc['baseline']}' != actual anchor '{baseline}'"
    )
    assert doc["bh_family"].endswith("(fdr_bh)")
    assert baseline in doc["bh_family"], (
        f"{exp}/{verifier}: bh_family names the wrong anchor: {doc['bh_family']}"
    )
    # n_queries_subset must be consistent with the contrasts it summarises.
    n_sub = doc["n_queries_subset"]
    for c in doc["contrasts"]:
        assert c["n_common_qids"] <= n_sub, (
            f"{exp}/{verifier}: contrast pairs {c['n_common_qids']} qids but the "
            f"artifact claims a {n_sub}-query subset"
        )


@pytest.mark.parametrize("exp,baseline", ARTIFACTS)
@pytest.mark.parametrize("verifier", VERIFIERS)
def test_bh_pvalues_consistent_with_family_size(exp, baseline, verifier):
    """p_bh >= p_value always, and a family of 1 leaves p untouched.

    Cheap arithmetic check that the correction was applied over the declared
    family rather than some other one.
    """
    doc = _load(exp, verifier)
    cs = doc["contrasts"]
    for c in cs:
        assert c["p_bh"] >= c["p_value"] - 1e-9, (
            f"{exp}/{verifier}/{c['arm']}: p_bh {c['p_bh']} < raw p {c['p_value']}"
        )
    if len(cs) == 1:
        c = cs[0]
        assert abs(c["p_bh"] - c["p_value"]) < 1e-9, (
            "BH over a family of 1 must be the identity; "
            f"got p={c['p_value']} -> p_bh={c['p_bh']}"
        )


# --------------------------------------------------------------------------
# 2. Metadata is genuinely derived (guards against re-hardcoding)
# --------------------------------------------------------------------------

def test_metadata_fields_are_not_string_literals():
    """Source-level guard: the three fields must not be assigned constants.

    A future edit that re-hardcodes them would pass the artifact tests above
    only until the artifacts are regenerated — this catches it at the source.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    assert '"baseline": "baseline_repro"' not in src, (
        "baseline is hardcoded again; it must come from args.baseline_arm"
    )
    assert '"n_queries_subset": 60' not in src, (
        "n_queries_subset is hardcoded again; it must be derived from the rows"
    )
    assert '"bh_family": "4 ' not in src, (
        "bh_family is hardcoded again; it must be built from len(contrasts)"
    )


# --------------------------------------------------------------------------
# 3. Decline-aware pairing
# --------------------------------------------------------------------------

def test_pair_drops_none_on_either_side_and_keeps_vacuous(mod):
    """None faithfulness (declined) drops the PAIR; vacuous 1.0 is kept."""
    base = {
        "q1": {"faithfulness": 0.4},
        "q2": {"faithfulness": None},   # baseline declined
        "q3": {"faithfulness": 0.6},
        "q4": {"faithfulness": 1.0},    # vacuous -> genuine value, must survive
        "q5": {"faithfulness": 0.2},    # not present in arm -> not a common qid
    }
    arm = {
        "q1": {"faithfulness": 0.5},
        "q2": {"faithfulness": 0.9},
        "q3": {"faithfulness": None},   # arm declined
        "q4": {"faithfulness": 1.0},
    }
    bvec, avec, n_common, ndb, nda = mod.pair(base, arm)

    assert n_common == 4, "q5 is absent from the arm and is not a common qid"
    assert bvec == [0.4, 1.0] and avec == [0.5, 1.0], (
        "only q1 and q4 are complete pairs; q2 (baseline None) and q3 (arm None) drop"
    )
    assert ndb == 1 and nda == 1, "one decline counted on each side"


def test_pair_reports_declines_even_when_pair_is_dropped(mod):
    """Both-declined queries count on both sides and contribute no pair."""
    base = {"q1": {"faithfulness": None}, "q2": {"faithfulness": 0.3}}
    arm = {"q1": {"faithfulness": None}, "q2": {"faithfulness": 0.3}}
    bvec, avec, n_common, ndb, nda = mod.pair(base, arm)
    assert n_common == 2
    assert bvec == [0.3] and avec == [0.3]
    assert ndb == 1 and nda == 1


def test_pair_is_order_stable_by_qid(mod):
    """Vectors are aligned by sorted qid so the pairing is reproducible."""
    base = {"q3": {"faithfulness": 0.3}, "q1": {"faithfulness": 0.1}}
    arm = {"q1": {"faithfulness": 0.9}, "q3": {"faithfulness": 0.7}}
    bvec, avec, *_ = mod.pair(base, arm)
    assert bvec == [0.1, 0.3] and avec == [0.9, 0.7]
