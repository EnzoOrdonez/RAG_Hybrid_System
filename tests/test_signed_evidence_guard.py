"""
Overwriting a file inside signed evidence must be impossible by accident.

`experiments/results/exp3..exp14` (+`exp8b`) are tagged, immutable, and back numbers already
delivered in the A.3 report and the LACCI paper. Two scripts aim at them by default:
`compute_faithfulness_metrics.py --experiment` defaults to **exp12_matrix** and
`compute_retrieval_metrics.py --experiment` defaults to **exp8**, and both write in place.
Nothing has been clobbered so far (git diff against the tag shows additions only), but one
distracted invocation with no arguments is all it would take.

The guard blocks exactly the forbidden action -- overwriting an EXISTING file in a signed
dir -- and nothing else. Creating new `_vN` artifacts there stays allowed, since that is the
sanctioned way to recompute.

Run: pytest tests/test_signed_evidence_guard.py -v
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import signed_evidence as SE  # noqa: E402

RESULTS = PROJECT_ROOT / "experiments" / "results"


def test_the_signed_set_is_still_only_what_carries_a_TAG():
    """SIGNED means signed. exp3..exp14 plus exp8b; exp1/exp2 never existed.

    The summer experiments are protected too (see below) but they are NOT tagged, and the
    error message quotes the reason at the reader. Folding them into SIGNED_EXPERIMENTS would
    make the guard state something false every time it fires.
    """
    assert "exp3" in SE.SIGNED_EXPERIMENTS and "exp14" in SE.SIGNED_EXPERIMENTS
    assert "exp8b" in SE.SIGNED_EXPERIMENTS
    assert "exp1" not in SE.SIGNED_EXPERIMENTS and "exp2" not in SE.SIGNED_EXPERIMENTS
    assert "exp15" not in SE.SIGNED_EXPERIMENTS, "summer experiments are protected, not signed"
    assert "exp18" not in SE.SIGNED_EXPERIMENTS


@pytest.mark.parametrize("name", ["exp12_matrix", "exp8", "exp8b", "exp11_retrieval194_fullrerank"])
def test_recognises_signed_dirs_by_experiment_prefix(name):
    """Suffixes vary (exp10_retrieval194, exp12_matrix...); the prefix is what matters."""
    assert SE._protected_dir_for(RESULTS / name / "whatever.json") == name


# --------------------------------------------------------- the registry, inverted 2026-08-21
# The set used to be a hand-written list of what to protect, and this phase has already paid
# twice for hand-written lists of work: pass_n's arm registry silently scored 1 of 4 exp18 arms
# (ledger entry 21) and verify_summer_offline.py's hardcoded experiment list left exp18
# unverified. A list of what to protect goes stale in the DANGEROUS direction -- a new
# experiment is unprotected by default, and nobody notices until something is overwritten.
#
# Inverted: everything under experiments/results is protected against OVERWRITE unless it is
# declared LIVE. A stale LIVE entry costs a false refusal, which is loud and harmless. A stale
# protect-list costs clobbered evidence, which is silent and permanent.
@pytest.mark.parametrize("name", ["exp15_ablation_tierA", "exp15_ablation_nli",
                                  "exp16_anchored_decoding", "exp17_crosscloud_balanced",
                                  "exp18_evidence_ceiling", "exp19a_selector_probe"])
def test_summer_dirs_are_protected_against_overwrite(name):
    """The project rule says exp15+ is not overwritten either; now the code says it too."""
    assert SE._protected_dir_for(RESULTS / name / "results.json") == name


@pytest.mark.parametrize("name", ["exp19b_anchored_selector",
                                  "exp19b_anchored_selector/_smoke"])
def test_the_live_experiment_stays_writable(name):
    """A run in flight rewrites its own results.json and checkpoints; that is not clobbering."""
    assert "exp19b" in SE.LIVE_EXPERIMENTS
    assert SE._protected_dir_for(RESULTS / name / "results.json") is None


def test_a_future_experiment_is_protected_until_it_is_declared_live():
    """The anti-staleness property, and the whole reason for inverting the registry."""
    assert SE._protected_dir_for(RESULTS / "exp20_whatever" / "results.json") == "exp20_whatever"


def test_non_experiment_dirs_under_results_are_not_guarded():
    assert SE._protected_dir_for(RESULTS / "scratch" / "notes.json") is None


@pytest.mark.needs_artifacts
def test_a_NEW_artifact_under_a_protected_summer_dir_is_still_allowed():
    """The requirement Enzo called out: summer runners must keep writing NEW files there.

    guard_write only refuses when the target ALREADY EXISTS, so widening the protected set
    cannot block creation. This pins that, because it is the property most likely to be broken
    by a future "tighten the guard" change.
    """
    p = RESULTS / "exp18_evidence_ceiling" / "does_not_exist_v99.json"
    assert not p.exists()
    assert SE.guard_write(p) == p
    assert not p.exists(), "guard_write must not create anything"


@pytest.mark.needs_artifacts
def test_the_refusal_states_the_RIGHT_reason_for_each_kind():
    """A guard that cites a tag the directory does not carry teaches people to distrust it."""
    signed = RESULTS / "exp12_matrix" / "faithfulness_metrics.json"
    frozen = RESULTS / "exp18_evidence_ceiling" / "results.json"
    if not (signed.exists() and frozen.exists()):
        pytest.skip("artifacts not present")
    with pytest.raises(SystemExit) as a:
        SE.guard_write(signed)
    with pytest.raises(SystemExit) as b:
        SE.guard_write(frozen)
    assert "nota3-evidencia" in str(a.value)
    assert "nota3-evidencia" not in str(b.value), "exp18 carries no such tag"
    assert "exp15" in str(b.value) or "verano" in str(b.value) or "summer" in str(b.value)


def test_paths_outside_results_are_not_guarded(tmp_path):
    assert SE._protected_dir_for(tmp_path / "x.json") is None
    assert SE.guard_write(tmp_path / "x.json") == tmp_path / "x.json"


@pytest.mark.needs_artifacts
def test_REFUSES_to_overwrite_an_existing_signed_artifact():
    """The whole point. Uses a real committed artifact, and never writes."""
    target = RESULTS / "exp12_matrix" / "faithfulness_metrics.json"
    if not target.exists():
        pytest.skip("signed artifact not present")
    with pytest.raises(SystemExit) as e:
        SE.guard_write(target)
    # The refusal, not its exact prose: the message now names the reason per directory kind
    # (tagged vs frozen summer), and pinning the whole sentence would break on every wording fix.
    assert "REFUSING to overwrite" in str(e.value)
    assert target.exists(), "the guard must not touch the file"


@pytest.mark.needs_artifacts
def test_allows_a_NEW_vN_file_in_a_signed_dir():
    """Recomputation to new _vN files is the sanctioned path and must stay open."""
    p = RESULTS / "exp12_matrix" / "faithfulness_metrics_v99_does_not_exist.json"
    assert not p.exists()
    assert SE.guard_write(p) == p          # no raise
    assert not p.exists(), "guard_write must not create anything"


@pytest.mark.needs_artifacts
def test_explicit_override_is_possible_but_must_be_asked_for():
    target = RESULTS / "exp12_matrix" / "faithfulness_metrics.json"
    if not target.exists():
        pytest.skip("signed artifact not present")
    assert SE.guard_write(target, allow_overwrite=True) == target


def test_both_in_place_scripts_route_their_writes_through_the_guard():
    """Source guard: the defaults of these two aim at signed dirs."""
    for name, n_expected in [("compute_faithfulness_metrics.py", 2),
                             ("compute_retrieval_metrics.py", 3)]:
        src = (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "from src.utils.signed_evidence import guard_write" in src, name
        # the import has no parenthesis, so this counts CALL SITES only
        assert src.count("guard_write(") >= n_expected, (
            f"{name}: expected >={n_expected} guarded write sites, "
            f"found {src.count('guard_write(')}")


@pytest.mark.parametrize("name", ["SIGNED_EXPERIMENTS", "LIVE_EXPERIMENTS"])
def test_the_registries_are_defined_once(name):
    """Copying these lists into callers is the duplication pattern that caused the phase's
    silent defects; they must live only in the helper.

    LIVE_EXPERIMENTS is covered too: a second copy of "what is still being written" is exactly
    how one of them would be forgotten when an experiment closes.
    """
    import subprocess
    hits = subprocess.run(
        ["git", "grep", "-l", name, "--", "src", "scripts", "tests"],
        cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.split()
    assert set(hits) <= {"src/utils/signed_evidence.py",
                         "tests/test_signed_evidence_guard.py"}, hits
