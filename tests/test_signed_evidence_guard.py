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


def test_the_signed_set_matches_the_project_rule():
    """exp3..exp14 plus exp8b; exp1/exp2 never existed."""
    assert "exp3" in SE.SIGNED_EXPERIMENTS and "exp14" in SE.SIGNED_EXPERIMENTS
    assert "exp8b" in SE.SIGNED_EXPERIMENTS
    assert "exp1" not in SE.SIGNED_EXPERIMENTS and "exp2" not in SE.SIGNED_EXPERIMENTS
    assert "exp15" not in SE.SIGNED_EXPERIMENTS, "summer experiments are NOT signed"
    assert "exp18" not in SE.SIGNED_EXPERIMENTS


@pytest.mark.parametrize("name", ["exp12_matrix", "exp8", "exp8b", "exp11_retrieval194_fullrerank"])
def test_recognises_signed_dirs_by_experiment_prefix(name):
    """Suffixes vary (exp10_retrieval194, exp12_matrix...); the prefix is what matters."""
    assert SE._signed_dir_for(RESULTS / name / "whatever.json") == name


@pytest.mark.parametrize("name", ["exp15_ablation_tierA", "exp17_crosscloud_balanced",
                                  "exp18_evidence_ceiling"])
def test_summer_dirs_are_not_guarded(name):
    assert SE._signed_dir_for(RESULTS / name / "results.json") is None


def test_paths_outside_results_are_not_guarded(tmp_path):
    assert SE._signed_dir_for(tmp_path / "x.json") is None
    assert SE.guard_write(tmp_path / "x.json") == tmp_path / "x.json"


@pytest.mark.needs_artifacts
def test_REFUSES_to_overwrite_an_existing_signed_artifact():
    """The whole point. Uses a real committed artifact, and never writes."""
    target = RESULTS / "exp12_matrix" / "faithfulness_metrics.json"
    if not target.exists():
        pytest.skip("signed artifact not present")
    with pytest.raises(SystemExit) as e:
        SE.guard_write(target)
    assert "REFUSING to overwrite signed evidence" in str(e.value)
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


def test_the_signed_set_is_defined_once():
    """Copying this list into callers is the duplication pattern that caused the phase's
    silent defects; it must live only in the helper."""
    import subprocess
    hits = subprocess.run(
        ["git", "grep", "-l", "SIGNED_EXPERIMENTS", "--", "src", "scripts", "tests"],
        cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.split()
    assert set(hits) <= {"src/utils/signed_evidence.py",
                         "tests/test_signed_evidence_guard.py"}, hits
