"""The gold harness must still run the day the annotations land — and its imports must not rot.

`analyze_gold_v4.py` is the consumer of `claim_audit_sample_v4.csv`, which is 4-5 hours of
Enzo's annotation time. It has never run on real data, so nothing was exercising it, and that
is exactly how it broke: retiring `deberta-large` renamed `NLI_TRIO` -> `NLI_MEMBERS` in
`compute_exp15_ensemble_sweep.py`, and the three `ens.NLI_TRIO` references in the gold script
became dead attribute lookups. The whole suite stayed green, because no test imported it.

Two guards, in increasing cost:
  1. cross-module symbol integrity — every `ens.<name>` referenced anywhere in the repo must
     exist on the module. Cheap, static, and catches the whole rename family, not this instance.
  2. the `--simulate` path end to end, so the day the CSV is filled the analysis runs.

Run: pytest tests/test_gold_harness.py -v
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS = PROJECT_ROOT / "scripts"
SWEEP = SCRIPTS / "compute_exp15_ensemble_sweep.py"
GOLD = SCRIPTS / "analyze_gold_v4.py"
PY = sys.executable


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def sweep():
    return _load("ens_t", SWEEP)


def _ens_references():
    """Every `ens.<attr>` written anywhere under scripts/, with the file it came from."""
    refs = []
    for p in sorted(SCRIPTS.glob("*.py")):
        for attr in set(re.findall(r"\bens\.([A-Za-z_][A-Za-z0-9_]*)", p.read_text(encoding="utf-8"))):
            refs.append((p.name, attr))
    return refs


def test_there_is_something_to_check():
    """A guard that silently checks nothing is worse than no guard."""
    assert _ens_references(), "no `ens.` references found — has the import alias changed?"


@pytest.mark.parametrize("src,attr", _ens_references(), ids=lambda v: str(v))
def test_every_cross_module_symbol_still_exists(src, attr, sweep):
    """The rename guard. `ens.NLI_TRIO` survived a rename here and nothing noticed."""
    assert hasattr(sweep, attr), (
        f"{src} references `ens.{attr}`, which no longer exists in "
        f"compute_exp15_ensemble_sweep.py — a rename left it orphaned")


def test_the_retired_member_is_gone_from_the_expected_set(sweep):
    """deberta-large is retired; the standard is small + base + HHEM (two families)."""
    assert "large" not in sweep.NLI_MEMBERS_EXPECTED
    assert sweep.NLI_MEMBERS_EXPECTED == ("small", "base")
    assert "large" in sweep.NLI_MEMBERS_RETIRED


def test_two_members_cannot_be_called_a_majority_vote(sweep):
    """E2_vote needs >=2 agreeing labels: a MAJORITY with three members, UNANIMITY with two.

    Anchored on the member count, not on a comparison against an expectation — otherwise
    retiring `large` would silently rename the estimator back to a claim it does not support.
    """
    if len(sweep.NLI_MEMBERS) >= 3:
        pytest.skip("a third member is present again; majority is genuinely possible")
    assert sweep.MAJORITY_POSSIBLE is False
    assert sweep.candidate_label("E2_vote") == f"E2_vote[{len(sweep.NLI_MEMBERS)}m=unanimity]"
    assert "unanimity" not in sweep.candidate_label("hhem"), "only the vote changes meaning"


@pytest.mark.slow
@pytest.mark.needs_artifacts
def test_the_gold_analysis_runs_end_to_end_on_simulated_judgements():
    """It must not first be exercised on the annotations it took 4-5 hours to produce."""
    if not (PROJECT_ROOT / "output/audit/claim_audit_sample_v4.csv").exists():
        pytest.skip("gold sample not present")
    r = subprocess.run([PY, str(GOLD), "--simulate", "0.15"], cwd=PROJECT_ROOT,
                       capture_output=True, text=True, timeout=900,
                       env={**__import__("os").environ, "PYTHONUTF8": "1",
                            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    assert r.returncode == 0, f"gold harness failed:\n{r.stderr[-2000:]}"
    assert "SIMULACION" in r.stdout, "the simulate path did not report completion"
    assert "kappa_weighted" in r.stdout, "no kappa reported"


@pytest.mark.needs_artifacts
def test_simulation_writes_nothing():
    """A smoke run must never leave an artifact that could be mistaken for real results."""
    out = PROJECT_ROOT / "experiments/results/exp15_ablation_nli"
    assert not (out / "gold_v4_analysis__simulated.json").exists()
    for p in out.glob("*simul*"):
        pytest.fail(f"simulation left an artifact behind: {p.name}")
