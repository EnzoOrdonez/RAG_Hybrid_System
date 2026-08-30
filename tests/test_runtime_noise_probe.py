"""The runtime noise probe — its verdict rule, pinned before any data exist.

Seccion de Claude Code — 2026-08-21 15:00 (hora local).

This probe decides whether exp21's pre-registered +/-0.081 band is meaningful at all, so the
dangerous outcome is not a wrong number: it is an INCONCLUSIVE result quietly read as a pass.
An underpowered CI that happens to straddle zero looks reassuring and says nothing. The rule
therefore has three branches, not two, and the third one is tested hardest.

Nothing here loads HHEM or talks to a server.

Run: pytest tests/test_runtime_noise_probe.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def probe():
    path = PROJECT_ROOT / "scripts" / "run_runtime_noise_probe.py"
    assert path.exists(), "scripts/run_runtime_noise_probe.py missing"
    spec = importlib.util.spec_from_file_location("runtime_noise", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_nuisance_limit_is_a_quarter_of_the_band(probe):
    """The band must absorb the tested effect AND this nuisance; a quarter keeps it second-order."""
    assert probe.TOST_BAND == 0.081
    assert probe.NUISANCE_LIMIT == pytest.approx(0.081 / 4, abs=1e-4)


def test_a_tight_ci_around_zero_says_the_band_survives(probe):
    code, text = probe.verdict(0.003, -0.010, 0.016)
    assert code == "b_band_survives" and "keep its pre-registered band" in text


def test_a_clear_shift_says_systematic_bias(probe):
    code, text = probe.verdict(0.06, 0.03, 0.09)
    assert code == "a_systematic_bias" and "could not be separated" in text


def test_a_wide_ci_straddling_zero_is_INCONCLUSIVE_not_a_pass(probe):
    """The failure mode this rule exists for: n too small, CI includes 0, and someone reads
    'no significant difference' as 'equivalent'."""
    code, text = probe.verdict(0.004, -0.15, 0.16)
    assert code == "c_underpowered"
    assert "not a pass" in text.lower()


def test_a_ci_excluding_zero_but_tiny_is_not_called_a_bias(probe):
    """A real but negligible shift must not trip the alarm: the band can absorb it."""
    code, _ = probe.verdict(0.006, 0.002, 0.010)
    assert code == "b_band_survives"


def test_the_boundary_is_inclusive_on_the_surviving_side(probe):
    lim = probe.NUISANCE_LIMIT
    assert probe.verdict(0.0, -lim, lim)[0] == "b_band_survives"
    assert probe.verdict(0.0, -lim * 1.01, lim * 1.01)[0] != "b_band_survives"


def test_paired_shift_drops_undefined_pairs_and_reports_how_many(probe):
    fa = [0.5, None, 0.4, 0.6, 0.55]
    fb = [0.52, 0.3, None, 0.61, 0.50]
    res = probe.paired_shift(fa, fb)
    assert res["n_paired"] == 3 and res["n_dropped"] == 2


def test_paired_shift_refuses_to_speak_on_too_few_pairs(probe):
    assert probe.paired_shift([0.5, None], [0.5, 0.4]) is None


def test_paired_shift_is_deterministic(probe):
    fa = [0.4, 0.5, 0.6, 0.7, 0.55, 0.45]
    fb = [0.42, 0.49, 0.63, 0.68, 0.57, 0.44]
    a, b = probe.paired_shift(fa, fb), probe.paired_shift(fa, fb)
    assert a["boot95"] == b["boot95"], "seeded bootstrap must repeat exactly"


def test_the_result_declares_it_is_not_evidence(probe):
    res = probe.paired_shift([0.4, 0.5, 0.6, 0.7], [0.41, 0.52, 0.58, 0.72])
    assert "no BH family" in res["not_evidence"] and "TOST" in res["not_evidence"]


def test_nothing_is_written_under_experiments_results(probe):
    """This measures the instrument; it must not look like experimental evidence."""
    assert probe.PROBE_DIR.name == "runtime_noise"
    assert "probes" in probe.PROBE_DIR.parts
    assert "results" not in probe.PROBE_DIR.parts


def test_the_hhem_scoring_rule_matches_the_phase(probe):
    """Same tau, same premise truncation as rescore_grounding_exp15, or the numbers are not
    comparable to any other HHEM figure in the project."""
    assert probe.TAU == 0.5
    assert probe.PREMISE_CHARS == 1500
