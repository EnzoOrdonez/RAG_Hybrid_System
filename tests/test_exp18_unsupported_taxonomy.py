"""Safety and sampling invariants for the exp18 unsupported-claim taxonomy."""

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import compute_exp18_unsupported_taxonomy as taxonomy  # noqa: E402


STRATUM_SIZES = {
    "d_threshold_artifact": 123,
    "a_synthesis_cand": 58,
    "b_parametric_cand": 178,
    "c_unattributed_cand": 400,
}


def _synthetic_strata():
    return {
        name: [
            {
                "query_id": f"{name}-{i}",
                "claim_idx": i,
                "claim": f"claim {i}",
                "best_over_pool": round(i / (size + 1), 4),
                "n_chunks_over_soft_tau": i % 3,
                "decline_class": "answered",
            }
            for i in range(size)
        ]
        for name, size in STRATUM_SIZES.items()
    }


def test_seed_42_produces_the_same_calibration_csv_twice():
    strata = _synthetic_strata()

    first, _ = taxonomy.build_calibration_sample(strata, seed=42)
    second, _ = taxonomy.build_calibration_sample(strata, seed=42)

    assert taxonomy.sample_csv_text(first) == taxonomy.sample_csv_text(second)


def test_calibration_design_is_coherent_for_all_759_claims():
    sample, summary = taxonomy.build_calibration_sample(_synthetic_strata(), seed=42)
    sampled_n = Counter(row["stratum"] for row in sample)

    assert len(sample) == 40
    assert sampled_n == {name: 10 for name in STRATUM_SIZES}
    assert sum(STRATUM_SIZES.values()) == 759
    assert all(row["stratum_size"] == STRATUM_SIZES[row["stratum"]] for row in sample)
    assert all(row["inclusion_prob"] * row["stratum_size"] == pytest.approx(10)
               for row in sample)
    assert all(row["human_verdict"] == row["human_notes"] == "" for row in sample)
    assert summary["kish_n_eff"] == pytest.approx(27.4)
    assert all(len(item["representative_examples"]) == 2
               for item in summary["strata"].values())


@pytest.mark.needs_artifacts
def test_out_suffix_v2_routes_every_write_away_from_the_original(monkeypatch):
    """Run the real analysis with in-memory sinks and prove no historical path is targeted."""
    original_paths = taxonomy.artifact_paths("")
    original_bytes = {kind: path.read_bytes() for kind, path in original_paths.items()}
    sinks = {}

    class Sink:
        def __init__(self, path):
            self.path = Path(path)
            self.text = None

        def write_text(self, text, **_kwargs):
            self.text = text
            return len(text)

    def capture_guard(path):
        path = Path(path)
        sink = Sink(path)
        sinks[path] = sink
        return sink

    monkeypatch.setattr(taxonomy, "guard_write", capture_guard)
    monkeypatch.setattr(sys, "argv", [taxonomy.__file__, "--out-suffix", "_v2"])

    taxonomy.main()

    expected_v2 = set(taxonomy.artifact_paths("_v2").values())
    assert set(sinks) == expected_v2
    assert expected_v2.isdisjoint(original_paths.values())
    assert all(sink.text is not None for sink in sinks.values())
    assert {kind: path.read_bytes() for kind, path in original_paths.items()} == original_bytes
