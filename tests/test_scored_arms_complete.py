"""
Every arm that was GENERATED must be SCORED, and a pass that skips work must not exit 0.

This exists because exp18 was scored with `run_exp15_ablation.py --pass N` and only
`baseline_repro` came out, silently: `main()` defaulted the arm list to the arm REGISTRY,
which defaults to Tier A's `ablation_arms.json`. exp18's arms (oracle_evidence,
evidence_swapped, final_top_k_10) are not in that registry, so the intersection was one arm,
three were skipped without a message, and the run still printed "Pass N done". HHEM was fine
because `rescore_grounding_tierA.py` iterates results.json directly -- that asymmetry between
the two scorers was the smell.

SCOPE. These tests deliberately only look at experiment dirs that carry the ARM SCHEMA
(results.json whose configs have a `scenario`, plus faithfulness_rows__* files). The signed
evidence in exp3..exp14 predates that schema and is immutable, so including it would paint
the suite red for reasons nobody is allowed to fix -- and a test that is red for untouchable
reasons is a test that gets loosened. The filter is by artifact SHAPE, not by experiment
number, so a future experiment is covered automatically without editing this file.

Run: pytest tests/test_scored_arms_complete.py -v
"""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS = PROJECT_ROOT / "experiments" / "results"
ABLATION = PROJECT_ROOT / "scripts" / "run_exp15_ablation.py"
DETCHECK = PROJECT_ROOT / "scripts" / "check_rescore_determinism.py"


def arm_schema_dirs():
    """Experiment dirs using the arm schema. Shape-based, so it grows by itself."""
    out = []
    if not RESULTS.exists():
        return out
    for d in sorted(RESULTS.iterdir()):
        rp = d / "results.json"
        if not (d.is_dir() and rp.exists()):
            continue
        try:
            cfgs = json.loads(rp.read_text(encoding="utf-8")).get("configs", {})
        except Exception:
            continue
        if cfgs and all("scenario" in c for c in cfgs.values()) and \
                any(d.glob("faithfulness_rows__*.json")):
            out.append(d)
    return out


def generated_arms(exp_dir):
    cfgs = json.loads((exp_dir / "results.json").read_text(encoding="utf-8"))["configs"]
    return {c["scenario"] for c in cfgs.values()}


def scored_arms(rows_path):
    return {k.split(" | ")[0]
            for k in json.loads(rows_path.read_text(encoding="utf-8"))["configs"]}


# ------------------------------------------------------------------- artifacts
@pytest.mark.needs_artifacts
def test_there_is_something_to_check():
    dirs = arm_schema_dirs()
    assert dirs, "no arm-schema experiment dirs found; the shape filter is too strict"


@pytest.mark.needs_artifacts
def test_signed_evidence_is_out_of_scope():
    """exp3..exp14 must never be pulled in: immutable, different schema."""
    names = {d.name for d in arm_schema_dirs()}
    legacy = {n for n in names
              if n.startswith("exp") and n[3:5].rstrip("_").isdigit()
              and int(n[3:5].rstrip("_")) <= 14}
    assert not legacy, f"signed evidence dirs matched the shape filter: {sorted(legacy)}"


@pytest.mark.needs_artifacts
@pytest.mark.parametrize("exp_dir", arm_schema_dirs(), ids=lambda d: d.name)
def test_every_generated_arm_was_scored(exp_dir):
    """The regression. Red on exp18/NLI until the fixed pass N is re-run."""
    gen = generated_arms(exp_dir)
    for rows in sorted(exp_dir.glob("faithfulness_rows__*.json")):
        got = scored_arms(rows)
        assert got == gen, (
            f"{exp_dir.name}/{rows.name}: scored {sorted(got)} but results.json generated "
            f"{sorted(gen)}. Missing: {sorted(gen - got)}")


@pytest.mark.needs_artifacts
@pytest.mark.parametrize("exp_dir", arm_schema_dirs(), ids=lambda d: d.name)
def test_raw_probs_cover_the_same_arms_as_the_rows(exp_dir):
    """rows and probs must agree; a mismatch means one of them was written from stale state."""
    for probs in sorted(exp_dir.glob("nli_probs__*.json.gz")):
        tag = probs.name.removeprefix("nli_probs__").removesuffix(".json.gz")
        if ".partial" in tag:
            continue
        rows = exp_dir / f"faithfulness_rows__{tag}__vb_agree.json"
        if not rows.exists():
            continue
        with gzip.open(probs, "rt", encoding="utf-8") as f:
            p_arms = {k.split(" | ")[0] for k in json.load(f)["configs"]}
        assert p_arms == scored_arms(rows), f"{exp_dir.name}: {probs.name} vs {rows.name}"


# ---------------------------------------------------------------------- source
def test_pass_n_resolves_arms_from_results_not_the_registry():
    src = ABLATION.read_text(encoding="utf-8")
    assert 'if args.pass_ == "G":' in src, (
        "the arm list must be resolved per pass; pass N must not inherit pass G's registry")
    tail = src[src.index('if args.pass_ == "G":'):]
    pass_n_branch = tail[tail.index("else:"):]
    assert "results.json" in pass_n_branch, (
        "pass N must take its work list from results.json")
    assert "registry" not in pass_n_branch.split("logger.info")[0], (
        "pass N still consults the arm registry")


def test_pass_n_has_a_completeness_gate():
    src = ABLATION.read_text(encoding="utf-8")
    assert "INCOMPLETE pass N" in src, "no completeness gate in pass_n"
    gate = src[src.index("INCOMPLETE pass N") - 600: src.index("INCOMPLETE pass N")]
    assert "arms_explicit" in gate, (
        "the gate must not fire when the caller asked for an explicit --arms subset")
    # the gate must abort, not warn
    i_gate = src.index("INCOMPLETE pass N")
    assert "sys.exit(" in src[i_gate - 200:i_gate], "the gate warns instead of aborting"
    assert i_gate < src.index('f"nli_probs__{args.verifier}.json.gz"'), (
        "the gate must run BEFORE the artifacts are written")


# ------------------------------------------------- determinism comparator works
def _detcheck():
    spec = importlib.util.spec_from_file_location("detchk", DETCHECK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(exp_dir, verifier, rows_payload):
    p = exp_dir / f"faithfulness_rows__{verifier}__vb_agree.json"
    p.write_text(json.dumps(rows_payload), encoding="utf-8")
    return p


def test_determinism_comparator_passes_on_identical(tmp_path, monkeypatch):
    m = _detcheck()
    payload = {"configs": {"a | m": {"q1": {"faithfulness": 0.5}}}}
    _write(tmp_path, "small", payload)
    monkeypatch.setattr(sys, "argv", ["x", "snapshot", "--exp-dir", str(tmp_path)])
    assert m.main() == 0
    monkeypatch.setattr(sys, "argv", ["x", "compare", "--exp-dir", str(tmp_path)])
    assert m.main() == 0


def test_determinism_comparator_DETECTS_drift(tmp_path, monkeypatch):
    """A comparator that can only say OK is worthless; prove it fails on a changed cell."""
    m = _detcheck()
    _write(tmp_path, "small", {"configs": {"a | m": {"q1": {"faithfulness": 0.5}}}})
    monkeypatch.setattr(sys, "argv", ["x", "snapshot", "--exp-dir", str(tmp_path)])
    assert m.main() == 0
    # one cell moves — exactly what a changed model/extractor/threshold would do
    _write(tmp_path, "small", {"configs": {"a | m": {"q1": {"faithfulness": 0.6}}}})
    monkeypatch.setattr(sys, "argv", ["x", "compare", "--exp-dir", str(tmp_path)])
    assert m.main() == 1, "drift went undetected"


def test_determinism_comparator_refuses_without_a_snapshot(tmp_path, monkeypatch):
    m = _detcheck()
    _write(tmp_path, "small", {"configs": {"a | m": {"q1": {"faithfulness": 0.5}}}})
    monkeypatch.setattr(sys, "argv", ["x", "compare", "--exp-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        m.main()
