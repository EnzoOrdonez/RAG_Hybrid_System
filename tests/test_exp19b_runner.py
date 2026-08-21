"""exp19b — the runner's decision rules, pinned before the GPU is spent.

Seccion de Claude Code — 2026-08-21 14:05 (hora local).

exp19b is the generative arm of the anchoring-guided selector: draft with the exp18 baseline
context, extract the draft's claims, re-rank the k=50 pool by (claim, chunk), regenerate from
the new top-5. Four things about it can go wrong silently, and each has a case here:

  1. A draft that asserts nothing has no claims to condition on. Left unhandled the selector
     would return an EMPTY selection and the arm would generate with no context at all --
     scoring as a spectacular "faithfulness" win on zero claims. The fallback rule is
     pre-registered, so it is a test, not a patch.
  2. Format artifacts (markdown headers, table rows) are not claims. Conditioning the selector
     on them steers the pool towards layout instead of content.
  3. The anchor arm must be named `baseline_repro`: verify_summer_offline.py and
     compute_exp18_diagnosis.py both LOCATE the anchor by that name. A different name means
     the experiment is silently unverified -- the same defect family as ledger entry 21.
  4. Smoke artifacts must not land where the shape-based discovery of
     tests/test_scored_arms_complete.py finds them. A 3-query smoke has no scores, so a smoke
     written into the experiment dir turns the suite red for a reason nobody should "fix".

Run: pytest tests/test_exp19b_runner.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(name):
    path = PROJECT_ROOT / "scripts" / name
    assert path.exists(), f"scripts/{name} missing"
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def selector():
    return _load("select_exp19b_evidence.py")


@pytest.fixture(scope="module")
def extractor():
    return _load("extract_exp19b_claims.py")


@pytest.fixture(scope="module")
def generation():
    return _load("run_exp19b_generation.py")


@pytest.fixture(scope="module")
def stats():
    return _load("compute_exp19b_stats.py")


# --------------------------------------------------------------- 1. the fallback rule
def test_a_draft_with_no_genuine_claims_falls_back_to_the_baseline_selection(selector):
    """Pre-registered: no claims to condition on => keep the baseline top-5, difference 0."""
    ids, reason = selector.selection_for_query(
        pool_ids=[f"c{i}" for i in range(20)], R=None, baseline_ids=["c3", "c1", "c7"])
    assert ids == ["c3", "c1", "c7"]
    assert reason == "no_genuine_claims"


def test_the_fallback_never_returns_an_empty_context(selector):
    """An empty selection would generate with NO evidence and score as a vacuous win."""
    import numpy as np
    for R in (None, np.zeros((0, 20))):
        ids, _ = selector.selection_for_query(["c0", "c1"], R, ["c0"])
        assert ids, "empty context is never an acceptable selection"


# --------------------------------------------------------------- 2. the selection itself
def test_selection_returns_k_distinct_pool_ids(selector):
    import numpy as np
    pool = [f"c{i}" for i in range(20)]
    R = np.random.default_rng(0).random((6, 20))
    ids, reason = selector.selection_for_query(pool, R, ["c0"])
    assert reason is None
    assert len(ids) == selector.FINAL_K
    assert len(set(ids)) == selector.FINAL_K
    assert set(ids) <= set(pool)


def test_selection_is_capped_by_a_short_pool(selector):
    import numpy as np
    ids, _ = selector.selection_for_query(["c0", "c1"], np.ones((3, 2)), ["c0"])
    assert len(ids) == 2, "cannot select more chunks than the pool holds"


def test_the_selector_reuses_the_exp19a_greedy_and_does_not_reimplement_it(selector):
    probe = _load("compute_exp19a_selector_probe.py")
    assert selector.select_by_claims.__code__.co_code == probe.select_by_claims.__code__.co_code, \
        "a second copy of the greedy is how the two would drift apart"


# --------------------------------------------------------------- 3. what conditions the selector
def test_format_artifacts_do_not_condition_the_selector(extractor):
    """Headers and table rows are layout, not assertions.

    The answer below is shaped the way the extractor actually segments granite's output --
    an ATX header, a header bullet and leaf bullets. A first version of this case put three
    bare lines in a row and the extractor returned them as ONE claim, flagged as a table row,
    swallowing the real sentence. That is pre-existing behaviour shared with Pass N and every
    scored artifact of the phase, so the case was fixed, not the extractor.
    """
    answer = ("## Limits and quotas.\n"
              "Limits and quotas by service:\n"
              "- Amazon EKS supports 100 nodes per managed node group.\n"
              "- | Service | Limit |\n")
    out = extractor.split_claims(answer)
    assert out["genuine"], "the one real sentence must survive"
    assert all(not c.strip().startswith("#") for c in out["genuine"])
    assert all(c.count("|") < 2 for c in out["genuine"])
    assert len(out["claims"]) == len(out["artifact"])
    assert sum(out["artifact"]) >= 1, "the header/table row must be flagged, not dropped silently"


def test_an_empty_draft_yields_no_claims(extractor):
    out = extractor.split_claims("   ")
    assert out["claims"] == [] and out["genuine"] == []


# --------------------------------------------------------------- 4. the arm schema
def test_the_anchor_arm_is_named_baseline_repro(generation):
    assert generation.ARM_FOR_STAGE["draft"] == "baseline_repro", \
        "verify_summer_offline.py and compute_exp18_diagnosis.py locate the anchor by this name"
    assert generation.ARM_FOR_STAGE["regen"] == "claim_selected"


def test_results_doc_carries_a_scenario_per_config(generation, tmp_path):
    label = "granite4.1-8b"
    for arm in ("baseline_repro", "claim_selected"):
        (tmp_path / f"checkpoint__{label}__{arm}.json").write_text(json.dumps({
            "config_name": f"{arm} | {label}", "completed_ids": ["q001"],
            "results": [{"query_id": "q001", "scenario": arm, "answer": "x",
                         "retrieved_ids": ["c1"], "tokens": {"input": 10, "output": 3}}]}),
            encoding="utf-8")
    doc = generation.build_results_doc(tmp_path, label, {}, ["q001"])
    assert doc["experiment_id"].startswith("exp19b")
    assert set(doc["configs"]) == {f"baseline_repro | {label}", f"claim_selected | {label}"}
    for cfg in doc["configs"].values():
        assert "scenario" in cfg, "the arm schema is what every downstream scorer discovers by"


def test_regen_does_not_drop_the_draft_sanity_check(generation):
    """Caught by the smoke run: only --stage draft computes it, and regen rewrites the file.

    The check had run and been logged; it simply stopped existing in the artifact anyone would
    later read. A number that exists only in a terminal that has scrolled away is not evidence.
    """
    prior = {"draft_vs_exp18_identical": {"checked": 3, "identical": 1, "rate": 0.3333}}
    doc = generation.carry_forward({"draft_vs_exp18_identical": None}, prior)
    assert doc["draft_vs_exp18_identical"]["rate"] == 0.3333


def test_a_fresh_draft_check_is_never_overwritten_by_a_stale_one(generation):
    fresh = {"draft_vs_exp18_identical": {"checked": 3, "identical": 3, "rate": 1.0}}
    doc = generation.carry_forward(dict(fresh), {"draft_vs_exp18_identical": {"rate": 0.0}})
    assert doc["draft_vs_exp18_identical"]["rate"] == 1.0


def test_smoke_output_is_invisible_to_shape_based_discovery(generation):
    """A 3-query smoke has no faithfulness rows; discovered, it would redden the suite."""
    results_root = PROJECT_ROOT / "experiments" / "results"
    smoke = generation.out_dir(smoke=True)
    real = generation.out_dir(smoke=False)
    assert smoke != real
    assert smoke.parent == real, "smoke lives INSIDE the experiment dir, not beside it"
    assert smoke.parent.parent == results_root
    assert smoke not in list(results_root.iterdir() if results_root.exists() else [])


# --------------------------------------------------------------- 5. the pre-registered analysis
def test_the_tost_band_is_the_pre_existing_exp17_effect(stats):
    assert stats.TOST_BAND == 0.081
    assert stats.TOST_ALPHA == 0.05
    assert stats.BASELINE_ARM == "baseline_repro" and stats.ARM == "claim_selected"
    assert stats.VERIFIERS == ["small", "base", "hhem"]


def test_the_bh_family_is_declared_and_says_what_it_means(stats):
    fam = stats.declared_family()
    assert fam["size"] == 1 and fam["scope"] == "per_verifier"
    assert "p_raw" in fam["note"], \
        "a family of one must SAY that BH is the identity instead of implying a correction"


def test_bh_of_a_single_p_value_is_that_p_value(stats):
    assert stats.bh_adjust([0.037]) == [0.037]


def test_bh_still_corrects_when_a_family_is_larger(stats):
    """The helper is general on purpose: a family of one is a declaration, not a shortcut."""
    out = stats.bh_adjust([0.01, 0.04])
    assert out[1] == pytest.approx(0.04) and out[0] == pytest.approx(0.02)


def test_the_selection_bound_is_never_used_as_a_denominator(stats):
    """exp18 pre-registered the bound as motivation only; a percent-of-headroom is fabricated."""
    src = (PROJECT_ROOT / "scripts" / "compute_exp19b_stats.py").read_text(encoding="utf-8")
    from conftest import code_only
    assert "0.5834" not in code_only(src)
    assert "frac_of_headroom" not in code_only(src)


# --------------------------------------------------------------- 6. hygiene registration
def test_the_exp19b_selector_is_registered_with_the_hygiene_guard():
    guard = (PROJECT_ROOT / "tests" / "test_selector_hygiene.py").read_text(encoding="utf-8")
    assert "select_exp19b_evidence.py" in guard, \
        "the script that actually picks the evidence for the SCORED arm must be guarded"
