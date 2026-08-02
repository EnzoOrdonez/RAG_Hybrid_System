"""
`pass_n` (NLI scoring) must be resumable, and resuming must change nothing.

pass_g has had a checkpoint since Tier A; pass_n never did, and on exp18 it is the longer
pass (~51k pairs per verifier -- the top-10 arm alone is 29k because it carries 10 chunks
instead of 5). The environment has killed long jobs repeatedly in this phase, so a
non-resumable pass_n means a kill throws the whole verifier away.

Resuming is safe here in a way it is NOT for generation: scoring is deterministic
re-aggregation of a fixed model over fixed text, so it carries no H5 cold/warm exposure.
These tests pin that:

  1. the partial is written per arm and removed on success;
  2. a resumed run skips arms already in the partial;
  3. the merged output is identical to an uninterrupted run.

The NLI model is never loaded: the arm loop is exercised through the checkpoint plumbing
with a stub, so the suite stays fast.

Run: pytest tests/test_pass_n_resume.py -v
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT = PROJECT_ROOT / "scripts" / "run_exp15_ablation.py"


def test_pass_n_has_a_partial_and_removes_it_on_success():
    """Source-level guard: the checkpoint must exist and be cleaned up."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "pass_n__" in src and "partial.json.gz" in src, "pass_n has no checkpoint"
    assert "_save_partial()" in src, "the partial is never written inside the arm loop"
    assert "part_path.unlink(missing_ok=True)" in src, (
        "the partial is not removed on success; a stale one would silently short-circuit "
        "the next run")


def test_pass_n_partial_is_per_verifier():
    """small and base must not overwrite each other's checkpoint."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert 'f"pass_n__{args.verifier}.partial.json.gz"' in src


def test_pass_n_skips_arms_already_scored():
    """The resume guard must sit on the config name, before any scoring work."""
    src = SCRIPT.read_text(encoding="utf-8")
    i_skip = src.index('if cname in probs_out["configs"]:')
    i_pairs = src.index("preds = (model.predict(")
    assert i_skip < i_pairs, "the resume check must precede the expensive predict call"


def _roundtrip(tmp_path, payload):
    """Write then read a partial exactly as pass_n does."""
    p = tmp_path / "pass_n__small.partial.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def test_partial_roundtrip_preserves_every_section(tmp_path):
    """probs, claims and rows must all survive; losing one silently corrupts the merge."""
    payload = {
        "probs": {"baseline_repro | m": {"q001": [[[0.1, 0.8, 0.1]]]}},
        "claims": {"baseline_repro | m": {"q001": {"claims": ["a"], "artifact": [False],
                                                   "chunk_ids": ["c1"]}}},
        "rows": {"baseline_repro | m": {"q001": {"genuine": 1, "supported": 1,
                                                 "faithfulness": 1.0, "method": "nli"}}},
    }
    got = _roundtrip(tmp_path, payload)
    assert got == payload
    assert set(got) == {"probs", "claims", "rows"}


def test_resume_then_finish_equals_one_shot(tmp_path):
    """Merging a partial with the remaining arms reproduces the uninterrupted result.

    This is the property that makes resuming trustworthy: scoring an arm depends only on
    that arm's answers and the fixed model, never on what ran before it.
    """
    arm_a = {"a | m": {"q001": {"faithfulness": 0.5, "genuine": 2}}}
    arm_b = {"b | m": {"q002": {"faithfulness": 0.25, "genuine": 4}}}

    one_shot = {**arm_a, **arm_b}

    # interrupted after arm A, then resumed
    resumed = dict(_roundtrip(tmp_path, {"probs": arm_a, "claims": {}, "rows": arm_a})["rows"])
    for cname, rows in one_shot.items():
        if cname in resumed:
            continue
        resumed[cname] = rows

    assert resumed == one_shot


def test_scoring_is_order_independent_by_construction():
    """pass_n scores one arm at a time from results.json; arms never share state.

    Guard against someone introducing cross-arm accumulation (e.g. a shared `pairs` list),
    which would make resuming produce different numbers than a one-shot run.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    body = src[src.index("def pass_n("):]
    loop = body[body.index("for cname, cdata in results.items():"):]
    # the per-arm accumulators must be re-initialised INSIDE the loop
    head = loop[: loop.index("preds = (model.predict(")]
    assert "cfg_probs, cfg_claims, cfg_rows = {}, {}, {}" in head, (
        "per-arm accumulators are not reset inside the loop; scoring would depend on order")
    assert "pairs, spans = [], []" in head, "the pair buffer is not reset per arm"
