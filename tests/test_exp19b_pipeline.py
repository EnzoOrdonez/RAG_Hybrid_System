"""exp19b — the supervised pipeline, driven with a mock runner.

Seccion de Claude Code — 2026-08-21 14:25 (hora local).

The pipeline is ~6.6 h of GPU that has to stay inside one warmed Ollama session, and the
expensive failure is not a crash: it is finishing successfully with two arms generated in
different generator states. Measured 2026-08-21: an Ollama restart changes the answer to a
byte-identical prompt (q001 jaccard-5gram 0.0705). So the properties pinned here are:

  1. the fingerprint re-check sits between `select` and `regen` -- the CPU stretch where a
     machine sleeps or a driver resets is exactly there;
  2. a changed fingerprint stops the pipeline and NOTHING gets scored;
  3. a failing stage stops everything after it, with a non-zero exit;
  4. a fresh draft implies a fresh selection, so their checkpoints can never be mixed.

Nothing here starts a server or a subprocess: both are injected.

Run: pytest tests/test_exp19b_pipeline.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def pipe():
    path = PROJECT_ROOT / "scripts" / "run_exp19b_pipeline.py"
    assert path.exists(), "scripts/run_exp19b_pipeline.py missing"
    spec = importlib.util.spec_from_file_location("exp19b_pipeline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stages(pipe):
    return pipe.build_stages(py="python")


def names(stages):
    return [n for n, _ in stages]


# ------------------------------------------------------------------ 1. the order is the contract
def test_generation_comes_before_any_scoring(pipe, stages):
    n = names(stages)
    assert n.index("draft") < n.index("extract") < n.index("select") < n.index("regen")
    assert n.index("regen") < n.index("pass_n_small"), "scoring may not precede the second arm"


def test_the_fingerprint_gate_sits_between_select_and_regen(pipe, stages):
    n = names(stages)
    assert n.index("select") < n.index("fingerprint_recheck") < n.index("regen")
    assert n.index("fingerprint_recheck") == n.index("regen") - 1, \
        "nothing may run between the gate and the arm it protects"


def test_all_three_verifiers_are_scored(pipe, stages):
    n = names(stages)
    for v in pipe.VERIFIERS:
        assert f"arm_stats_{v}" in n and f"diagnosis_{v}" in n
    assert "pass_n_small" in n and "pass_n_base" in n and "grounding_hhem" in n


def test_the_primary_and_the_offline_verifier_come_last(pipe, stages):
    n = names(stages)
    assert n.index("primary_tost") < n.index("verify_offline")
    assert n[-1] == "verify_offline"
    assert n.index("guards") < n.index("primary_tost")


def test_a_fresh_draft_always_implies_a_fresh_selection(pipe, stages):
    stage_argv = dict(stages)
    assert "--no-resume" in stage_argv["draft"]
    assert "--no-cache" in stage_argv["draft"]
    assert "--no-resume" in stage_argv["select"], \
        "select must not mix claims from a fresh draft with an older selection checkpoint"


# ------------------------------------------------------------------ 2. the gate
def test_an_unchanged_fingerprint_passes_the_gate(pipe):
    ok, msg = pipe.fingerprint_gate("abc", "abc")
    assert ok and "unchanged" in msg


def test_a_changed_fingerprint_is_loud_and_names_the_marker(pipe):
    ok, msg = pipe.fingerprint_gate("abc", "xyz")
    assert not ok
    assert pipe.STATE_CHANGED_MARKER in msg and "abc" in msg and "xyz" in msg
    assert "scored" in msg, "the message must say that nothing was scored"


def test_a_state_change_scores_absolutely_nothing(pipe, stages):
    """The expensive failure is a pipeline that finishes and reports a number anyway."""
    ran = []
    fps = iter(["state_A", "state_B"])          # after draft, then at the gate
    code, report = pipe.run_pipeline(
        stages, runner=lambda argv: (ran.append(argv), 0)[1],
        fingerprint_fn=lambda: next(fps), log=lambda m: None)

    assert code == pipe.EXIT_STATE_CHANGED
    done = [r["stage"] for r in report]
    assert "draft" in done and "select" in done
    for scoring in ("regen", "pass_n_small", "arm_stats_hhem", "primary_tost"):
        assert scoring not in done, f"{scoring} ran after the generator state changed"


def test_a_stable_state_runs_the_whole_pipeline(pipe, stages):
    code, report = pipe.run_pipeline(
        stages, runner=lambda argv: 0, fingerprint_fn=lambda: "same",
        log=lambda m: None)
    assert code == pipe.EXIT_OK
    assert [r["stage"] for r in report][-1] == "verify_offline"
    assert all(r["ok"] for r in report)


# ------------------------------------------------------------------ 3. failure propagation
def test_a_failing_stage_stops_everything_after_it(pipe, stages):
    def runner(argv):
        return 1 if "select_exp19b_evidence.py" in " ".join(argv) else 0

    code, report = pipe.run_pipeline(stages, runner=runner,
                                     fingerprint_fn=lambda: "same", log=lambda m: None)
    assert code == pipe.EXIT_STAGE_FAILED
    done = [r["stage"] for r in report]
    assert done[-1] == "select" and not report[-1]["ok"]
    assert "regen" not in done and "primary_tost" not in done


def test_exit_codes_are_distinct_so_a_wrapper_can_tell_them_apart(pipe):
    assert len({pipe.EXIT_OK, pipe.EXIT_STAGE_FAILED, pipe.EXIT_STATE_CHANGED}) == 3
    assert pipe.EXIT_OK == 0


def test_the_summary_states_that_nothing_was_scored_on_a_state_change(pipe):
    text = pipe.render_summary(pipe.EXIT_STATE_CHANGED,
                               [{"stage": "draft", "ok": True, "seconds": 120}])
    assert pipe.STATE_CHANGED_MARKER in text and "nothing was scored" in text


# ------------------------------------------------------------------ 4. the launcher itself
def test_the_launcher_delegates_and_documents_the_exit_codes():
    ps1 = (PROJECT_ROOT / "scripts" / "launch_exp19b_full.ps1").read_text(encoding="utf-8")
    assert "run_exp19b_pipeline.py" in ps1, "the .ps1 must not re-implement the stage list"
    assert "RUNTIME_STATE_CHANGED" in ps1
    for warning in ("ENCHUFADO", "standby-timeout-ac", "NO reinicies Ollama"):
        assert warning in ps1, f"preflight must warn about {warning}"
    assert "granite4.1:8b" in ps1, "preflight must check the generator is present"
    assert "$env:PYTHONUTF8" in ps1
