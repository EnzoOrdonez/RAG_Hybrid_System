"""exp19b — the supervised pipeline, driven with a mock runner.

Seccion de Claude Code — 2026-08-21 14:25 (hora local).

The pipeline is ~6.6 h of GPU that has to stay inside one warmed Ollama session, and the
expensive failure is not a crash: it is finishing successfully with two arms generated in
different generator states. Measured 2026-08-21: an Ollama restart changes the answer to a
byte-identical prompt (q001 jaccard-5gram 0.0705). So the properties pinned here are:

  1. the direct draft replay sits between `select` and `regen`;
  2. any replay mismatch stops the pipeline and NOTHING gets scored;
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


def test_the_draft_replay_gate_sits_between_select_and_regen(pipe, stages):
    n = names(stages)
    assert n.index("select") < n.index("draft_replay_check") < n.index("regen")
    assert n.index("draft_replay_check") == n.index("regen") - 1, \
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


# --------------------------------------------------------- explicit artifact-backed resumption
def test_start_from_skips_completed_stages(pipe, stages, tmp_path):
    resumed = pipe.stages_from(stages, "select", tmp_path)

    assert names(resumed)[0] == "select"
    assert "draft" not in names(resumed) and "extract" not in names(resumed)


@pytest.mark.parametrize("start_from", ["draft_replay_check", "regen"])
def test_start_from_gate_or_regen_lists_every_missing_artifact(
        pipe, stages, tmp_path, start_from):
    with pytest.raises(ValueError) as exc:
        pipe.stages_from(stages, start_from, tmp_path)

    message = str(exc.value)
    assert start_from in message
    for required in pipe.RESUME_ARTIFACTS:
        assert required in message


def test_start_from_regen_rechecks_replay_gate_when_artifacts_exist(pipe, stages, tmp_path):
    for required in pipe.RESUME_ARTIFACTS:
        (tmp_path / required).write_text("{}", encoding="utf-8")

    resumed = pipe.stages_from(stages, "regen", tmp_path)

    assert names(resumed)[:2] == ["draft_replay_check", "regen"]
    assert "draft" not in names(resumed) and "select" not in names(resumed)


# ------------------------------------------------------------------ 2. the direct replay gate
def _archived_rows():
    return [{"query_id": f"q{i:03d}", "answer": f"answer {i}"} for i in range(1, 6)]


def test_five_identical_draft_replays_pass(pipe):
    expected = {row["query_id"]: row["answer"] for row in _archived_rows()}
    ok, msg, detail = pipe.draft_replay_check(
        _archived_rows(), generate_fn=lambda qid: expected[qid])

    assert ok and "5/5 bit-identical" in msg
    assert detail == {"checked": 5, "identical": 5, "mismatched_qids": []}


def test_one_different_draft_replay_is_loud(pipe):
    expected = {row["query_id"]: row["answer"] for row in _archived_rows()}
    expected["q003"] = "different"
    ok, msg, detail = pipe.draft_replay_check(
        _archived_rows(), generate_fn=lambda qid: expected[qid])

    assert not ok
    assert pipe.STATE_CHANGED_MARKER in msg and "4/5" in msg and "nothing was scored" in msg
    assert detail["mismatched_qids"] == ["q003"]


def test_a_state_change_scores_absolutely_nothing(pipe, stages):
    """The expensive failure is a pipeline that finishes and reports a number anyway."""
    ran = []
    code, report = pipe.run_pipeline(
        stages, runner=lambda argv: (ran.append(argv), 0)[1],
        fingerprint_fn=lambda: "informational-only",
        replay_check_fn=lambda: (
            False, f"{pipe.STATE_CHANGED_MARKER}: 4/5; nothing was scored",
            {"checked": 5, "identical": 4, "mismatched_qids": ["q003"]}),
        log=lambda m: None)

    assert code == pipe.EXIT_STATE_CHANGED
    done = [r["stage"] for r in report]
    assert "draft" in done and "select" in done
    for scoring in ("regen", "pass_n_small", "arm_stats_hhem", "primary_tost"):
        assert scoring not in done, f"{scoring} ran after the generator state changed"


def test_a_stable_state_runs_the_whole_pipeline(pipe, stages):
    code, report = pipe.run_pipeline(
        stages, runner=lambda argv: 0, fingerprint_fn=lambda: "same",
        replay_check_fn=lambda: (
            True, "draft replay 5/5 bit-identical",
            {"checked": 5, "identical": 5, "mismatched_qids": []}),
        log=lambda m: None)
    assert code == pipe.EXIT_OK
    assert [r["stage"] for r in report][-1] == "verify_offline"
    assert all(r["ok"] for r in report)


# ------------------------------------------------------------------ 3. failure propagation
def test_a_failing_stage_stops_everything_after_it(pipe, stages):
    def runner(argv):
        return 1 if "select_exp19b_evidence.py" in " ".join(argv) else 0

    code, report = pipe.run_pipeline(stages, runner=runner,
                                     fingerprint_fn=lambda: "same",
                                     replay_check_fn=lambda: (True, "ok", {}),
                                     log=lambda m: None)
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
