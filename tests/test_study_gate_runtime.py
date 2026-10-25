from datetime import datetime, timedelta, timezone
import json
import time
from types import SimpleNamespace

import pytest

from scripts import run_study_gate as gate


def test_real_mode_cannot_be_authorized_by_declared_booleans(tmp_path):
    with pytest.raises(ValueError, match="Real supervisor"):
        gate.run(
            tmp_path / "real",
            window=1,
            operator_zoom_active=True,
            screen_share_declared=True,
        )


def test_window_two_cannot_start_without_first(tmp_path):
    with pytest.raises(ValueError, match="Window 2"):
        gate.run(tmp_path / "orphan", window=2, dry_run=True)


def test_checkpoint_exists_before_adapter_and_interruption_is_terminal(tmp_path):
    root = tmp_path / "window"

    def call(slot):
        pending = json.loads((root / "attempts" / "001-started.json").read_text())
        assert pending["query_id"] == slot["query_id"]
        raise KeyboardInterrupt()

    result = gate.run(root, window=1, dry_run=True, adapter=call)
    assert result["status"] == "INVALID_TERMINAL"
    rows = json.loads((root / "attempts.json").read_text())
    assert len(rows) == 1 and not rows[0]["valid"] and rows[0]["elapsed_s"] is None
    with pytest.raises(ValueError, match="incomplete"):
        gate.run(tmp_path / "second", window=2, dry_run=True, previous=root)


def test_aggregate_rechecks_hashes_and_never_promotes_synthetic(tmp_path):
    root = tmp_path / "cohort"
    result = gate.run_cohort(root, dry_run=True)
    assert result["go_decision"] == "SYNTHETIC_NOT_GO"
    assert all(v["n"] == 60 for v in result["systems"].values())
    (root / "window-1" / "attempts" / "001-result.json").write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        gate.aggregate(root)


def test_identity_drift_between_windows_is_rejected(tmp_path):
    root = tmp_path / "first"
    gate.run(
        root,
        window=1,
        dry_run=True,
        preflight=lambda: dict(valid=True, synthetic=True, identity={"build": "a"}),
    )
    with pytest.raises(ValueError, match="Identity"):
        gate.run(
            tmp_path / "second",
            window=2,
            dry_run=True,
            previous=root,
            preflight=lambda: dict(valid=True, synthetic=True, identity={"build": "b"}),
        )


def test_adapter_timer_includes_durable_write_and_excludes_response_flush(
    tmp_path, monkeypatch
):
    ticks = [0.0]
    original = gate.atomic_json

    def write(path, payload):
        ticks[0] += 100 if str(path).endswith("-response.json") else 2
        original(path, payload)

    monkeypatch.setattr(gate, "atomic_json", write)

    def query(question):
        assert json.loads((tmp_path / "w1-001.json").read_text())["status"] == "running"
        ticks[0] += 3
        return SimpleNamespace(
            answer="answer",
            error=None,
            confidence="HIGH",
            sources=[],
            hallucination_report={"method": "nli"},
        )

    adapter = gate.make_app_adapter(
        {"queries": {"q": {"question": "text"}}},
        lambda _: SimpleNamespace(query=query),
        clock=lambda: ticks[0],
        evidence_root=tmp_path,
    )
    assert adapter(dict(query_id="q", condition="hybrid", attempt_id="w1-001")) == (
        5.0,
        True,
        None,
    )
    with pytest.raises(FileExistsError):
        adapter(dict(query_id="q", condition="hybrid", attempt_id="w1-001"))


def test_supervisor_bounds_a_blocked_call_and_reaps_its_worker(tmp_path):
    from scripts.study_gate_supervisor import Supervisor

    root = tmp_path / "worker"
    root.mkdir()
    with Supervisor(
        dict(synthetic=True, root=str(root)), call_seconds=0.2, preparation_seconds=15
    ) as supervisor:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            supervisor({"hang": True})
        assert time.monotonic() - started < 10
        assert not supervisor.process.is_alive()
        with pytest.raises(RuntimeError, match="Terminal"):
            supervisor({})


def test_supervisor_obeys_window_deadline_even_with_long_call_budget(tmp_path):
    from scripts.study_gate_supervisor import Supervisor

    root = tmp_path / "worker"
    root.mkdir()
    with Supervisor(
        dict(synthetic=True, root=str(root)), call_seconds=60, preparation_seconds=15
    ) as supervisor:
        supervisor.deadline = time.monotonic() + 0.2
        with pytest.raises(TimeoutError):
            supervisor({"hang": True})
        assert not supervisor.process.is_alive()


def test_slow_telemetry_cannot_extend_worker_execution(tmp_path):
    from scripts.study_gate_supervisor import Supervisor

    root = tmp_path / "worker"
    root.mkdir()
    with Supervisor(
        dict(synthetic=True, root=str(root)), call_seconds=1.2, preparation_seconds=15
    ) as supervisor:
        observed = []

        def slow_poll():
            time.sleep(0.5)
            observed.append(supervisor.process.is_alive())

        supervisor.poll = slow_poll
        with pytest.raises(TimeoutError):
            supervisor({"hang": True})
        assert observed == [False]


def test_device_policy_is_explicit_and_preserves_gpu_opt_in(monkeypatch):
    from src.ui.components.study_pipeline import configure_study_device

    monkeypatch.delenv("CLOUDRAG_DEMO_GPU", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    configure_study_device()
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == ""
    monkeypatch.setenv("CLOUDRAG_DEMO_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    configure_study_device()
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_durable_adapter_uses_actual_session_service(tmp_path, monkeypatch):
    from tests.study_helpers import configured
    from src.ui.components import study_service

    _, _, protocol = configured(tmp_path)
    calls = []
    original = study_service.answer

    def observed(session, *args, **kwargs):
        calls.append(session.store.purpose)
        return original(session, *args, **kwargs)

    monkeypatch.setattr(study_service, "answer", observed)
    response = SimpleNamespace(
        answer="answer",
        error=None,
        confidence="HIGH",
        sources=[],
        hallucination_report={"method": "nli"},
    )
    adapter = gate.make_app_adapter(
        protocol,
        lambda _: SimpleNamespace(query=lambda _: response),
        evidence_root=tmp_path / "gate",
    )
    _, valid, error = adapter(
        dict(query_id="q001", condition="hybrid", attempt_id="w1-001")
    )
    assert valid and error is None
    assert calls == ["technical"]


@pytest.mark.parametrize("problem", ["load", "resident", "process", "gap"])
def test_observed_preflight_rejects_bad_state(problem):
    from scripts.study_gate_environment import assess

    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    rows = [
        dict(
            monotonic_s=t,
            cpu_percent=0,
            errors=[],
            processes=[],
            gpu={"utilization.gpu": "0"},
            ollama_ps_api={
                "models": [dict(digest="d", expires_at=expiry, context_length=4096)]
            },
        )
        for t in range(0, 61, 5)
    ]
    if problem == "load":
        for row in rows:
            row["cpu_percent"] = 10
    elif problem == "resident":
        rows[-1]["ollama_ps_api"] = {"models": []}
    elif problem == "process":
        rows[-1]["processes"] = [dict(pid=999, name="chrome", cpu_percent=0)]
    else:
        rows[-1]["monotonic_s"] = 90
    assert assess(rows, {"model_digest": "d"}, admission=True)
