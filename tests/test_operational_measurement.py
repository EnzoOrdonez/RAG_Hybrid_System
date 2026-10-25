"""Durability across actual worker death, without models or Ollama."""
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from filelock import FileLock, Timeout

from scripts import measure_interview_gate as gate


def test_failures_and_unknown_durations_never_enter_percentiles():
    rows = [dict(system="hybrid", phase="cold", index=i, warmup=False,
                 status="success", elapsed_s=float(i + 1)) for i in range(20)]
    rows += [dict(system="hybrid", phase="cold", index=4, warmup=False,
                  status="aborted", elapsed_s=None),
             dict(system="hybrid", phase="cold", index=21, warmup=False,
                  status="error", elapsed_s=.001)]
    result = gate.summarize(rows)["cells"][0]
    assert result["attempts"] == 22
    assert result["failures"] == 2
    assert result["p50_s"] == 10.5
    assert result["p95_s"] == pytest.approx(19.05)


def test_recovery_of_killed_worker_is_durable_and_idempotent(tmp_path):
    code = """
import sys, time
from pathlib import Path
from scripts.measure_interview_gate import measure_attempt
measure_attempt(Path(sys.argv[1]), dict(system='lexical', phase='cold', index=10,
    warmup=False), lambda: time.sleep(60), interval=.05)
"""
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)])
    worker_pid = None
    try:
        deadline = time.monotonic() + 15
        journal = None
        while time.monotonic() < deadline:
            journals = list(tmp_path.glob("attempts/*/events.jsonl"))
            if journals and len(gate.read_events(journals[0])) >= 2:
                journal = journals[0]
                break
            time.sleep(.02)
        assert journal is not None, "worker failed to persist heartbeat"
        worker_pid = gate.read_json(journal.with_name("request.json"))["worker_pid"]
        with pytest.raises(Timeout):
            gate.recover(tmp_path)
        os.kill(worker_pid, signal.SIGTERM)
        child.wait(timeout=10)
        before = journal.read_bytes()
        recovered = gate.recover(tmp_path)
        assert len(recovered) == 1
        row = recovered[0]
        assert row["status"] == "aborted"
        assert row["elapsed_s"] is None
        assert row["elapsed_lower_bound_s"] > 0
        assert gate.recover(tmp_path) == []
        assert journal.read_bytes() == before
        assert gate.summarize([row])["cells"][2]["p95_s"] is None
    finally:
        if worker_pid and child.poll() is None:
            os.kill(worker_pid, signal.SIGTERM)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


def test_completed_failure_is_not_retried_and_warmup_is_separate(tmp_path):
    gate.measure_attempt(tmp_path, dict(system="semantic", phase="warm", index=0,
                         warmup=False), lambda: {"status": "error", "error": "timeout"})
    gate.measure_attempt(tmp_path, dict(system="semantic", phase="warm", index=-1,
                         warmup=True), lambda: {"status": "success"})
    rows = gate.local_records(tmp_path)
    assert ("semantic", "warm", 0) in gate.completed_slots(rows)
    assert ("semantic", "warm", -1) not in gate.completed_slots(rows)
    report = gate.summarize(rows)
    assert report["warmups"]["attempts"] == 1
    assert report["cells"][5]["attempts"] == 1
    assert report["cells"][5]["failures"] == 1
    assert report["cells"][5]["p50_s"] is None


def test_partial_tail_is_preserved_but_complete_corruption_is_rejected(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event":"start"}\n{"event":')
    assert gate.read_events(path) == [{"event": "start"}]
    path.write_bytes(b'{broken}\n')
    with pytest.raises(ValueError):
        gate.read_events(path)


def test_completed_result_survives_recovery_and_result_is_exclusive(tmp_path):
    row = gate.measure_attempt(tmp_path, dict(system="hybrid", phase="cold", index=0,
                               warmup=False), lambda: {"status": "success", "answer": "ok"})
    assert gate.recover(tmp_path) == []
    assert gate.local_records(tmp_path) == [row]
    result = next(tmp_path.glob("attempts/*/result.json"))
    with pytest.raises(FileExistsError):
        gate.write_new(result, {"replacement": True})
    assert json.loads(result.read_text(encoding="utf-8"))["answer"] == "ok"


def test_active_coordinator_blocks_second_run(tmp_path):
    with FileLock(str(tmp_path / "run.lock"), timeout=0):
        with pytest.raises(Timeout):
            with gate.coordinator_lock(tmp_path):
                pytest.fail("concurrent coordinator admitted")


def test_killed_coordinator_does_not_allow_recovering_live_worker(tmp_path):
    child_code = """
import sys, time
from pathlib import Path
from scripts.measure_interview_gate import measure_attempt
measure_attempt(Path(sys.argv[1]), dict(system='lexical', phase='cold', index=10,
    warmup=False), lambda: time.sleep(60), interval=.05)
"""
    parent_code = """
import sys, time, subprocess
from pathlib import Path
from scripts.measure_interview_gate import coordinator_lock, write_new
root = Path(sys.argv[1])
with coordinator_lock(root):
    child = subprocess.Popen([sys.executable, '-c', sys.argv[2], str(root)])
    write_new(root / 'child.json', dict(pid=child.pid, parent_pid=__import__('os').getpid()))
    child.wait()
"""
    parent = subprocess.Popen([sys.executable, "-c", parent_code, str(tmp_path), child_code])
    child_pid = None
    parent_pid = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            pid_file = tmp_path / "child.json"
            if pid_file.exists():
                parent_pid = gate.read_json(pid_file)["parent_pid"]
            journals = list(tmp_path.glob("attempts/*/events.jsonl"))
            if parent_pid and journals and len(gate.read_events(journals[0])) >= 2:
                child_pid = gate.read_json(journals[0].with_name("request.json"))["worker_pid"]
                break
            time.sleep(.02)
        assert child_pid and journals
        os.kill(parent_pid, signal.SIGTERM)
        parent.wait(timeout=10)
        with gate.coordinator_lock(tmp_path):
            with pytest.raises(Timeout):
                gate.recover(tmp_path)
    finally:
        if parent.poll() is None:
            if parent_pid:
                os.kill(parent_pid, signal.SIGTERM)
            else:
                parent.kill()
            parent.wait(timeout=10)
        if child_pid:
            os.kill(child_pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while True:
        try:
            assert len(gate.recover(tmp_path)) == 1
            break
        except Timeout:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.02)


def test_death_before_first_heartbeat_keeps_unknown_duration(tmp_path):
    directory = tmp_path / "attempts" / "physical-id"
    gate.write_new(directory / "request.json", dict(system="hybrid", phase="cold",
                   index=0, warmup=False, attempt_id="physical-id"))
    row = gate.recover(tmp_path)[0]
    assert row["elapsed_s"] is row["elapsed_lower_bound_s"] is None
    assert gate.pending([row])[0] == ("hybrid", "cold", 0)


def test_duplicate_finalized_slot_is_rejected():
    row = dict(system="hybrid", phase="cold", index=0, warmup=False, status="error")
    with pytest.raises(ValueError, match="Duplicate"):
        gate.completed_slots([row, row])


def test_changed_application_or_uncommitted_source_blocks_resume(monkeypatch):
    monkeypatch.setattr(gate, "git", lambda *args: "modified" if args[0] == "diff" else "head")
    with pytest.raises(ValueError, match="source changed"):
        gate.validate_app_identity("baseline")
    monkeypatch.setattr(gate, "git", lambda *args: "head" if args[0] == "rev-parse" else "")
    monkeypatch.setenv("CLOUDRAG_BUILD_ID", "wrong")
    with pytest.raises(ValueError, match="current HEAD"):
        gate.validate_app_identity("baseline")


def test_plan_reads_legacy_and_writes_nothing(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    query = {"question": "q", "query_id": "q1"}
    gate.write_new(source / "protocol.json", dict(seed=42, llm_cache=False, queries=[query] * 20))
    gate.write_new(source / "hybrid-cold-00.json", dict(query=query, system="hybrid", phase="cold",
                   index=0, warmup=False, success=True, elapsed_s=10))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = subprocess.run([sys.executable, str(gate.__file__), "plan", "--source", str(source),
                             "--output", str(tmp_path / "new")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["pending"]) == 119
    assert not (tmp_path / "new").exists()
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


def test_source_evidence_mutation_is_detected(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    gate.write_new(source / "protocol.json", dict(seed=42, llm_cache=False, queries=[{}] * 20))
    root = tmp_path / "resume"
    gate.initialize(source, root)
    (source / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        gate.all_records(root)


def test_warm_resume_loads_once_and_records_new_warmup_separately(tmp_path, monkeypatch):
    queries = [{"question": f"q{i}"} for i in range(20)]
    gate.write_new(tmp_path / "source-manifest.json", {"protocol": {"queries": queries, "build_id": "old"}})
    monkeypatch.setenv("CLOUDRAG_BUILD_ID", "runner")
    monkeypatch.setattr(gate, "check_model", lambda: "model")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda=None)))
    calls, loads = [], []

    class Response:
        def model_dump(self, **kwargs):
            return dict(error=None, answer="ok", confidence="MEDIUM", hallucination_report={"method": "nli"})

    def query(text):
        calls.append(text)
        return Response()

    pipeline = SimpleNamespace(query=query,
        llm=SimpleNamespace(cache_enabled=False, seed=42, timeout=60, max_retries=1,
                            num_ctx=4096, model_digest="digest"),
        config=SimpleNamespace(model_dump=lambda **kwargs: {"name": "semantic"}),
        hybrid_index=SimpleNamespace(deployment_manifest_sha256="manifest"))

    def load(*args, **kwargs):
        loads.append(True)
        return pipeline

    monkeypatch.setitem(sys.modules, "src.ui.components.index_loader",
                        SimpleNamespace(load_pipeline=load, load_hybrid_index=lambda: object()))
    gate.worker(tmp_path, "semantic", "warm", [17, 18, 19])
    assert calls == ["q0", "q17", "q18", "q19"]
    assert len(loads) == 1
    rows = gate.local_records(tmp_path)
    assert len(rows) == 4
    assert gate.summarize(rows)["cells"][5]["completed_slots"] == 3
    assert gate.summarize(rows)["warmups"]["attempts"] == 1


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1])
def test_invalid_successful_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="duration"):
        gate.percentile([duration], .95)


def test_fresh_cohort_is_empty_idempotent_and_rejects_drift(tmp_path):
    protocol = dict(environment={"driver": "616.64"}, queries=[{"question": "q"}] * 20)
    root = tmp_path / "fresh"
    gate.initialize_fresh(root, protocol)
    original = (root / "source-manifest.json").read_bytes()
    assert gate.all_records(root) == []
    assert len(gate.pending(gate.all_records(root))) == 120
    gate.initialize_fresh(root, protocol)
    assert (root / "source-manifest.json").read_bytes() == original
    with pytest.raises(ValueError, match="identity"):
        gate.initialize_fresh(root, dict(protocol, environment={"driver": "610.62"}))
    assert (root / "source-manifest.json").read_bytes() == original


def test_fresh_abort_consumes_slot_once_and_never_enters_percentiles(tmp_path):
    gate.initialize_fresh(tmp_path, {"queries": []})
    gate.write_new(tmp_path / "attempts" / "killed" / "request.json",
                   dict(system="hybrid", phase="cold", index=0, warmup=False,
                        consumes_slot=True))
    gate.recover(tmp_path)
    rows = gate.all_records(tmp_path)
    assert ("hybrid", "cold", 0) not in gate.pending(rows)
    assert len(gate.pending(rows)) == 119
    assert gate.recover(tmp_path) == []
    cell = gate.summarize(rows)["cells"][0]
    assert cell["completed_slots"] == cell["failures"] == cell["aborted"] == 1
    assert cell["p95_s"] is None


def test_environment_drift_after_response_is_persisted_as_failure(tmp_path):
    def changed():
        raise ValueError("environment changed")

    row = gate.measure_attempt(tmp_path, dict(system="hybrid", phase="cold", index=0),
                               lambda: {"status": "success", "answer": "actual response"},
                               validate_after=changed)
    assert row["status"] == "error"
    assert row["environment_invalid"] is True
    assert row["answer"] == "actual response"
    assert gate.summarize([row])["cells"][0]["p95_s"] is None
    assert gate.local_records(tmp_path) == [row]


def test_fresh_environment_guard_rejects_driver_and_commit_drift(monkeypatch):
    expected = {"gpu": "616.64", "commit": "first"}
    monkeypatch.setattr(gate, "environment_identity", lambda: expected)
    gate.check_environment({"environment": dict(expected)})
    for changed in ({"gpu": "610.62", "commit": "first"},
                    {"gpu": "616.64", "commit": "second"}):
        with pytest.raises(ValueError, match="environment"):
            gate.check_environment({"environment": changed})


def test_fresh_cli_plan_requires_no_legacy_source(tmp_path):
    gate.initialize_fresh(tmp_path, {"queries": []})
    result = subprocess.run([sys.executable, str(gate.__file__), "plan", "--output", str(tmp_path)],
                             capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["pending"]) == 120


def test_fresh_refuses_legacy_output_and_checkout(tmp_path):
    gate.write_new(tmp_path / "source-manifest.json", {"source": "historical"})
    with pytest.raises(ValueError, match="identity"):
        gate.initialize_fresh(tmp_path, {})
    with pytest.raises(ValueError, match="outside"):
        gate.initialize_fresh(gate.PROJECT / "not-created", {})


def test_invocation_distinguishes_first_run_from_resume(tmp_path, monkeypatch):
    gate.initialize_fresh(tmp_path, {"model_digest": "digest"})
    monkeypatch.setenv("CLOUDRAG_MODEL_DIGEST", "digest")
    monkeypatch.setattr(gate, "preflight", lambda protocol: None)
    monkeypatch.setattr(gate, "SYSTEMS", ())  # no model work; exercise real invocation persistence
    gate.run(None, tmp_path)
    first = next((tmp_path / "invocations").glob("*.json"))
    assert gate.read_json(first)["interrupted_cohort"] is False
    gate.measure_attempt(tmp_path, dict(system="hybrid", phase="cold", index=0,
                         consumes_slot=True), lambda: {"status": "error"})
    gate.run(None, tmp_path)
    second = next(p for p in (tmp_path / "invocations").glob("*.json") if p != first)
    assert gate.read_json(second)["interrupted_cohort"] is True
    assert len(gate.local_records(tmp_path)) == 1
