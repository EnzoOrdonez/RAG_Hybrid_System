"""Controlled cohort contracts, using synthetic work and telemetry only."""
import json
import os
import subprocess
import sys

import pytest

from scripts import measure_interview_gate as gate
from scripts import observe_interview_gate as observe


def sample(t=0, **changes):
    return dict(monotonic_s=t, ac=True, scheme=observe.BALANCED,
                overlay=observe.BEST_PERFORMANCE, cpu_percent=2,
                gpu={'utilization.gpu': '2', 'temperature.gpu': '90',
                     'clocks_event_reasons.sw_power_cap': 'Active'},
                ram_available_bytes=100, processes=[], errors=[], **changes)


def test_subset_manifest_and_cli_plan_do_not_expand_to_120(tmp_path):
    protocol = {'systems': ['hybrid'], 'queries': [{}] * 20}
    gate.initialize_fresh(tmp_path, protocol)
    assert gate.read_json(tmp_path / 'source-manifest.json')['total_attempts'] == 40
    result = subprocess.run([sys.executable, gate.__file__, 'plan', '--output', str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert len(plan['pending']) == 40
    assert {s for s, _, _ in plan['pending']} == {'hybrid'}
    assert len(plan['report']['cells']) == 2
    with pytest.raises(ValueError, match='identity'):
        gate.initialize_fresh(tmp_path, dict(protocol, systems=['lexical']))


@pytest.mark.parametrize('systems', [[], ['hybrid', 'hybrid'], ['unknown']])
def test_invalid_selection_is_rejected(systems):
    with pytest.raises(ValueError):
        gate.selected_systems({'systems': systems})


def test_control_invalid_preserves_success_but_excludes_latency_and_consumes_slot():
    rows = [dict(system='hybrid', phase='cold', index=i, consumes_slot=True,
                 status=status, elapsed_s=duration, conditions_invalid=invalid)
            for i, status, duration, invalid in [(0, 'success', 5, False),
                (1, 'success', 500, True), (2, 'error', 60, False),
                (3, 'aborted', None, False)]]
    cell = gate.summarize(rows, ('hybrid',))['cells'][0]
    assert cell['attempts'] == cell['completed_slots'] == 4
    assert cell['conditions_invalid'] == 1
    assert cell['failures'] == 2
    assert cell['p50_s'] == cell['p95_s'] == 5
    assert len(gate.pending(rows, ('hybrid',))) == 36


def test_power_loss_and_missing_telemetry_invalidate_but_throttling_does_not():
    assert observe.assess([sample(), sample(5)]) == []
    bad = sample(5)
    bad['ac'] = False
    assert 'ac_unavailable' in observe.assess([sample(), bad])
    bad['errors'] = ['gpu unavailable']
    assert 'telemetry_error' in observe.assess([sample(), bad])
    assert 'telemetry_gap' in observe.assess([sample(), sample(20)])


def test_admission_requires_full_idle_window_and_current_cpu():
    rows = [sample(t) for t in range(0, 61, 5)]
    assert observe.assess(rows, admission=True) == []
    assert 'idle_window_incomplete' in observe.assess(rows[:-1], admission=True)
    for row in rows:
        row['cpu_percent'] = 12
    assert 'idle_cpu' in observe.assess(rows, admission=True)
    current = observe.process_deltas(
        [{'Id': 1, 'ProcessName': 'app', 'CPU': 100.0, 'WorkingSet64': 1}],
        {1: 99.0}, 2, 4)
    assert current[0]['cpu_percent'] == 12.5  # delta / elapsed / logical CPUs


def test_external_overlay_and_sustained_load_pause():
    row = sample()
    row['processes'] = [{'pid': 7, 'name': 'NVIDIA Overlay', 'cpu_percent': 0}]
    assert 'prohibited_process' in observe.assess([row])
    rows = [sample(), sample(5)]
    for row in rows:
        row['processes'] = [{'pid': 8, 'name': 'unrelated', 'cpu_percent': 20}]
    assert 'external_cpu_load' in observe.assess(rows)
    assert observe.assess(rows, allowed_pids={8}) == []


def test_observer_persists_samples_and_error_without_losing_response(tmp_path):
    def unavailable():
        raise OSError('sensor failed')

    observer = observe.Observer(tmp_path / 'telemetry.jsonl', unavailable, interval=.01)
    row = gate.measure_attempt(tmp_path, dict(system='hybrid', phase='cold', index=0),
                               lambda: dict(status='success', answer='preserved'), observer=observer)
    assert row['status'] == 'success'
    assert row['answer'] == 'preserved'
    assert row['conditions_invalid'] is True
    assert 'telemetry_error' in row['control_reasons']
    assert gate.read_events(tmp_path / 'telemetry.jsonl')
    assert gate.summarize([row])['cells'][0]['p95_s'] is None


def test_parallel_model_and_missing_cpu_are_not_silent():
    row = sample(5)
    row['ollama_ps_api'] = {'models': [{'name': 'unrelated-model'}]}
    row['cpu_percent'] = None
    reasons = observe.assess([sample(), row])
    assert 'concurrent_model' in reasons
    assert 'cpu_telemetry_missing' in reasons


def test_admission_failure_prevents_any_inference(tmp_path, monkeypatch):
    gate.initialize_fresh(tmp_path, {'systems': ['hybrid'], 'model_digest': 'test-digest', 'controls': {'version': 1}})
    monkeypatch.setenv('CLOUDRAG_MODEL_DIGEST', 'test-digest')
    monkeypatch.setattr(gate, 'preflight', lambda _: None)

    def denied(_):
        raise RuntimeError('not idle')

    monkeypatch.setattr(observe, 'admission', denied)
    monkeypatch.setattr(gate, 'api', lambda *args: pytest.fail('No model call before admission'))
    with pytest.raises(RuntimeError, match='not idle'):
        gate.run(None, tmp_path)
    assert not list(tmp_path.glob('attempts/*/request.json'))
    assert not list(tmp_path.glob('unloads/*'))


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows reference telemetry')
def test_native_process_snapshot_reads_current_process_without_subprocess():
    own = next(r for r in observe.windows_processes() if r['Id'] == os.getpid())
    assert own['CPU'] is not None and own['CPU'] >= 0
    assert own['WorkingSet64'] > 0


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows reference telemetry')
def test_native_power_and_memory_getters_return_real_values():
    state = observe.windows_state()
    assert state['ram_total_bytes'] > state['ram_available_bytes'] > 0
    assert len(state['overlay']) == len(state['scheme']) == 36
    assert isinstance(state['ac'], bool)


def test_real_coordinator_runs_only_40_slots_and_resume_does_not_duplicate(tmp_path, monkeypatch):
    gate.initialize_fresh(tmp_path, {'systems': ['hybrid'], 'model_digest': 'test-digest'})
    monkeypatch.setenv('CLOUDRAG_MODEL_DIGEST', 'test-digest')
    monkeypatch.setattr(gate, 'preflight', lambda _: None)
    monkeypatch.setattr(gate, 'git', lambda *args: 'build')
    monkeypatch.setattr(gate, 'check_model', lambda: 'granite4.1:8b')
    monkeypatch.setattr(gate, 'api', lambda *args: {'models': []})
    invocations = []

    def synthetic_worker(command, **kwargs):
        system = command[command.index('--system') + 1]
        phase = command[command.index('--phase') + 1]
        indices = list(map(int, command[command.index('--indices') + 1:]))
        invocations.append((system, phase, indices))
        for index in ([-1] + indices if phase == 'warm' else indices):
            gate.measure_attempt(tmp_path, dict(system=system, phase=phase, index=index,
                                 warmup=index == -1, consumes_slot=True), lambda: {'status': 'success'})
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gate.subprocess, 'run', synthetic_worker)
    gate.run(None, tmp_path)
    assert len(invocations) == 21  # 20 fresh workers + one persistent warm worker
    assert invocations[-1] == ('hybrid', 'warm', list(range(20)))
    rows = gate.all_records(tmp_path)
    assert len(rows) == 41
    assert len(gate.completed_slots(rows)) == 40
    assert gate.pending(rows, ('hybrid',)) == []
    gate.run(None, tmp_path)
    assert len(invocations) == 21
    assert gate.all_records(tmp_path) == rows
