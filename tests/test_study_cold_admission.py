import copy

import pytest

from scripts.study_operator.cold_admission import assess_cold


CONFIG = dict(service_mode='fresh_runner', service_boot_id='fixture', model_digest='a'*64)


def rows():
    return [dict(monotonic_s=100+i*5, at='2026-10-08T00:00:00+00:00', errors=[], cpu_percent=1,
        gpu={'utilization.gpu':'0'}, gpu_pids=[], processes=[], ollama_ps_api=dict(models=[]),
        service_state=dict(schema_version=1, mode='fresh_runner', phase='STARTING', sequence=1,
            boot_id='fixture', request_id=None, deadline_monotonic_s=None, written_monotonic_s=50))
        for i in range(13)]


def test_cold_admission_accepts_only_genuine_empty_initial_service_without_rewriting():
    values = rows()
    before = copy.deepcopy(values)
    assert assess_cold(values, CONFIG, preparation_started=100, admission=True) == []
    assert values == before


@pytest.mark.parametrize('change', ['sequence', 'boot', 'resident', 'expired', 'future'])
def test_altered_history_or_clock_rejects_cold_admission(change):
    values = rows()
    row = values[-1]
    if change == 'sequence':
        row['service_state']['sequence'] = 5
    elif change == 'boot':
        row['service_state']['boot_id'] = 'other'
    elif change == 'resident':
        row['ollama_ps_api']['models'] = [dict(digest='a'*64)]
    elif change == 'expired':
        row['monotonic_s'] = 1001
    else:
        row['service_state']['written_monotonic_s'] = 1000
    assert 'cold_history_invalid' in assess_cold(values, CONFIG, preparation_started=100, admission=True)


def test_foreign_gpu_cpu_prohibited_process_and_gap_checks_are_preserved():
    values = rows()
    for row in values:
        row['gpu_pids'] = [901]
        row['processes'] = [dict(pid=901, name='chrome', cpu_percent=20)]
    values[-1]['monotonic_s'] += 20
    reasons = assess_cold(values, CONFIG, preparation_started=100, admission=True)
    assert {'foreign_gpu_process','prohibited_process','external_cpu_load','telemetry_gap'} <= set(reasons)


def test_cold_exception_does_not_mask_bad_later_transition():
    values = rows()
    values[-1]['service_state'].update(phase='RESIDENT', sequence=5, request_id='b'*32, runner_pids=[])
    reasons = assess_cold(values, CONFIG, preparation_started=100)
    assert 'service_transition' in reasons and 'model_residency' in reasons


def test_insufficient_idle_observations_and_missing_bound_are_rejected():
    assert 'idle_window_incomplete' in assess_cold(rows()[:2], CONFIG, preparation_started=100, admission=True)
    with pytest.raises(ValueError):
        assess_cold(rows(), CONFIG, preparation_started=float('nan'))


def test_initial_state_cannot_reappear_after_a_generation_transition():
    values = rows()
    values[5]['service_state'].update(phase='RESETTING', request_id='b'*32,
        deadline_monotonic_s=130, written_monotonic_s=125)
    assert 'cold_history_invalid' in assess_cold(values, CONFIG, preparation_started=100)
