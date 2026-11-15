"""ADR-0034: archived admission and window slices are replayed at their own last observation.

Attempt 14 completed 140/140 gate calls and then failed its own verification: the last admission
sample (and the last sample of each window) was judged against the gate's closing clock, so an
Ollama residency lease that was healthy when sampled looked expired ~60 minutes later.
"""
from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_operator.i7_gate_host import gate_admission_receipt, hot_reasons, observed_at, verify
from scripts.study_operator.stimulus_host_evidence import assessment_config, sha
from tests.test_study_i7_gate import full_host_proof

BASE = datetime(2026, 11, 15, 7, 0, tzinfo=timezone.utc)
LEASE_S = 300


def advancing_clock_proof(lease_s=LEASE_S):
    """Give every host sample a wall clock that advances with monotonic_s and a fixed lease."""
    proof = full_host_proof()
    host = proof['host_evidence']
    # Admission and window rows are the same objects as the continuous samples.
    for sample in host['samples']:
        observed = BASE + timedelta(seconds=sample['monotonic_s'])
        sample['at'] = observed.isoformat()
        for model in sample['ollama_ps_api']['models']:
            model['expires_at'] = (observed + timedelta(seconds=lease_s)).isoformat()
    proof['host_admission_sha256'] = sha(gate_admission_receipt(host))
    return proof


def window(host, index):
    return next(w['samples'] for w in host['window_admissions'] if w['index'] == index)


def test_gate_longer_than_the_lease_verifies_when_each_slice_was_healthy():
    proof = advancing_clock_proof()
    host = proof['host_evidence']
    closing = host['samples'][-1]['monotonic_s']
    # The scenario of attempt 14: the admission lease ends before the closing clock + 180 s.
    assert host['admission_samples'][-1]['monotonic_s'] + LEASE_S < closing + 180
    assert verify(proof)['decision'] == 'GO'


@pytest.mark.parametrize('slice_name', ['admission', 21, 81])
def test_slice_is_judged_at_its_own_observation_not_at_the_closing_clock(slice_name):
    proof = advancing_clock_proof()
    host = proof['host_evidence']
    rows = host['admission_samples'] if slice_name == 'admission' else window(host, slice_name)
    config = assessment_config(proof['inventory'])
    began = host['preparation_started']
    closing = observed_at(host['samples'])
    # The defect: at the closing clock the slice's last lease looks expired.
    assert 'residency_lease' in hot_reasons(rows, config, began, reference_time=closing)
    assert hot_reasons(rows, config, began, reference_time=observed_at(rows)) == []


@pytest.mark.parametrize('slice_name, error', [('admission', 'GATE_TELEMETRY_REJECTED'),
                                               (21, 'WINDOW_PREFLIGHT_DIFFERS'),
                                               (81, 'WINDOW_PREFLIGHT_DIFFERS')])
def test_slice_only_admission_checks_still_run(slice_name, error):
    # Busy CPU during an idle window is checked only by the slice replay (admission=True).
    proof = advancing_clock_proof()
    host = proof['host_evidence']
    rows = host['admission_samples'] if slice_name == 'admission' else window(host, slice_name)
    for row in rows:
        row['cpu_percent'] = 50.0
    proof['host_admission_sha256'] = sha(gate_admission_receipt(host))
    with pytest.raises(ValueError, match=error):
        verify(proof)


@pytest.mark.parametrize('slice_name', ['admission', 21, 81])
def test_a_lease_already_short_when_sampled_is_still_rejected(slice_name):
    # Each slice row is also a non-last row of the continuous telemetry, judged at its own instant.
    proof = advancing_clock_proof()
    host = proof['host_evidence']
    rows = host['admission_samples'] if slice_name == 'admission' else window(host, slice_name)
    last = rows[-1]
    observed = datetime.fromisoformat(last['at'])
    last['ollama_ps_api']['models'][0]['expires_at'] = (observed + timedelta(seconds=60)).isoformat()
    proof['host_admission_sha256'] = sha(gate_admission_receipt(host))
    with pytest.raises(ValueError, match='GATE_TELEMETRY_REJECTED'):
        verify(proof)


def test_observed_at_returns_the_last_row_clock_and_requires_an_aware_clock():
    rows = [{'at': '2026-11-15T07:00:00+00:00'}, {'at': '2026-11-15T07:44:24+00:00'}]
    assert observed_at(rows) == datetime(2026, 11, 15, 7, 44, 24, tzinfo=timezone.utc).timestamp()
    with pytest.raises(ValueError, match='GATE_CLOCK_UNAWARE'):
        observed_at([{'at': '2026-11-15T07:44:24'}])
