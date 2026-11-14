"""ADR-0033: the long status polls of the stimulus and the gate tolerate a longer transport outage.

In attempt 13 the IAP SSH transport to the GPU VM failed for about five minutes in boot 3 of the
stimulus while the Google Cloud API still answered. `stimulus-status` retried after 15 s and 45 s,
gave up, and the series became terminal. These polls only read; the guest job keeps running and is
bounded by its own limit (7200 s / 9000 s) and by the 600 s acknowledgement window (ADR-0031).
"""
import pytest

from scripts.study_operator.lifecycle import (
    LONG_POLL_OPERATIONS,
    LONG_POLL_TRANSPORT_RETRY_DELAYS_S,
    RETRYABLE_BRIDGE_OPERATIONS,
    TRANSPORT_RETRY_DELAYS_S,
)
from scripts.study_operator.policy import TransportUnavailable
from tests.test_study_i7_attempt10_fixes import OK, _operator, _transport


def test_long_polls_are_the_stimulus_and_gate_status_only():
    assert LONG_POLL_OPERATIONS == frozenset({'stimulus-status', 'gate-status'})
    assert LONG_POLL_OPERATIONS <= RETRYABLE_BRIDGE_OPERATIONS


LONG = [15, 45, 60, 60, 60, 60, 60]


def test_long_poll_schedule_extends_the_short_one_and_stays_bounded():
    assert LONG_POLL_TRANSPORT_RETRY_DELAYS_S == tuple(LONG)
    # Same first attempts as the short schedule: no outage it survived is handled differently.
    assert LONG_POLL_TRANSPORT_RETRY_DELAYS_S[:len(TRANSPORT_RETRY_DELAYS_S)] == TRANSPORT_RETRY_DELAYS_S
    # After an outage ends the next try comes within 60 s, well inside the 600 s acknowledgement
    # window (ADR-0031); the total wait stays far under the 7200 s guest job limit.
    assert max(LONG_POLL_TRANSPORT_RETRY_DELAYS_S) == 60
    assert sum(LONG_POLL_TRANSPORT_RETRY_DELAYS_S) == 360


@pytest.mark.parametrize('operation', sorted(LONG_POLL_OPERATIONS))
def test_long_poll_survives_an_outage_of_seven_failed_attempts(tmp_path, operation):
    op, sleeps = _operator(tmp_path, [_transport()] * 7 + [OK])
    assert op.bridge({'operation': operation}, private=True)['result']['status'] == 'RUNNING'
    assert len(op.cloud.calls) == 8
    assert sleeps == LONG


@pytest.mark.parametrize('operation', sorted(LONG_POLL_OPERATIONS))
def test_long_poll_gives_up_after_eight_failed_attempts(tmp_path, operation):
    op, sleeps = _operator(tmp_path, [_transport()] * 8 + [OK])
    with pytest.raises(TransportUnavailable):
        op.bridge({'operation': operation}, private=True)
    assert len(op.cloud.calls) == 8
    assert sleeps == LONG


@pytest.mark.parametrize('operation', sorted(RETRYABLE_BRIDGE_OPERATIONS - LONG_POLL_OPERATIONS))
def test_other_retryable_operations_keep_the_short_schedule(tmp_path, operation):
    # preflight must keep READY within 900 s; collection steps keep their ADR-0030 schedule.
    op, sleeps = _operator(tmp_path, [_transport()] * 3 + [OK])
    with pytest.raises(TransportUnavailable):
        op.bridge({'operation': operation}, private=True)
    assert len(op.cloud.calls) == 3
    assert sleeps == [15, 45]
