"""ADR-0034: the read-only maintenance reads of the technical cleanup retry a transport failure.

In attempt 14 the technical cleanup after the gate stopped at `download-disk`: the SSH no-op that
authenticates the transport timed out (90 s) and the call had no retry. `inventory` and
`download-disk` only read the session stores under maintenance (the guest re-checks that the
inventory still equals the plan), so they take the short schedule. Calls with effects keep one try.
"""
import pytest

from scripts.study_operator.lifecycle import (
    LONG_POLL_OPERATIONS,
    MAINTENANCE_READ_OPERATIONS,
    RETRYABLE_BRIDGE_OPERATIONS,
)
from scripts.study_operator.policy import TransportUnavailable
from tests.test_study_i7_attempt10_fixes import OK, _operator, _transport


def test_maintenance_reads_are_inventory_and_download_only():
    assert MAINTENANCE_READ_OPERATIONS == frozenset({'inventory', 'download-disk'})
    assert MAINTENANCE_READ_OPERATIONS <= RETRYABLE_BRIDGE_OPERATIONS
    assert not MAINTENANCE_READ_OPERATIONS & LONG_POLL_OPERATIONS


@pytest.mark.parametrize('operation', sorted(MAINTENANCE_READ_OPERATIONS))
def test_maintenance_read_survives_two_transport_failures(tmp_path, operation):
    op, sleeps = _operator(tmp_path, [_transport(), _transport(), OK])
    assert op.bridge({'operation': operation}, private=True)['result']['status'] == 'RUNNING'
    assert len(op.cloud.calls) == 3
    assert sleeps == [15, 45]


@pytest.mark.parametrize('operation', ['maintenance', 'i7-technical-close', 'restore', 'clean-disk'])
def test_maintenance_calls_with_effects_keep_a_single_attempt(tmp_path, operation):
    op, sleeps = _operator(tmp_path, [_transport(), OK])
    with pytest.raises(TransportUnavailable):
        op.bridge({'operation': operation}, private=True)
    assert len(op.cloud.calls) == 1 and sleeps == []
