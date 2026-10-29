"""SDK millisecond round trips, without a provider or private installation."""
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_operator.i7 import Operator
from tests.test_study_i7_human import closed_phase_b
from tests.test_study_i7_ledger import ReservationCloud, human_operator


class MillisecondCloud(ReservationCloud):
    def __init__(self, operator, *, offset=0, **kwargs):
        super().__init__(operator, **kwargs)
        self.offset = timezone(timedelta(hours=offset))

    def command(self, arguments, **kwargs):
        try:
            return super().command(arguments, **kwargs)
        finally:
            if arguments[:4] == ['beta', 'compute', 'reservations', 'create']:
                until = datetime.fromisoformat(self.resource['deleteAtTime'])
                self.resource['deleteAtTime'] = until.astimezone(self.offset).isoformat(
                    timespec='milliseconds').replace('+00:00', 'Z')


@pytest.mark.parametrize('hours', [1, 4, 12])
@pytest.mark.parametrize('offset', [0, -5])
def test_reserve_verifies_sdk_millisecond_readback(tmp_path, hours, offset):
    op, _ = closed_phase_b(tmp_path) if hours == 12 else human_operator(tmp_path)
    if hours == 12:
        op.state['spent_usd'] = 0
    instant = op.now().replace(microsecond=968532).astimezone(timezone(timedelta(hours=offset)))
    op.now = lambda: instant
    cloud = MillisecondCloud(op, offset=offset)
    op.cloud = cloud
    result = op.reserve(hours)
    assert result['status'] == 'SPECIFIC_RESERVATION_HELD'
    expected = (instant + timedelta(hours=hours)).astimezone(timezone.utc).replace(microsecond=0)
    sent = next(a.split('=', 1)[1] for a in cloud.calls[0] if a.startswith('--delete-at-time='))
    assert sent == expected.isoformat(timespec='seconds')
    assert result['delete_at_utc'] == sent
    assert op.state['reservation']['cost_basis']['delete_at_utc'] == sent
    assert op.state['reservation']['cost_basis']['requested_utc'] == instant.isoformat()
    assert 'reservation_intent' not in json.loads(op.state_path.read_bytes())
    assert datetime.fromisoformat(cloud.resource['deleteAtTime']) == expected


@pytest.mark.parametrize('offset', [0, -5])
@pytest.mark.parametrize('delta_ms', [0, -1, 1])
def test_existing_unknown_uses_sdk_precision_without_recreate(tmp_path, offset, delta_ms):
    op, _ = human_operator(tmp_path)
    instant = op.now().replace(microsecond=968532)
    op.now = lambda: instant
    cloud = MillisecondCloud(op, offset=offset, lose_create_response=True)
    op.cloud = cloud
    with pytest.raises(TimeoutError):
        op.reserve(4)
    # Reproduce an intent durably written by 468eb25, rather than a new seconds-only intent.
    legacy_until = instant + timedelta(hours=4)
    op.state['reservation_intent']['delete_at_utc'] = legacy_until.isoformat()
    op.persist()
    expected = legacy_until.replace(microsecond=968000)
    cloud.resource['deleteAtTime'] = (expected + timedelta(milliseconds=delta_ms)).astimezone(
        cloud.offset).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    before = copy.deepcopy(op.state)
    restarted = Operator(tmp_path, cloud, now=op.now)
    if delta_ms:
        with pytest.raises(ValueError, match='UNKNOWN_RESERVATION_NATIVE_TTL_CHANGED'):
            restarted.reconcile()
        assert restarted.state == before
    else:
        assert restarted.reconcile()['status'] == 'RESERVATION_FOUND_RECONCILED'
        assert restarted.reconcile()['status'] == 'RESERVATION_STILL_PRESENT'
        assert restarted.state['spent_usd'] == before['spent_usd']
        assert restarted.state['open_exposure_usd'] == before['open_exposure_usd']
        assert restarted.state['human_phase']['reservation_blocks'] == 1
        assert restarted.state['reservation_gpu_hours_used'] == 4
        assert 'reservation_intent' not in json.loads(restarted.state_path.read_bytes())
    assert sum(c[:4] == ['beta', 'compute', 'reservations', 'create'] for c in cloud.calls) == 1


@pytest.mark.parametrize('invalid', ['2026-11-01T04:00:00.000', '2026-11-01T04:00:00.001Z'])
def test_reserve_rejects_unzoned_or_changed_native_deadline(tmp_path, invalid):
    op, cloud = human_operator(tmp_path)
    command = cloud.command
    def changed(arguments, **kwargs):
        result = command(arguments, **kwargs)
        if arguments[:3] == ['compute', 'reservations', 'describe']:
            result['deleteAtTime'] = invalid
        return result
    cloud.command = changed
    with pytest.raises(ValueError):
        op.reserve(4)
    assert 'reservation_intent' in json.loads(op.state_path.read_bytes())
