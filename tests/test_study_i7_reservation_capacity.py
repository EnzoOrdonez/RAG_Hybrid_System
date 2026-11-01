"""ADR-0026: a stated capacity refusal releases its admission only after an absent readback."""
import copy
import json
import subprocess
from datetime import timedelta

import pytest

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.i7 import PHASE_B_RESERVATION_1_H, Operator
from scripts.study_operator.policy import CapacityRejected, OperatorError
from tests.test_study_i7_human import closed_phase_b
from tests.test_study_i7_ledger import ReservationCloud

CREATE = ['beta', 'compute', 'reservations', 'create']
LIST = ['compute', 'reservations', 'list']
CAPACITY = b'ZONE_RESOURCE_POOL_EXHAUSTED: zone us-central1-a lacks available resources'


class RejectingCloud(ReservationCloud):
    """Create refused by the provider; the listing is controlled by each test."""
    def __init__(self, operator, *, error=None, listed=None, list_fails=False):
        super().__init__(operator)
        self.error = error or CapacityRejected('ZONE_RESOURCE_POOL_EXHAUSTED: fixture',
                                               code='ZONE_RESOURCE_POOL_EXHAUSTED')
        self.listed = listed or (lambda: [])
        self.list_fails = list_fails
        self.reject = True

    def command(self, arguments, **kwargs):
        if arguments[:4] == CREATE and self.reject:
            self.calls.append(arguments)
            self.at_create = json.loads(self.operator.state_path.read_bytes())
            raise self.error
        if arguments[:3] == LIST and self.reject:
            self.calls.append(arguments)
            return None if self.list_fails else self.listed()
        return super().command(arguments, **kwargs)


def operator_for(tmp_path, purpose):
    op, _ = closed_phase_b(tmp_path)
    if purpose == 'human':
        op.human_phase_open()
        return op, 4
    op.state['spent_usd'] = 0
    op.persist()
    return op, PHASE_B_RESERVATION_1_H


def ledger(op):
    state = json.loads(op.state_path.read_bytes())
    return (state.get('reservation_intent'), state.get('reservation_gpu_hours_used', 0),
            state.get('open_exposure_usd', 0), state.get('human_phase', {}).get('reservation_blocks'))


def released(op):
    folder = op.root/'receipts'/'reservation'
    rows = [json.loads(p.read_bytes()) for p in sorted(folder.glob('*.json'))] if folder.exists() else []
    return [r for r in rows if r['status'] == 'RESERVATION_CAPACITY_REJECTED_ABSENT']


def assert_still_unknown(op, hours):
    intent, used, exposure, _ = ledger(op)
    assert intent['status'] == 'UNKNOWN' and used == hours and exposure > 0
    assert not released(op)


@pytest.mark.parametrize('purpose', ['phase_b', 'human'])
def test_capacity_refusal_with_absent_readback_releases_admission(tmp_path, purpose):
    op, hours = operator_for(tmp_path, purpose)
    before = ledger(op)
    cloud = RejectingCloud(op)
    op.cloud = cloud
    with pytest.raises(CapacityRejected):
        op.reserve(hours)
    # The intent was durable before the provider call, and the listing came after it.
    assert cloud.at_create['reservation_intent']['status'] == 'UNKNOWN'
    assert [c[:4] if c[:4] == CREATE else c[:3] for c in cloud.calls] == [CREATE, LIST]
    assert ledger(op) == before
    [receipt] = released(op)
    assert receipt['cloud_error_code'] == 'ZONE_RESOURCE_POOL_EXHAUSTED'
    assert receipt['zone'] == 'us-central1-a' and receipt['hours'] == hours
    # A restarted operator may retry, and the retry is admitted from the original ledger.
    cloud.reject = False
    restarted = Operator(tmp_path, cloud, now=op.now)
    assert restarted.reserve(hours)['status'] == 'SPECIFIC_RESERVATION_HELD'
    assert restarted.state['reservation_gpu_hours_used'] == before[1] + hours
    assert sum(c[:4] == CREATE for c in cloud.calls) == 2


@pytest.mark.parametrize('listed', ['same_name', 'same_marker'])
def test_capacity_refusal_with_listed_reservation_stays_unknown(tmp_path, listed):
    op, hours = operator_for(tmp_path, 'phase_b')
    cloud = RejectingCloud(op)

    def rows():
        resource = copy.deepcopy(cloud.resource)
        if listed == 'same_name':
            resource.update(name=op.state['reservation_intent']['name'], description='other-owner')
        else:
            resource.update(name='cloudrag-i7-other', description=op.config['resource_marker'])
        return [resource]
    cloud.listed = rows
    op.cloud = cloud
    with pytest.raises(CapacityRejected):
        op.reserve(hours)
    assert_still_unknown(op, hours)
    kept = kept_receipts(op)
    assert len(kept) == 1 and kept[0]['cloud_error_code'] == 'ZONE_RESOURCE_POOL_EXHAUSTED'
    with pytest.raises(ValueError, match='UNKNOWN_RESERVATION_RECONCILE_FIRST'):
        op.reserve(hours)


def kept_receipts(op):
    folder = op.root/'receipts'/'reservation'
    rows = [json.loads(p.read_bytes()) for p in sorted(folder.glob('*.json'))] if folder.exists() else []
    return [r for r in rows if r['status'] == 'RESERVATION_CAPACITY_REJECTED_PRESENT_KEPT_UNKNOWN']


def test_release_persist_failure_keeps_memory_and_disk_unknown(tmp_path):
    op, hours = operator_for(tmp_path, 'phase_b')
    op.cloud = RejectingCloud(op)
    persist = op.persist

    def failing_release_persist():
        if op.cloud.at_create and 'reservation_intent' not in op.state:
            raise OSError('disk full')
        persist()
    op.persist = failing_release_persist
    with pytest.raises(OSError, match='disk full'):
        op.reserve(hours)
    assert op.state['reservation_intent']['status'] == 'UNKNOWN'
    assert op.state['reservation_gpu_hours_used'] == hours
    assert_still_unknown(op, hours)


def test_repeated_capacity_refusals_never_accumulate_admission(tmp_path):
    op, hours = operator_for(tmp_path, 'phase_b')
    before = ledger(op)
    op.cloud = RejectingCloud(op)
    start = op.now()
    for minutes in range(3):
        op.now = lambda minutes=minutes: start + timedelta(minutes=30*minutes)
        with pytest.raises(CapacityRejected):
            op.reserve(hours)
        assert ledger(op) == before
    assert len(released(op)) == 3


def test_capacity_refusal_with_failed_readback_stays_unknown(tmp_path):
    op, hours = operator_for(tmp_path, 'phase_b')
    op.cloud = RejectingCloud(op, list_fails=True)
    with pytest.raises(TypeError, match='RESERVATION_LIST_NOT_SUCCESSFUL'):
        op.reserve(hours)
    assert_still_unknown(op, hours)


@pytest.mark.parametrize('error', [OperatorError('Google Cloud rechazó la operación.'),
                                   OperatorError('ZONE_RESOURCE_POOL_EXHAUSTED: untyped'),
                                   TimeoutError('response lost')])
def test_other_create_failures_never_read_back_or_release(tmp_path, error):
    op, hours = operator_for(tmp_path, 'phase_b')
    cloud = RejectingCloud(op, error=error)
    op.cloud = cloud
    with pytest.raises(type(error)):
        op.reserve(hours)
    assert [c[:4] for c in cloud.calls] == [CREATE]
    assert_still_unknown(op, hours)


@pytest.mark.parametrize('stderr,code', [
    (CAPACITY, 'ZONE_RESOURCE_POOL_EXHAUSTED'),
    (b'ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS: fixture', 'ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS'),
    (b"The zone 'projects/fixture/zones/us-central1-a' does not have enough resources", 'NOT_ENOUGH_RESOURCES')])
def test_reservation_create_capacity_is_typed_and_receipted(tmp_path, stderr, code):
    cloud = Cloud('fixture', 'pure-loop-474323-a8', tmp_path,
        invoke=lambda *a, **k: subprocess.CompletedProcess([], 1, b'', stderr))
    with pytest.raises(CapacityRejected, match='ZONE_RESOURCE_POOL_EXHAUSTED') as raised:
        cloud.command([*CREATE, 'fixture', '--zone=us-central1-a'])
    assert raised.value.code == code
    receipt = json.loads((tmp_path/'0001-receipt.json').read_bytes())
    assert receipt['exit_code'] == 1 and receipt['cloud_error_code'] == code


def test_reservation_capacity_footer_with_identity_stays_private(tmp_path):
    stderr = CAPACITY + b' identity PERSON_EMAIL_CANARY@example.invalid'
    cloud = Cloud('fixture', 'pure-loop-474323-a8', tmp_path,
        invoke=lambda *a, **k: subprocess.CompletedProcess([], 1, b'', stderr))
    with pytest.raises(CapacityRejected):
        cloud.command([*CREATE, 'fixture', '--zone=us-central1-a'])
    receipt = json.loads((tmp_path/'0001-receipt.json').read_bytes())
    assert receipt['cloud_error_code'] == 'ZONE_RESOURCE_POOL_EXHAUSTED'
    assert 'capacity_error_text' not in receipt and 'PERSON_EMAIL_CANARY' not in json.dumps(receipt)


def test_other_provider_refusal_is_not_capacity(tmp_path):
    cloud = Cloud('fixture', 'pure-loop-474323-a8', tmp_path,
        invoke=lambda *a, **k: subprocess.CompletedProcess([], 1, b'', b'PERMISSION_DENIED: fixture'))
    with pytest.raises(OperatorError) as raised:
        cloud.command([*CREATE, 'fixture', '--zone=us-central1-a'])
    assert not isinstance(raised.value, CapacityRejected)
    assert 'cloud_error_code' not in json.loads((tmp_path/'0001-receipt.json').read_bytes())
