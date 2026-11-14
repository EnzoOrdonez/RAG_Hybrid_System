"""ADR-0032: once the GPU VM exists, every reservation keeps the name the VM consumes.

Compute Engine binds a VM to one SPECIFIC reservation by name when the VM is created and
cannot rebind an existing VM to another specific reservation (only ANY or NONE). The gate
of attempt 12 created reservation 2 under a fresh name and start() refused the boot with
SPECIFIC_RESERVATION_NOT_BOUND after the single gate attempt was already recorded.
"""
from datetime import timedelta

import pytest

from scripts.study_operator.i7 import PHASE_B_GATE_RESERVATION_H, PHASE_B_RESERVATION_1_H
from scripts.study_operator.policy import CapacityRejected
from tests.test_study_i7_human import closed_phase_b
from tests.test_study_i7_ledger import ReservationCloud, human_operator
from tests.test_study_i7_reservation_capacity import RejectingCloud, released

CREATE = ['beta', 'compute', 'reservations', 'create']


def created(cloud):
    return [call[4] for call in cloud.calls if call[:4] == CREATE]


class AlteredVmCloud(ReservationCloud):
    """The provider's VM as read back, altered by one test case."""

    def __init__(self, operator, alter):
        super().__init__(operator)
        self.alter = alter

    def command(self, arguments, **kwargs):
        vm = super().command(arguments, **kwargs)
        if arguments[:3] == ['compute', 'instances', 'describe']:
            self.alter(vm)
        return vm


def gate_operator(tmp_path, alter=None):
    op, _ = closed_phase_b(tmp_path, None)
    op.state['spent_usd'] = 0
    op.state['reservation_gpu_hours_used'] = PHASE_B_RESERVATION_1_H
    cloud = AlteredVmCloud(op, alter or (lambda vm: None))
    op.cloud = cloud
    return op, cloud


def test_gate_reservation_reuses_the_name_the_vm_consumes(tmp_path):
    op, cloud = gate_operator(tmp_path)
    bound = op.state['reservation']['name']
    assert op.reserve(PHASE_B_GATE_RESERVATION_H)['status'] == 'SPECIFIC_RESERVATION_HELD'
    assert created(cloud) == [bound]
    assert op.state['reservation']['name'] == bound and op.state['reservation']['id'] == '21'


@pytest.mark.parametrize('alter', [
    lambda vm: vm['reservationAffinity'].update(values=['cloudrag-i7-0101000000-other']),
    lambda vm: vm.update(id='999'),
    lambda vm: vm['reservationAffinity'].update(consumeReservationType='ANY_RESERVATION'),
    lambda vm: vm['reservationAffinity'].update(consumeReservationType='NO_RESERVATION', values=[]),
    lambda vm: vm.pop('reservationAffinity'),
], ids=['other_name', 'other_vm', 'any', 'none', 'missing'])
def test_reservation_the_vm_cannot_consume_is_never_created(tmp_path, alter):
    op, cloud = gate_operator(tmp_path, alter)
    with pytest.raises(ValueError, match='^SPECIFIC_RESERVATION_NOT_BOUND$'):
        op.reserve(PHASE_B_GATE_RESERVATION_H)
    assert not created(cloud)
    assert 'reservation_intent' not in op.state
    assert op.state['reservation_gpu_hours_used'] == PHASE_B_RESERVATION_1_H
    assert op.state['reservation']['deleted_verified'] is True


def test_allocated_vm_without_a_recorded_reservation_is_refused(tmp_path):
    op, cloud = gate_operator(tmp_path)
    op.state.pop('reservation')
    with pytest.raises(ValueError, match='^SPECIFIC_RESERVATION_NOT_BOUND$'):
        op.reserve(PHASE_B_GATE_RESERVATION_H)
    assert not created(cloud)


def test_first_reservation_before_the_vm_exists_gets_a_fresh_name(tmp_path):
    op, cloud = gate_operator(tmp_path)
    op.config['primary_vm'] = {'name': 'cloudrag-i7-unallocated', 'id': '0', 'zone': 'us-central1-a'}
    op.state.pop('reservation')
    op.state['reservation_gpu_hours_used'] = 0
    op.reserve(PHASE_B_RESERVATION_1_H)
    [name] = created(cloud)
    assert name == 'cloudrag-i7-'+op.now().strftime('%m%d%H%M%S')+'-'+op.config['period_id'][:8]
    assert not [call for call in cloud.calls if call[:3] == ['compute', 'instances', 'describe']]


def test_human_blocks_keep_reusing_the_bound_name(tmp_path):
    op, cloud = human_operator(tmp_path)
    bound = op.state['reservation']['name']
    op.reserve(4)
    assert op.guard()['status'] == 'RESERVATION_DELETED_VERIFIED'
    later = op.now() + timedelta(hours=5)  # the fixture clock is frozen; real blocks are hours apart
    op.now = lambda: later
    op.reserve(4)
    assert created(cloud) == [bound, bound]
    assert op.state['human_phase']['reservation_blocks'] == 2


def test_capacity_refusal_on_a_reused_name_releases_the_admission(tmp_path):
    op, _ = human_operator(tmp_path)
    bound = op.state['reservation']['name']
    cloud = RejectingCloud(op)
    op.cloud = cloud
    with pytest.raises(CapacityRejected):
        op.reserve(4)
    assert created(cloud) == [bound]
    assert 'reservation_intent' not in op.state
    assert released(op)[0]['name'] == bound
