"""I7 money boundaries, exercised without a provider or participant data."""
import copy
import json
from datetime import timedelta

import pytest

from scripts.study_operator.i7 import Operator
from scripts.study_operator.lifecycle import Operator as LegacyOperator
from tests.test_study_i7_human import closed_phase_b
from tests.test_study_i7_operator import setup


@pytest.mark.parametrize('human', [False, True])
@pytest.mark.parametrize('legacy_instance', [False, True])
@pytest.mark.parametrize('operation', ['reserve_cost', 'ip_reserve'])
def test_i7_blocks_legacy_spending_before_cloud(tmp_path, human, legacy_instance, operation):
    if human:
        operator, cloud = closed_phase_b(tmp_path)
        operator.human_phase_open()
    else:
        operator, cloud, _ = setup(tmp_path)
    cloud.root = tmp_path
    if legacy_instance:
        operator = LegacyOperator(tmp_path, cloud)
    operator.config['ip_name'] = 'cloudrag-i7-fixture'
    with pytest.raises(ValueError, match='I7_LEGACY_PAID_OPERATION_FORBIDDEN'):
        if operation == 'reserve_cost':
            operator.reserve_cost('fixture', .72)
        else:
            operator.ip_reserve()
    assert cloud.calls == []


class ReservationCloud:
    """Provider boundary with a durable create observed before its response."""
    def __init__(self, operator, *, lose_create_response=False):
        self.operator = operator
        self.lose_create_response = lose_create_response
        self.calls = []
        self.exists = True
        self.resource = copy.deepcopy(operator.state['reservation'])
        self.resource.pop('deleted_verified', None)
        self.resource.pop('cost_basis', None)
        self.resource.pop('created_utc', None)
        self.at_create = None

    def command(self, arguments, **kwargs):
        self.calls.append(arguments)
        if arguments[:4] == ['beta', 'compute', 'reservations', 'create']:
            self.at_create = json.loads(self.operator.state_path.read_bytes())
            self.resource.update(name=arguments[4], id='21',
                creationTimestamp=self.operator.now().isoformat(),
                deleteAtTime=next(a.split('=', 1)[1] for a in arguments if a.startswith('--delete-at-time=')))
            if self.lose_create_response:
                raise TimeoutError('create succeeded; response lost')
            return None
        if arguments[:3] == ['compute', 'reservations', 'describe']:
            return copy.deepcopy(self.resource)
        if arguments[:3] == ['compute', 'reservations', 'list']:
            return [copy.deepcopy(self.resource)] if self.exists else []
        if arguments[:3] == ['compute', 'reservations', 'delete']:
            self.exists = False
            return None
        if arguments[:3] == ['compute', 'instances', 'describe']:
            return dict(self.operator.selected(), status='TERMINATED')
        raise AssertionError(arguments)


def human_operator(tmp_path, *, lose_create_response=False):
    operator, _ = closed_phase_b(tmp_path)
    operator.human_phase_open()
    cloud = ReservationCloud(operator, lose_create_response=lose_create_response)
    operator.cloud = cloud
    return operator, cloud


def test_reservation_intent_is_durable_before_create(tmp_path):
    operator, cloud = human_operator(tmp_path, lose_create_response=True)
    rate = operator.config['rates']['compute_usd_h']
    with pytest.raises(TimeoutError):
        operator.reserve(4)
    durable = cloud.at_create
    intent = durable['reservation_intent']
    assert durable.get('open_exposure_usd') == pytest.approx(4 * rate + .25)
    assert intent['hours'] == 4
    assert intent['compute_usd_h'] == rate
    assert intent['price_receipt_sha256'] == operator.config['rates']['price_receipt_sha256']
    assert intent['exposure_usd'] == pytest.approx(4 * rate + .25)
    assert intent['requested_utc'] == operator.now().isoformat()
    restarted = Operator(tmp_path, cloud, now=operator.now)
    before = len(cloud.calls)
    with pytest.raises(ValueError, match='UNKNOWN_RESERVATION_RECONCILE_FIRST'):
        restarted.reserve(4)
    assert len(cloud.calls) == before
    assert restarted.state['reservation_gpu_hours_used'] == 4


@pytest.mark.parametrize('hours', [1, 4])
@pytest.mark.parametrize('outcome', ['found', 'expired_absent'])
def test_unknown_reservation_keeps_requested_hours_and_is_idempotent(tmp_path, hours, outcome):
    operator, cloud = human_operator(tmp_path, lose_create_response=True)
    original_rate = operator.config['rates']['compute_usd_h']
    expected = hours * original_rate + .25
    before = operator.state['spent_usd']
    started = operator.now()
    with pytest.raises(TimeoutError):
        operator.reserve(hours)
    # Reconciliation uses the admitted intent even if the current quote changes.
    operator.config['rates']['compute_usd_h'] = original_rate / 2
    operator.persist()
    if outcome == 'expired_absent':
        cloud.exists = False
        operator.now = lambda: started + timedelta(hours=hours + 1)
    restarted = Operator(tmp_path, cloud, now=operator.now)
    restarted.reconcile()
    if outcome == 'found':
        assert restarted.state['open_exposure_usd'] == pytest.approx(expected)
        assert restarted.state['spent_usd'] == before
        assert restarted.state['reservation']['cost_basis']['hours'] == hours
    else:
        assert restarted.state['spent_usd'] - before == pytest.approx(expected)
        assert restarted.state['open_exposure_usd'] == 0
    assert 'reservation_intent' not in restarted.state
    durable = json.loads(restarted.state_path.read_bytes())
    restarted = Operator(tmp_path, cloud, now=operator.now)
    restarted.reconcile()
    assert restarted.state['spent_usd'] == durable['spent_usd']
    assert restarted.state['open_exposure_usd'] == durable['open_exposure_usd']
    assert sum(c[:4] == ['beta', 'compute', 'reservations', 'create'] for c in cloud.calls) == 1


@pytest.mark.parametrize('closure', ['guard', 'reconcile'])
def test_open_reservation_keeps_price_after_refresh(tmp_path, monkeypatch, closure):
    from scripts.study_operator import i7_costs, pricing
    operator, cloud = human_operator(tmp_path)
    started = operator.now()
    old_rate = operator.config['rates']['compute_usd_h']
    operator.reserve(4)
    before = copy.deepcopy(operator.state['reservation'])
    assert before.get('cost_basis', {}).get('compute_usd_h') == old_rate
    operator.now = lambda: started + timedelta(hours=1)
    plan = copy.deepcopy(operator.config['cost_plan'])
    plan['verified_utc'] = operator.now().isoformat()
    monkeypatch.setattr(i7_costs, 'list_prices', lambda *_: plan)
    monkeypatch.setattr(pricing, 'quote_archive', lambda *_:
        {'usd_per_hour': str(old_rate / 2), 'catalog_receipt_sha256': 'b' * 64})
    compute = tmp_path / 'compute'
    compute.mkdir()
    (compute / 'receipt.json').write_text(json.dumps({'at': operator.now().isoformat()}))
    operator.refresh_prices(compute, tmp_path / 'storage')
    assert operator.config['rates']['compute_usd_h'] == old_rate / 2
    assert operator.state['reservation'] == before
    assert operator.state['open_exposure_usd'] == pytest.approx(4 * old_rate + .25)
    spent = operator.state['spent_usd']
    if closure == 'guard':
        operator.guard()
        elapsed = 1
    else:
        cloud.exists = False
        operator.now = lambda: started + timedelta(hours=5)
        operator.reconcile()
        elapsed = 4
    assert operator.state['spent_usd'] - spent == pytest.approx(elapsed * old_rate)
    assert operator.state['open_exposure_usd'] == 0
    operator.reconcile()
    assert operator.state['spent_usd'] - spent == pytest.approx(elapsed * old_rate)
