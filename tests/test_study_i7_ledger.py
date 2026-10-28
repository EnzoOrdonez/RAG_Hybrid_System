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


@pytest.mark.parametrize('hours', [1, 4])
def test_expired_no_show_is_charged_only_to_ttl_once(tmp_path, hours):
    operator, cloud = human_operator(tmp_path)
    rate = operator.config['rates']['compute_usd_h']
    started, spent = operator.now(), operator.state['spent_usd']
    operator.reserve(hours)
    cloud.exists = False
    operator.now = lambda: started + timedelta(days=3)
    operator.reconcile()
    assert operator.state['spent_usd'] - spent == pytest.approx(hours * rate)
    assert operator.state['open_exposure_usd'] == 0
    restarted = Operator(tmp_path, cloud, now=operator.now)
    restarted.reconcile()
    assert restarted.state['spent_usd'] - spent == pytest.approx(hours * rate)
    assert not any(c[:3] == ['compute', 'instances', 'start'] for c in cloud.calls)


def test_human_extra_exposure_blocks_even_without_another_reservation(tmp_path):
    operator, cloud = human_operator(tmp_path)
    operator.state['extra_exposure_usd'] = 67
    with pytest.raises(ValueError, match='I7_HUMAN_BUDGET_STOP'):
        operator.admit_paid(0)
    assert cloud.calls == []


@pytest.mark.parametrize('extra', [-1, float('nan'), float('inf'), None, True])
def test_human_extra_exposure_must_be_finite_nonnegative(tmp_path, extra):
    operator, cloud = human_operator(tmp_path)
    operator.state['extra_exposure_usd'] = extra
    with pytest.raises(ValueError, match='FINITE_EXTRA_EXPOSURE_REQUIRED'):
        operator.admit_paid(0)
    assert cloud.calls == []


def test_noncompute_package_blocks_reservation_before_create(tmp_path):
    operator, cloud = human_operator(tmp_path)
    operator.config['cost_plan']['rates']['gcs_a']['usd'] = .0066
    with pytest.raises(ValueError, match='I7_HUMAN_BUDGET_STOP'):
        operator.reserve(1)
    assert cloud.calls == []
    assert operator.state['human_phase'].get('reservation_blocks', 0) == 0
    assert 'reservation_intent' not in operator.state


def test_noncompute_full_runbook_package_is_retained_per_reservation(tmp_path):
    from scripts.study_operator.i7_costs import default_quantities
    operator, cloud = human_operator(tmp_path)
    operator.state['extra_exposure_usd'] = .4
    for _ in range(3):
        operator.reserve(1)
        # The package count is durable before CREATE, even for a no-show.
        assert cloud.at_create['human_phase']['reservation_blocks'] == _ + 1
        current = operator.now()
        operator.now = lambda current=current: current + timedelta(hours=1)
        operator.guard()
    quantities = default_quantities(24)
    retained = {'pd_gib_h', 'snapshot_gib_h', 'ip_unused_h'}
    compute = {'g2_vcpu', 'g2_ram', 'l4', 'e2_vcpu', 'e2_ram'}
    rates = operator.config['cost_plan']['rates']
    package = 3 * sum(quantities[k] * rates[k]['usd'] for k in quantities if k not in compute)
    # Three full 24h packages dominate the continuous 48h minimum here.
    assert sum(quantities[k] * rates[k]['usd'] for k in retained) > 0
    assert operator.human_committed_usd(operator.now()) == pytest.approx(
        package + 3 * operator.config['rates']['compute_usd_h'] + .4)
    assert operator.state['human_phase']['reservation_blocks'] == 3


@pytest.mark.parametrize('key,quantity', [
    ('pd_gib_h', 540 * 48), ('snapshot_gib_h', 552 * 48), ('ip_unused_h', 48),
    ('egress_gib', 1), ('gcs_gib_h', 24), ('gcs_a', 10000),
    ('gcs_b', 100000), ('gcs_egress_gib', 1),
])
def test_human_each_noncompute_sku_contributes(tmp_path, key, quantity):
    operator, _ = human_operator(tmp_path)
    for row in operator.config['cost_plan']['rates'].values():
        row['usd'] = 0
    operator.config['cost_plan']['rates'][key]['usd'] = 35 / quantity
    operator.reserve(1)
    compute = operator.state['open_exposure_usd']
    assert operator.human_committed_usd(operator.now()) == pytest.approx(35 + compute)


@pytest.mark.parametrize('blocks', [None, -1, True])
def test_human_reservation_count_requires_nonnegative_integer(tmp_path, blocks):
    operator, cloud = human_operator(tmp_path)
    operator.state['human_phase']['reservation_blocks'] = blocks
    with pytest.raises(ValueError, match='HUMAN_RESERVATION_LEDGER_REQUIRED'):
        operator.admit_paid(0)
    assert cloud.calls == []
