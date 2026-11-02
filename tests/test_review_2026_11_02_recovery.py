"""Auditoría 2026-11-02, pasada de recuperación (B-1..B-6) sobre la base b470d3e; solo dobles, sin nube.

Estos tests FIJAN el comportamiento defectuoso actual (cada uno afirma que el defecto existe):
cuando se aplique la corrección deben fallar y hay que invertir su aserción.
"""
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_operator.i7 import Operator
from tests.test_study_operator_deployment import config as deployment_config

NOW = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)


def vm_row(name, vid, status='TERMINATED', reservation='cloudrag-i7-res'):
    return {'name': name, 'id': vid, 'zone': 'zones/us-central1-a', 'status': status,
            'machineType': 'zones/us-central1-a/machineTypes/g2-standard-4', 'deletionProtection': True,
            'disks': [{'autoDelete': False, 'boot': True, 'source': 'x/' + name + '-boot'}],
            'scheduling': {'instanceTerminationAction': 'STOP', 'maxRunDuration': {'seconds': '10800'}},
            'reservationAffinity': {'consumeReservationType': 'SPECIFIC_RESERVATION', 'values': [reservation]},
            'description': 'cloudrag-i7-test'}


class FakeCloud:
    """Minimal stateful GCE double. `fail` maps a command prefix tuple to an exception."""
    def __init__(self, cfg, reservation):
        self.cfg = cfg
        self.calls = []
        self.fail = {}
        self.vms = {cfg['primary_vm']['name']: vm_row(cfg['primary_vm']['name'], cfg['primary_vm']['id'])}
        self.reservations = {reservation['name']: copy.deepcopy(reservation)} if reservation else {}
        self.disks, self.snapshots, self.firewalls = {}, {}, {}
        self.stop_sticks = True   # stop() leaves VM TERMINATED

    def require_beta(self):
        return {'status': 'GCLOUD_BETA_INSTALLED'}

    def command(self, argv, **kwargs):
        self.calls.append(list(argv))
        for prefix, error in list(self.fail.items()):
            if tuple(argv[:len(prefix)]) == prefix:
                raise error
        kind, verb = argv[1], argv[2]
        name = argv[3] if len(argv) > 3 else None
        table = {'instances': self.vms, 'reservations': self.reservations, 'disks': self.disks,
                 'snapshots': self.snapshots, 'firewall-rules': self.firewalls}.get(kind)
        if verb == 'list':
            return [copy.deepcopy(v) for v in table.values()]
        if verb == 'describe':
            if name not in table:
                raise RuntimeError('NOT_FOUND ' + kind + '/' + name)
            return copy.deepcopy(table[name])
        if verb == 'delete':
            if name not in table:
                raise RuntimeError('NOT_FOUND ' + kind + '/' + name)
            del table[name]
            return None
        if kind == 'instances' and verb == 'stop':
            self.vms[name]['status'] = 'TERMINATED' if self.stop_sticks else 'STOPPING'
            return None
        if kind == 'instances' and verb in ('start', 'add-metadata', 'update', 'set-scheduling'):
            if verb == 'start':
                self.vms[name]['status'] = 'RUNNING'
            return None
        raise AssertionError(argv)


def make_op(tmp_path, *, primary=None, reservation_ttl_h=12, with_reservation=True):
    from scripts.study_operator.i7_costs import KEYS
    cfg = dict(deployment_config(), iteration=7, service_account='sa', network='default', subnet='default',
               primary_vm=primary or {'name': 'cloudrag-i7-vm', 'id': '10', 'zone': 'us-central1-a'},
               native_stop_s=10800, budget_usd=20, closure_reserve_usd=3,
               rates={'compute_usd_h': .706832255, 'price_receipt_sha256': 'a' * 64,
                      'verified_utc': '2026-11-01T00:00:00+00:00'},
               resource_marker='cloudrag-i7-test')
    cfg['cost_plan'] = {'status': 'PRICED_LIST', 'verified_utc': '2026-11-01T00:00:00+00:00',
                        'rates': {key: {'usd': .000001, 'sku_id': key} for key in KEYS}}
    (tmp_path / 'installation.json').write_text(json.dumps(cfg))
    (tmp_path / 'I7.json').write_text(json.dumps({'iteration': 7}))
    reservation = {'name': 'cloudrag-i7-res', 'id': '20', 'zone': 'zones/us-central1-a',
                   'description': 'cloudrag-i7-test',
                   'deleteAtTime': (NOW + timedelta(hours=reservation_ttl_h)).isoformat(),
                   'specificReservation': {'count': '1', 'instanceProperties': {'machineType': 'g2-standard-4',
                       'guestAccelerators': [{'acceleratorType': 'nvidia-l4', 'acceleratorCount': 1}]}}}
    cloud = FakeCloud(cfg, reservation if with_reservation else None)
    tick = iter(range(10**6))
    op = Operator(tmp_path, cloud, now=lambda: NOW + timedelta(microseconds=next(tick)), sleep=lambda s: None)
    if with_reservation:
        op.state['reservation'] = dict(reservation, created_utc='2026-11-01T00:00:00+00:00',
            cost_basis={'hours': 12, 'compute_usd_h': .706832255, 'exposure_usd': 12 * .706832255 + .25,
                        'price_receipt_sha256': 'a' * 64})
        op.state['open_exposure_usd'] = 12 * .706832255 + .25
        op.persist()
    return op, cloud


def reload(op):
    return Operator(op.root, op.cloud, now=op.now, sleep=op.sleep)


# ---------------------------------------------------------------- B-1
def test_b1_series_burns_stimulus_slot_on_free_precondition_refusal(tmp_path):
    """INTENT is written before start() checks TTL; no boot happened, yet the slot is terminal."""
    from scripts.study_operator.i7_workflows import series
    op, cloud = make_op(tmp_path, reservation_ttl_h=2)       # < native_stop_s + 60 s
    with pytest.raises(ValueError) as first:
        series(op)
    print('first run raised:', first.value)
    assert not any(c[:3] == ['compute', 'instances', 'start'] for c in cloud.calls)   # nothing paid/booted
    assert json.loads((tmp_path / 'attempts' / 'stimulus-01.json').read_text())['status'] == 'INTENT'
    assert reload(op).state['reservation']['deleted_verified']  # finally: guard closed reservation 1
    # Retry (e.g. after a fresh reservation) can never pass slot 1:
    op2 = reload(op)
    op2.state['reservation'].pop('deleted_verified')   # pretend a new open reservation exists
    op2.state['reservation']['deleteAtTime'] = (NOW + timedelta(hours=12)).isoformat()
    cloud.reservations['cloudrag-i7-res'] = {k: v for k, v in op2.state['reservation'].items()
                                             if k not in ('created_utc', 'cost_basis', 'delete_intent_sha256', 'delete_api_success', 'deleted_utc')}
    with pytest.raises(ValueError) as second:
        series(op2)
    print('second run raised:', second.value)
    assert str(second.value) == 'TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW'


# ---------------------------------------------------------------- B-2
def test_b2a_guard_keyerror_masks_real_refusal(tmp_path):
    from scripts.study_operator.i7_workflows import series
    op, cloud = make_op(tmp_path, with_reservation=False)
    with pytest.raises(BaseException) as raised:
        series(op)
    print('raised:', type(raised.value).__name__, raised.value,
          '| context:', type(raised.value.__context__).__name__, raised.value.__context__)
    assert isinstance(raised.value, KeyError)        # real cause STOPPED_VM_AND_SPECIFIC_RESERVATION_REQUIRED hidden


def test_b2b_already_closed_masks_real_refusal(tmp_path):
    from scripts.study_operator.i7_workflows import series
    op, cloud = make_op(tmp_path)
    op.state['reservation']['deleted_verified'] = True
    with pytest.raises(ValueError) as raised:
        series(op)
    print('raised:', raised.value, '| context:', raised.value.__context__)
    assert str(raised.value) == 'RESERVATION_ALREADY_CLOSED'


def test_b2c_stop_failure_in_finally_skips_guard_and_masks(tmp_path):
    from scripts.study_operator.i7_workflows import series
    op, cloud = make_op(tmp_path, reservation_ttl_h=2)
    cloud.vms['cloudrag-i7-vm']['status'] = 'STOPPING'   # e.g. native STOP in progress / slow API
    cloud.stop_sticks = False
    with pytest.raises(ValueError) as raised:
        series(op)
    print('raised:', raised.value, '| context:', raised.value.__context__)
    assert str(raised.value) == 'STOP_NOT_VERIFIED'
    assert not any(c[:3] == ['compute', 'reservations', 'delete'] for c in cloud.calls)
    assert 'cloudrag-i7-res' in cloud.reservations and not reload(op).state['reservation'].get('deleted_verified')


# ---------------------------------------------------------------- B-3
def _cpu_fixture(op, cloud):
    from scripts.study_operator.i7_cpu_pipeline import cpu_label
    marker = 'CloudRAG-I5-restore-cpu-' + cpu_label(op.config)
    vm = {'name': 'cloudrag-i5-restore-' + cpu_label(op.config), 'id': '31', 'description': marker,
          'status': 'TERMINATED', 'zone': 'zones/us-central1-a'}
    src = {'name': vm['name'] + '-boot', 'id': '32', 'description': marker, 'zone': 'zones/us-central1-a'}
    svc = {'name': 'cloudrag-i7-service-x', 'id': '33', 'description': 'cloudrag-i7-test', 'zone': 'zones/us-central1-a'}
    stage = {'name': 'cloudrag-i7-stage-x', 'id': '34', 'description': 'cloudrag-i7-test'}
    fw = {'name': vm['name'] + '-iap', 'id': '35', 'description': marker}
    cloud.vms[vm['name']] = dict(vm)
    cloud.disks.update({src['name']: dict(src), svc['name']: dict(svc)})
    cloud.snapshots[stage['name']] = dict(stage)
    cloud.firewalls[fw['name']] = dict(fw)
    (op.root / 'cpu').mkdir()
    (op.root / 'cpu' / 'STATE.json').write_text(json.dumps({'resources': [
        dict(type='firewall', name=fw['name'], id='35', ownership_marker=marker)]}))
    legacy = op.root / 'private' / 'legacy'
    legacy.mkdir(parents=True)
    (legacy / 'i3-nine-originals.tar').write_bytes(b'tar')
    op.state.update(cpu_resources={'vm': vm, 'source_disk': src, 'service_disk': svc, 'stage_snapshot': stage},
                    legacy_archive_sha256=hashlib.sha256(b'tar').hexdigest(),
                    qualified_service_snapshot={'id': '40', 'name': 'cloudrag-i7-clean-x'})
    op.persist()


def test_b3_cleanup_cpu_cannot_resume_after_partial_delete(tmp_path):
    from scripts.study_operator.i7_cpu_pipeline import cleanup_cpu
    op, cloud = make_op(tmp_path, with_reservation=False)
    _cpu_fixture(op, cloud)
    cloud.fail[('compute', 'disks', 'delete', 'cloudrag-i7-service-x')] = RuntimeError('API_TIMEOUT')
    with pytest.raises(RuntimeError):
        cleanup_cpu(op)
    print('after 1st attempt: vms', sorted(cloud.vms), 'disks', sorted(cloud.disks))
    cloud.fail.clear()                     # transient error gone
    with pytest.raises(RuntimeError) as retry:
        cleanup_cpu(reload(op))
    print('retry raised:', retry.value)
    assert 'NOT_FOUND instances/' in str(retry.value)
    assert 'cloudrag-i7-service-x' in cloud.disks and 'cloudrag-i7-stage-x' in cloud.snapshots


# ---------------------------------------------------------------- B-4
def test_b4_close_cannot_release_reservation_held_before_gpu_vm_exists(tmp_path, monkeypatch):
    """gpu_restore fails after reserve() (disk create error); close then can never DELETE the reservation."""
    from scripts.study_operator import i7_cpu_pipeline
    from scripts.study_operator.i7_workflows import dispatch
    op, cloud = make_op(tmp_path, with_reservation=False,
                        primary={'name': 'cloudrag-i7-unallocated', 'id': '0', 'zone': 'us-central1-a'})
    del cloud.vms['cloudrag-i7-unallocated']
    monkeypatch.setattr(i7_cpu_pipeline, 'verify_image_or_stop_cpu', lambda owner: None)
    cloud.snapshots['cloudrag-i7-clean-x'] = {'name': 'cloudrag-i7-clean-x', 'id': '40', 'status': 'READY',
                                              'storageLocations': ['us-central1']}
    op.state['qualified_service_snapshot'] = dict(cloud.snapshots['cloudrag-i7-clean-x'])
    real = cloud.command
    def command(argv, **kw):
        if argv[:4] == ['beta', 'compute', 'reservations', 'create']:
            cloud.calls.append(list(argv))
            delete_at = [a for a in argv if a.startswith('--delete-at-time=')][0].split('=', 1)[1]
            cloud.reservations[argv[4]] = {'name': argv[4], 'id': '21', 'zone': 'zones/us-central1-a',
                'description': 'cloudrag-i7-test', 'deleteAtTime': delete_at, 'creationTimestamp': NOW.isoformat(),
                'specificReservation': {'count': '1', 'instanceProperties': {'machineType': 'g2-standard-4',
                    'guestAccelerators': [{'acceleratorType': 'nvidia-l4', 'acceleratorCount': 1}]}}}
            return None
        if argv[:3] == ['compute', 'disks', 'create']:
            cloud.calls.append(list(argv))
            raise RuntimeError('QUOTA_EXCEEDED_SSD_TOTAL_GB')
        return real(argv, **kw)
    cloud.command = command
    with pytest.raises(RuntimeError):
        i7_cpu_pipeline.gpu_restore(op)
    op = reload(op)
    print('held reservation:', [r['name'] for r in cloud.reservations.values()],
          'hours used', op.state['reservation_gpu_hours_used'])
    with pytest.raises(RuntimeError) as closing:
        dispatch(op, 'close')
    print('close raised:', closing.value)
    assert cloud.reservations, 'reservation still billing until native TTL'
    assert not any(c[:3] == ['compute', 'reservations', 'delete'] for c in cloud.calls)


# ---------------------------------------------------------------- B-5
def test_b5_persist_pair_is_not_atomic(tmp_path, monkeypatch):
    """start(): installation.json gets purpose/period_id=smoke, active.json keeps purpose=technical."""
    from scripts.study_operator import i7
    op, cloud = make_op(tmp_path)
    op.config['purpose'] = 'technical'
    op.config['period_ids'] = {'technical': 'a' * 32}
    op.state['purpose'] = 'technical'
    op.persist()
    real = i7.save_state
    def flaky(path, value):
        if path.name == 'active.json' and value.get('purpose') == 'smoke':
            raise OSError('crash between the two writes')
        return real(path, value)
    monkeypatch.setattr(i7, 'save_state', flaky)
    monkeypatch.setattr('scripts.study_operator.startup.write_startup', lambda path, config, **k: None)
    with pytest.raises(OSError):
        op.start('smoke')
    disk = reload(op)
    print('disk config purpose/period:', disk.config['purpose'], disk.config['period_id'][:8],
          '| disk state purpose:', disk.state['purpose'])
    assert disk.config['purpose'] == 'smoke' and disk.state['purpose'] == 'technical'


# ---------------------------------------------------------------- B-6
def test_b6_gate_reuses_open_short_ttl_reservation_and_burns_single_attempt(tmp_path, monkeypatch):
    """A reservation left open (e.g. B-2c) is reused without a TTL check; the no-replay
    gate attempt file is written, then the first start() refuses for free."""
    from scripts.study_operator import i7_phase_b, i7_smoke, i7_admission, stimulus_evidence
    from src.ui.components import study_protocol
    op, cloud = make_op(tmp_path, reservation_ttl_h=2)
    (tmp_path / 'stimulus-acceptance.json').write_text(json.dumps({'status': 'ACCEPTED'}))
    ref = tmp_path / 'ref.json'
    ref.write_text(json.dumps({'rag': {'contexts': {}}}))
    op.config.update(local_reviewed_config=str(tmp_path), i3_context_reference_path=str(ref))
    monkeypatch.setattr(stimulus_evidence, 'accept', lambda p, c: {'status': 'ACCEPTED', 'software_sha256': 's'})
    monkeypatch.setattr(study_protocol, 'verify_draw', lambda path: {'config': {}})
    monkeypatch.setattr(i7_admission, 'compare_contexts', lambda a, b: None)
    monkeypatch.setattr(i7_smoke, 'validate_receipt', lambda path: {})
    with pytest.raises(ValueError) as raised:
        i7_phase_b._gate_after_prerequisites(op, {'software_sha256': 's'})
    print('gate raised:', raised.value, '| boots started:',
          sum(c[:3] == ['compute', 'instances', 'start'] for c in cloud.calls))
    assert (tmp_path / 'gate-cohort-intent.json').exists()
    assert not any(c[:4] == ['beta', 'compute', 'reservations', 'create'] for c in cloud.calls)
    with pytest.raises(ValueError, match='GATE_ATTEMPT_EXISTS_NO_REPLAY'):
        i7_phase_b._gate_after_prerequisites(reload(op), {'software_sha256': 's'})
