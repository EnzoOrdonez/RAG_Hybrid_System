from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path

import pytest

from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_lifecycle import installation


def standby(tmp_path, *, lose_reply=False):
    operator,cloud = installation(tmp_path)
    operator.state['failover_data_reconciled'] = True
    operator.state['ready'] = dict(tls=dict(certificate_sha256='1'*64))
    operator.config.update(network='private',subnet='regional',service_account='minimal-sa')
    operator.config['prepared_snapshot'] = dict(name='prepared',id='456',image_id=operator.config['image_id'],
        hostname=operator.config['hostname'],certificate_sha256='1'*64,session_data='ALL_I4_PERIODS_EMPTY_VERIFIED')
    instances,disks,calls = [cloud.vm],[],[]
    def value(args,key):
        return next(arg.removeprefix('--'+key+'=') for arg in args if arg.startswith('--'+key+'='))
    def command(args,**options):
        calls.append(args)
        if args[:3] == ['compute','instances','describe']:
            return next(row for row in instances if row['name'] == args[3])
        if args[:3] == ['compute','instances','list']:
            return instances
        if args[:3] == ['compute','snapshots','describe']:
            return dict(name='prepared',id='456',status='READY',storageLocations=['us-central1'])
        if args[:3] == ['compute','disks','list']:
            return disks
        if args[:3] == ['compute','disks','describe']:
            return disks[0]
        if args[:3] == ['compute','disks','create']:
            disks.append(dict(name=args[3],id='789',zone='zones/us-central1-b',sourceSnapshotId='456',
                selfLink='projects/p/zones/us-central1-b/disks/'+args[3],description=value(args,'description')))
        if args[:3] == ['compute','instances','create']:
            assert instances[0]['status'] == 'TERMINATED'
            assert '--address=203.0.113.8' in args and '--no-address' not in args
            startup_path = value(args,'metadata-from-file').removeprefix('startup-script=')
            startup = Path(startup_path).read_text(encoding='utf-8')
            assert 'host_runtime' in startup and 'Metadata-Flavor' in startup
            row = deepcopy(cloud.vm)
            row.update(name=args[3],id='777',zone='zones/us-central1-b',status='RUNNING',
                description=value(args,'description'),creationTimestamp='2026-10-05T00:00:00+00:00',
                lastStartTimestamp='2026-10-05T00:00:00+00:00')
            row['metadata'] = dict(items=[dict(key='startup-script',value=startup)])
            row['disks'][0]['source'] = disks[0]['selfLink']
            row['networkInterfaces'][0]['accessConfigs'] = [dict(name='External NAT',natIP='203.0.113.8')]
            instances.append(row)
            if lose_reply:
                raise OperatorError('create reply lost')
        if args[:3] == ['compute','instances','stop']:
            row = next(row for row in instances if row['name'] == args[3])
            row.update(status='TERMINATED',lastStopTimestamp='2026-10-05T00:02:00+00:00')
        return None
    cloud.command = command
    operator.transfer_ip = lambda target:calls.append(['verified-transfer',target['id']])
    return operator,cloud,instances,disks,calls


def test_standby_keeps_acquired_capacity_and_accounts_for_one_uninterrupted_boot(tmp_path):
    operator,_,instances,disks,calls = standby(tmp_path)
    assert operator.failover('us-central1-b')['status'] == 'ALTERNATE_STARTED_SUPERVISED'
    marker = operator.state['alternate_vms'][0]['ownership_marker']
    assert instances[-1]['description'] == disks[0]['description'] == marker
    assert operator.state['cost']['estimated_usd'] == 0
    assert 'alternate-b' in operator.state['cost']['reservations']
    assert instances[-1]['status'] == 'RUNNING'
    assert not any(args[:3] == ['compute','instances','stop'] for args in calls)
    assert not any(args[0] == 'verified-transfer' for args in calls)
    assert operator.state['selected_vm']['id'] == '777'
    operator.now = lambda:datetime(2026,10,5,0,2,tzinfo=timezone.utc)
    operator.stop()
    estimate = operator.state['cost']['estimated_usd']
    assert estimate == pytest.approx(.706832276*120/3600)
    assert 'alternate-b' not in operator.state['cost']['reservations']
    operator.settle_alternate_creation(operator.selected())
    assert operator.state['cost']['estimated_usd'] == estimate


def test_lost_create_reply_adopts_only_its_marked_disk_and_vm_before_transfer(tmp_path):
    operator,_,instances,_,calls = standby(tmp_path,lose_reply=True)
    with pytest.raises(OperatorError,match='create reply lost'):
        operator.failover('us-central1-b')
    assert operator.state['alternate_creation_intent']['source_snapshot_id'] == '456'
    assert not any(row[0] == 'verified-transfer' for row in calls)
    assert operator.failover('us-central1-b')['status'] == 'ALTERNATE_STARTED_SUPERVISED'
    assert instances[-1]['status'] == 'RUNNING'
    assert len(operator.state['alternate_vms']) == 1


def test_study_failover_without_ethical_record_rejects_before_cloud_effect(tmp_path):
    operator,_,_,_,calls = standby(tmp_path)
    operator.state['purpose'] = 'study'
    with pytest.raises(OperatorError,match='ética'):
        operator.failover('us-central1-b')
    assert not calls


def test_lost_reply_with_changed_startup_is_not_admitted(tmp_path):
    operator,_,instances,_,_ = standby(tmp_path,lose_reply=True)
    with pytest.raises(OperatorError):
        operator.failover('us-central1-b')
    instances[-1]['metadata']['items'][0]['value'] = 'different startup'
    with pytest.raises(OperatorError,match='arranque recuperable'):
        operator.failover('us-central1-b')


def test_foreign_vm_after_lost_reply_is_never_stopped_or_adopted(tmp_path):
    operator,_,instances,_,calls = standby(tmp_path,lose_reply=True)
    with pytest.raises(OperatorError):
        operator.failover('us-central1-b')
    instances[-1]['description'] = 'foreign-owner'
    count = len(calls)
    with pytest.raises(OperatorError,match='creación propia'):
        operator.failover('us-central1-b')
    assert not any(args[:3] == ['compute','instances','stop'] and args[3] == instances[-1]['name']
                   for args in calls[count:])
    assert not operator.state.get('alternate_vms')


def test_ip_transfer_gap_margin_is_in_budget_guard(tmp_path):
    operator,_ = installation(tmp_path)
    operator.config['cost'].update(estimated_usd=89.9,margin_usd=0)
    operator.state['ip_transfer_gap_margin_usd'] = .2
    with pytest.raises(OperatorError,match='USD90'):
        operator.reserve_cost('read-only-fixture',0)
