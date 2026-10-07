import json

import pytest

from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_lifecycle import installation


def fixture(tmp_path, *, lost_reply=False, stockout=False):
    operator, cloud = installation(tmp_path)
    operator.config.update(network='cloudrag-study-20261002', subnet='cloudrag-study-central1-20261002',
        service_account='cloudrag-study-i4@pure-loop-474323-a8.iam.gserviceaccount.com',
        prepared_snapshot=dict(name='cloudrag-i5-ready-fixture', id='900', image_id=operator.config['image_id'],
            hostname=operator.config['hostname'], certificate_sha256='1'*64, session_data='ALL_I4_PERIODS_EMPTY_VERIFIED'))
    operator.state.update(purpose='smoke', failover_data_reconciled=True, ready=dict(tls=dict(certificate_sha256='1'*64)))
    disks, vms, creates = [], [], []
    original = cloud.command

    def command(args, **options):
        cloud.calls.append((args, options))
        if args[:3] == ['compute', 'instances', 'list']:
            return [cloud.vm, *vms]
        if args[:3] == ['compute', 'snapshots', 'describe']:
            return dict(name='cloudrag-i5-ready-fixture', id='900', status='READY', storageLocations=['us-central1'])
        if args[:3] == ['compute', 'disks', 'list']:
            return disks
        if args[:3] == ['compute', 'disks', 'describe']:
            return disks[0]
        if args[:3] == ['compute', 'disks', 'create']:
            marker = next(a.split('=', 1)[1] for a in args if a.startswith('--description='))
            disks.append(dict(name=args[3], id='777', zone='zones/us-central1-b', sourceSnapshotId='900',
                description=marker, selfLink='disks/'+args[3], creationTimestamp='2026-10-05T00:00:00+00:00'))
            return
        if args[:3] == ['compute', 'instances', 'create']:
            # Must already be durable even if this API call fails or loses its reply.
            saved = json.loads((operator.root/'active.json').read_bytes())
            assert saved['audit_resources'][0]['id'] == '777'
            assert saved['alternate_creation_intent']['disk_name'] == disks[0]['name']
            assert '--max-run-duration=3h' in args and '--instance-termination-action=STOP' in args
            assert '--scopes=storage-rw' in args and '--provisioning-model=STANDARD' in args
            assert 'auto-delete=no' in next(a for a in args if a.startswith('--disk='))
            marker = next(a.split('=', 1)[1] for a in args if a.startswith('--description='))
            assert marker.startswith('CloudRAG-I5-') and args[3].startswith('cloudrag-i5-')
            creates.append(args)
            if stockout:
                raise OperatorError('ZONE_RESOURCE_POOL_EXHAUSTED')
            script = next(a.split('=', 2)[2] for a in args if a.startswith('--metadata-from-file=startup-script='))
            from pathlib import Path

            vms.append(dict(name=args[3], id='998', zone='zones/us-central1-b', status='RUNNING',
                machineType='types/g2-standard-4', deletionProtection=True,
                disks=[dict(autoDelete=False, source=disks[0]['selfLink'])], description=marker,
                scheduling=dict(instanceTerminationAction='STOP', maxRunDuration=dict(seconds='10800')),
                creationTimestamp='2026-10-05T00:00:00+00:00',
                metadata=dict(items=[dict(key='startup-script', value=Path(script).read_text())])))
            if lost_reply:
                raise OperatorError('CREATE_RESPONSE_LOST')
            return
        if args[:3] == ['compute', 'instances', 'describe'] and vms and args[3] == vms[0]['name']:
            return vms[0]
        return original(args, **options)

    cloud.command = command
    return operator, cloud, disks, vms, creates


def test_alternate_records_actual_disk_before_vm_and_keeps_acquired_capacity(tmp_path):
    operator, cloud, disks, vms, creates = fixture(tmp_path)
    result = operator.failover('us-central1-b')
    assert result['status'] == 'ALTERNATE_STARTED_SUPERVISED'
    assert operator.state['selected_vm']['id'] == '998'
    assert operator.state['alternate_vms'][0]['disk_id'] == '777'
    assert vms[0]['status'] == 'RUNNING'
    assert 'preflight' in result['next_action'] and not operator.state['ready_verified']
    assert len(creates) == len(disks) == 1
    assert not any(a[:3] == ['compute', 'instances', 'stop'] and a[3] == vms[0]['name'] for a, _ in cloud.calls)


def test_stockout_keeps_owned_disk_and_creation_intent_for_reconciliation(tmp_path):
    operator, _, disks, vms, creates = fixture(tmp_path, stockout=True)
    with pytest.raises(OperatorError, match='ZONE_RESOURCE_POOL_EXHAUSTED'):
        operator.failover('us-central1-b')
    saved = json.loads((operator.root/'active.json').read_bytes())
    assert saved['audit_resources'][0]['id'] == disks[0]['id']
    assert saved['alternate_creation_intent']['ownership_marker'] == disks[0]['description']
    assert not vms and len(creates) == 1


def test_lost_vm_create_reply_recovers_same_running_id_without_recreating(tmp_path):
    operator, _, _, vms, creates = fixture(tmp_path, lost_reply=True)
    with pytest.raises(OperatorError, match='CREATE_RESPONSE_LOST'):
        operator.failover('us-central1-b')
    result = operator.failover('us-central1-b')
    assert result['status'] == 'ALTERNATE_STARTED_SUPERVISED'
    assert operator.state['selected_vm']['id'] == vms[0]['id']
    assert operator.state['alternate_vms'][0]['disk_id'] == '777'
    assert len(creates) == 1
