import pytest

from scripts.study_operator.policy import OperatorError
from scripts.study_operator.prepared_snapshot import prepare
from tests.test_study_operator_lifecycle import installation


def snapshot_cloud(operator,cloud, *, lose_reply=False):
    cloud.vm['disks'][0]['source'] = 'projects/p/zones/us-central1-a/disks/boot'
    live = []
    original = cloud.command
    def command(args,**options):
        if args[:3] == ['compute','disks','describe']:
            return dict(selfLink=cloud.vm['disks'][0]['source'],id='789')
        if args[:3] == ['compute','snapshots','list']:
            return live
        if args[:3] == ['compute','snapshots','create']:
            marker = next(arg.removeprefix('--description=') for arg in args if arg.startswith('--description='))
            live.append(dict(name=args[3],id='456',sourceDiskId='789',status='READY',description=marker,
                storageLocations=['us-central1'],storageBytes=str(2**30),creationTimestamp='2026-10-05T00:00:00+00:00'))
            if lose_reply:
                raise OperatorError('response lost')
        if args[:3] == ['compute','snapshots','describe']:
            return live[0]
        return original(args,**options)
    cloud.command = command
    return live


def test_prepared_snapshot_contains_same_image_certificate_and_no_sessions(tmp_path):
    operator,cloud = installation(tmp_path)
    operator.state['snapshot_empty_verified'] = True
    snapshot_cloud(operator,cloud)
    result = prepare(operator,'1'*64)
    assert result['source_disk_id'] == '789' and result['source_vm_id'] == '123'
    assert result['hostname'] == operator.config['hostname']
    assert result['image_id'] == operator.config['image_id'] and result['certificate_sha256'] == '1'*64
    assert result['idle_usd_day'] == pytest.approx(.000068493*24)
    assert result['retain_at_closure'] and result['session_data'] == 'ALL_I4_PERIODS_EMPTY_VERIFIED'
    assert 'snapshot_empty_verified' not in operator.state
    assert not any(args[:3] == ['compute','instances','start'] for args,_ in cloud.calls)


def test_snapshot_requires_stopped_empty_periods_and_recovers_lost_create_reply(tmp_path):
    operator,cloud = installation(tmp_path)
    with pytest.raises(OperatorError,match='STOP'):
        prepare(operator,'1'*64)
    assert not cloud.calls
    operator.state['snapshot_empty_verified'] = True
    snapshot_cloud(operator,cloud,lose_reply=True)
    with pytest.raises(OperatorError,match='response lost'):
        prepare(operator,'1'*64)
    assert operator.state['snapshot_creation_intent']['source_disk_id'] == '789'
    assert prepare(operator,'1'*64)['id'] == '456'


def test_foreign_snapshot_refused_and_old_ip_snapshot_cannot_failover(tmp_path):
    operator,cloud = installation(tmp_path)
    operator.state['snapshot_empty_verified'] = True
    live = snapshot_cloud(operator,cloud)
    live.append(dict(id='999',sourceDiskId='789',status='READY',description='foreign',storageLocations=['us-central1']))
    with pytest.raises(OperatorError,match='ajena'):
        prepare(operator,'1'*64)
    operator.config['prepared_snapshot'] = dict(name='old-snapshot',id='42',image_id=operator.config['image_id'],
        hostname='203.0.113.9.sslip.io',certificate_sha256='1'*64,session_data='ALL_I4_PERIODS_EMPTY_VERIFIED')
    with pytest.raises(OperatorError,match='desactualizada'):
        operator.failover('us-central1-b')
