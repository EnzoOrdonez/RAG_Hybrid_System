import hashlib
import json
from pathlib import Path

import pytest

from scripts.study_operator.bootstrap import bootstrap
from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_lifecycle import installation


def fixture(tmp_path, *, lost_reply=False, stockout=False):
    operator, cloud = installation(tmp_path)
    proof = dict(status='CPU_RESTORATION_VERIFIED', synthetic=False, source_snapshot_id='900',
        image_id=operator.config['image_id'], source=dict(files=19, all_expected_files_verified=True),
        all_expected_files_verified=True, image_config_verified=True, model_manifest_and_blobs_verified=True,
        runtime_user_pair=dict(status='PAIRED_RUNTIME_USER_SUPPORTED'))
    path = tmp_path/'restore-proof.json'
    path.write_text(json.dumps(proof))
    operator.config.update(zone='us-central1-b', network='study-net', subnet='study-subnet',
        service_account='fixture-sa', final_snapshot=dict(name='cloudrag-i5-final-fixture', id='900',
            restoration_proof=str(path), restoration_proof_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    operator.config['official_rates']['persistent_disk_gib_usd_h'] = .1/730
    operator.state.update(reserved_address_id='333', ip_ownership_marker='CloudRAG-I5-ip-fixture')
    disks, vms, creates = [], [], []
    original = cloud.command

    def command(args, **options):
        cloud.calls.append((args, options))
        if args[:3] == ['compute', 'snapshots', 'describe']:
            return dict(name='cloudrag-i5-final-fixture', id='900', status='READY')
        if args[:3] == ['compute', 'addresses', 'list']:
            return [dict(name=operator.config['ip_name'], id='333', region='regions/us-central1',
                description='CloudRAG-I5-ip-fixture', address=operator.config['static_ip'])]
        if args[:4] == ['compute', 'networks', 'subnets', 'describe']:
            return dict(network='networks/study-net', region='regions/us-central1', privateIpGoogleAccess=True)
        if args[:3] == ['compute', 'instances', 'list']:
            return [cloud.vm, *vms]
        if args[:3] == ['compute', 'disks', 'list']:
            return disks
        if args[:3] == ['compute', 'disks', 'create']:
            marker = next(a.split('=', 1)[1] for a in args if a.startswith('--description='))
            disks.append(dict(name=args[3], id='777', zone='zones/us-central1-b', sourceSnapshotId='900',
                description=marker, selfLink='disks/'+args[3], creationTimestamp='2026-10-05T00:00:00+00:00'))
            return
        if args[:3] == ['compute', 'instances', 'create']:
            saved = json.loads((operator.root/'active.json').read_bytes())
            assert saved['audit_resources'][0]['id'] == '777'
            assert saved['primary_creation_intent']['disk_name'] == disks[0]['name']
            assert '--max-run-duration=3h' in args and '--instance-termination-action=STOP' in args
            assert '--deletion-protection' in args and '--scopes=storage-rw' in args
            assert '--provisioning-model=STANDARD' in args
            assert 'auto-delete=no' in next(a for a in args if a.startswith('--disk='))
            creates.append(args)
            if stockout:
                raise OperatorError('ZONE_RESOURCE_POOL_EXHAUSTED')
            script = next(a.split('=', 2)[2] for a in args if a.startswith('--metadata-from-file=startup-script='))
            marker = next(a.split('=', 1)[1] for a in args if a.startswith('--description='))
            vms.append(dict(name=args[3], id='998', zone='zones/us-central1-b', status='RUNNING',
                machineType='machines/g2-standard-4', deletionProtection=True,
                disks=[dict(autoDelete=False, source=disks[0]['selfLink'])], description=marker,
                scheduling=dict(maxRunDuration=dict(seconds='10800'), instanceTerminationAction='STOP'),
                creationTimestamp='2026-10-05T00:00:00+00:00',
                metadata=dict(items=[dict(key='startup-script', value=Path(script).read_text())])))
            if lost_reply:
                raise OperatorError('CREATE_RESPONSE_LOST')
            return
        if args[:3] == ['compute', 'instances', 'describe'] and vms and args[3] == vms[0]['name']:
            return vms[0]
        return original(args, **options)

    cloud.command = command
    return operator, cloud, disks, vms, creates, path


def test_bootstrap_preserves_original_and_qualified_image_without_stopping_acquired_gpu(tmp_path):
    operator, cloud, disks, vms, creates, _ = fixture(tmp_path)
    original = dict(operator.config['primary_vm'])
    result = bootstrap(operator, 'us-central1-b')
    assert result['status'] == 'FINAL_PRIMARY_CREATED_NOT_READY'
    assert operator.config['original_preserved_vm'] == original
    assert operator.config['primary_vm']['id'] == vms[0]['id']
    assert disks[0]['sourceSnapshotId'] == '900' and vms[0]['status'] == 'RUNNING'
    assert not operator.state['ready_verified']
    assert bootstrap(operator, 'us-central1-b')['status'] == 'PRIMARY_ALREADY_CREATED'
    assert len(creates) == 1
    assert not any(a[:3] == ['compute', 'instances', 'stop'] and a[3] == vms[0]['name'] for a, _ in cloud.calls)


def test_bootstrap_stockout_preserves_disk_intent_and_original_selection(tmp_path):
    operator, _, disks, vms, creates, _ = fixture(tmp_path, stockout=True)
    original = dict(operator.config['primary_vm'])
    with pytest.raises(OperatorError, match='ZONE_RESOURCE_POOL_EXHAUSTED'):
        bootstrap(operator, 'us-central1-b')
    assert operator.config['primary_vm'] == original
    assert operator.state['audit_resources'][0]['id'] == disks[0]['id']
    assert operator.state['primary_creation_intent']['ownership_marker'] == disks[0]['description']
    assert not vms and len(creates) == 1


def test_bootstrap_recovers_lost_reply_without_second_create(tmp_path):
    operator, _, _, vms, creates, _ = fixture(tmp_path, lost_reply=True)
    with pytest.raises(OperatorError, match='CREATE_RESPONSE_LOST'):
        bootstrap(operator, 'us-central1-b')
    assert bootstrap(operator, 'us-central1-b')['vm_id'] == vms[0]['id']
    assert len(creates) == 1


@pytest.mark.parametrize('tamper', ['proof', 'synthetic', 'zone', 'image'])
def test_bootstrap_rejects_bad_qualification_before_paid_effect(tmp_path, tamper):
    operator, cloud, _, _, creates, path = fixture(tmp_path)
    zone = 'us-central1-b'
    if tamper == 'proof':
        path.write_bytes(b'altered')
    elif tamper == 'synthetic':
        proof = json.loads(path.read_bytes())
        proof['synthetic'] = True
        path.write_text(json.dumps(proof))
        operator.config['final_snapshot']['restoration_proof_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif tamper == 'zone':
        zone = 'europe-west4-a'
    else:
        operator.config['image_id'] = 'sha256:'+'0'*64
    with pytest.raises(OperatorError):
        bootstrap(operator, zone)
    assert not creates and not any(a[2] == 'create' for a, _ in cloud.calls if len(a) > 2)
