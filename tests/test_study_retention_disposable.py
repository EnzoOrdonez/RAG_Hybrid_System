import copy
import json

import pytest

from scripts.study_operator.retention import digest
from scripts.study_operator.retention_cleanup import execute
from scripts.study_operator.retention_disposable import controller_job, owned_plan


def fixture():
    original = dict(id='1', name='cloudrag-study-l4-20261002', zone='zones/us-central1-a',
        status='TERMINATED', deletionProtection=True, disks=[dict(autoDelete=False, source='disk/original')])
    cpu = dict(id='6', name='cloudrag-i5-cpu', zone='zones/us-central1-a',
        machineType='types/e2-standard-2', description='CloudRAG-I5-cpu',
        status='TERMINATED', deletionProtection=True, disks=[dict(autoDelete=False, source='disk/cpu')])
    rows = dict(vms=[original, cpu], disks=[
        dict(id='2', name=original['name'], zone=original['zone'], selfLink='disk/original'),
        dict(id='5', name='cloudrag-i5-cpu-boot', zone=cpu['zone'], selfLink='disk/cpu',
             sourceSnapshotId='3', description='CloudRAG-I5-cpu', users=['vm/cloudrag-i5-cpu']),
        dict(id='8', name='cloudrag-i5-capacity-boot', zone='zones/us-west1-a',
             description='CloudRAG-I5-capacity', users=[])], snapshots=[
        dict(id='3', name='cloudrag-i5-final', status='READY', storageLocations=['us-central1']),
        dict(id='4', name='cloudrag-i4-old', status='READY')])
    inventory = dict(resources=rows, listing_sha256=digest(rows), protected_ids=dict(vm='1', disk='2'))
    proof = dict(status='CPU_RESTORATION_VERIFIED', synthetic=False, source_snapshot_id='3',
        all_expected_files_verified=True, image_config_verified=True, model_manifest_and_blobs_verified=True,
        restored_disk_id='5', cpu_vm_id='6', source=dict(files=19), artifacts=dict(files=79), image_id='sha256:final',
        runtime_user_pair=dict(status='PAIRED_RUNTIME_USER_SUPPORTED'))
    owners = [dict(type=kind, id=r['id'], name=r['name'], zone=r.get('zone', '').split('/')[-1],
                   disposable=r['name'].startswith('cloudrag-i5-'), inherited=r['name'].startswith('cloudrag-i4-'),
                   ownership_marker=r.get('description'))
        for group, kind in [('vms', 'vm'), ('disks', 'disk'), ('snapshots', 'snapshot')] for r in rows[group]]
    final = dict(image_id=proof['image_id'], snapshot=dict(id='3'))
    return inventory, proof, owners, final


class Cloud:
    def __init__(self, rows):
        self.rows, self.effects = copy.deepcopy(rows), []
        self.lost_response = False

    def command(self, argv, **kwargs):
        assert kwargs['private_output']
        group = {'instances': 'vms', 'disks': 'disks', 'snapshots': 'snapshots'}[argv[1]]
        if argv[2] == 'list':
            return [r for r in self.rows[group] if r['name'] == argv[-1].split('=', 2)[-1]]
        target = next(r for r in self.rows[group] if r['name'] == argv[3])
        assert target['id'] in {'6', '5', '8', '4'}
        self.effects.append(argv)
        if argv[2] == 'update':
            target['deletionProtection'] = False
            return
        assert argv[2] == 'delete'
        if group == 'vms':
            assert '--keep-disks=all' in argv
            self.rows['disks'][1]['users'] = []
        self.rows[group].remove(target)
        if self.lost_response:
            self.lost_response = False
            raise RuntimeError('LOST_DELETE_RESPONSE')


def test_declared_cpu_retired_after_final_restoration_and_resumption(tmp_path):
    inventory, proof, owners, final = fixture()
    plan = owned_plan(inventory, proof, owners, final)
    assert [r['id'] for r in plan['removals']] == ['6', '5', '8', '4']
    (tmp_path/'STATE.json').write_text(json.dumps(dict(resources=owners)))
    cloud = Cloud(inventory['resources'])
    result = execute(tmp_path, cloud, inventory, plan, proof)
    assert result['qualified_snapshot_id'] == '3'
    effects = cloud.effects[:]
    execute(tmp_path, cloud, inventory, plan, proof)
    assert cloud.effects == effects
    assert cloud.rows['vms'][0]['id'] == '1' and cloud.rows['vms'][0]['deletionProtection']
    assert [r['id'] for r in cloud.rows['disks']] == ['2']
    assert [r['id'] for r in cloud.rows['snapshots']] == ['3']


def test_own_cpu_lost_delete_reply_reconciles_without_repeating(tmp_path):
    inventory, proof, owners, final = fixture()
    plan = owned_plan(inventory, proof, owners, final)
    (tmp_path/'STATE.json').write_text(json.dumps(dict(resources=owners)))
    cloud = Cloud(inventory['resources'])
    cloud.lost_response = True
    with pytest.raises(RuntimeError, match='LOST_DELETE'):
        execute(tmp_path, cloud, inventory, plan, proof)
    execute(tmp_path, cloud, inventory, plan, proof)
    assert sum(a[1:4] == ['instances', 'delete', 'cloudrag-i5-cpu'] for a in cloud.effects) == 1


@pytest.mark.parametrize('defect', ['synthetic', 'image', 'runtime', 'snapshot', 'cpu_running',
    'cpu_machine', 'source_snapshot', 'auto_delete', 'owner_missing', 'owner_marker', 'owner_zone',
    'foreign_user', 'duplicate_owner', 'listing'])
def test_unqualified_or_unowned_plan_fails_closed(defect):
    inventory, proof, owners, final = fixture()
    rows = inventory['resources']
    if defect == 'synthetic':
        proof['synthetic'] = True
    elif defect == 'image':
        final['image_id'] = 'other'
    elif defect == 'runtime':
        proof['runtime_user_pair']['status'] = 'NOT_SUPPORTED'
    elif defect == 'snapshot':
        rows['snapshots'][0]['status'] = 'CREATING'
    elif defect == 'cpu_running':
        rows['vms'][1]['status'] = 'RUNNING'
    elif defect == 'cpu_machine':
        rows['vms'][1]['machineType'] = 'types/g2-standard-4'
    elif defect == 'source_snapshot':
        rows['disks'][1]['sourceSnapshotId'] = 'wrong'
    elif defect == 'auto_delete':
        rows['vms'][1]['disks'][0]['autoDelete'] = True
    elif defect == 'owner_missing':
        owners.pop(1)
    elif defect == 'owner_marker':
        owners[1]['ownership_marker'] = 'foreign'
    elif defect == 'owner_zone':
        owners[1]['zone'] = 'us-east1-b'
    elif defect == 'foreign_user':
        rows['disks'][1]['users'] = ['vm/foreign']
    elif defect == 'duplicate_owner':
        owners.append(owners[1].copy())
    else:
        inventory['listing_sha256'] = 'changed'
    if defect != 'listing':
        inventory['listing_sha256'] = digest(rows)
    with pytest.raises(ValueError):
        owned_plan(inventory, proof, owners, final)


def test_ownership_changed_after_plan_prevents_delete(tmp_path):
    inventory, proof, owners, final = fixture()
    plan = owned_plan(inventory, proof, owners, final)
    (tmp_path/'STATE.json').write_text(json.dumps(dict(resources=owners)))
    cloud = Cloud(inventory['resources'])
    cloud.rows['vms'][1]['description'] = 'changed'
    with pytest.raises(ValueError, match='marker changed'):
        execute(tmp_path, cloud, inventory, plan, proof)
    assert not cloud.effects


def test_original_and_final_snapshot_cannot_be_injected_into_removals(tmp_path):
    inventory, proof, owners, final = fixture()
    (tmp_path/'STATE.json').write_text(json.dumps(dict(resources=owners)))
    for group, index in [('vms', 0), ('disks', 0), ('snapshots', 0)]:
        plan = owned_plan(inventory, proof, owners, final)
        plan['removals'].insert(0, dict(resource_kind=group, **inventory['resources'][group][index]))
        cloud = Cloud(inventory['resources'])
        with pytest.raises(ValueError, match='protected IDs'):
            execute(tmp_path, cloud, inventory, plan, proof)
        assert not cloud.effects


@pytest.mark.parametrize('command,accepted', [
    (['python', '-B', '-m', 'scripts.study_operator.cpu_restoration'], True),
    (['python', '-c', 'from scripts.study_operator.cpu_restoration import main; main(["--label", "fixture"])'], True),
    (['python', '-c', 'print("from scripts.study_operator.cpu_restoration import main; main([)")'], False),
    (['python', '-c', 'from scripts.study_operator.cpu_restoration import main; print("main([)")'], False),
    (['python', '-c', 'if False:\n from scripts.study_operator.cpu_restoration import main\n main([])'], False),
    (['python', '-c', 'invalid('], False),
    (['python', 'scripts.study_operator.cpu_restoration'], False),
])
def test_job_must_actually_invoke_controller(command, accepted):
    job = dict(status='PASS', limited_token=True, results=[dict(command=command, exit_code=0)])
    assert controller_job(job) is accepted
    job['limited_token'] = False
    assert not controller_job(job)
    job['limited_token'] = True
    job['results'][0]['exit_code'] = 1
    assert not controller_job(job)
