import copy
import json

import pytest

from scripts.study_operator.failed_bootstrap_retention import failed_disk_plan
from scripts.study_operator.retention import digest
from scripts.study_operator.retention_cleanup import execute


def fixture():
    disk = dict(id='8', name='cloudrag-i5-primary-fixture-boot', zone='zones/us-west1-a',
                users=[], description='CloudRAG-I5-fixture', sourceSnapshotId='3')
    rows = dict(vms=[dict(name='original', id='1', status='TERMINATED')], disks=[disk],
                snapshots=[dict(id='3', status='READY')])
    rows['vms'][0].update(zone='zones/us-central1-a', deletionProtection=True, disks=[dict(autoDelete=False)])
    rows['disks'].append(dict(name='original', id='2', zone='zones/us-central1-a'))
    rows['snapshots'][0]['name'] = 'qualified-final'
    inventory = dict(resources=rows, listing_sha256=digest(rows), protected_ids=dict(vm='1', disk='2'))
    owner = dict(type='disk', id='8', name=disk['name'], zone='us-west1-a', disposable=True, ownership_marker=disk['description'])
    state = dict(resources=[owner])
    intent = dict(name='cloudrag-i5-primary-fixture', disk_name=disk['name'], zone='us-west1-a',
                  source_snapshot_id='3', ownership_marker=disk['description'])
    proof = dict(status='CPU_RESTORATION_VERIFIED', synthetic=False, source_snapshot_id='3',
        cpu_vm_id='6', restored_disk_id='5',
        source=dict(files=19), artifacts=dict(files=79), all_expected_files_verified=True,
        image_config_verified=True, model_manifest_and_blobs_verified=True,
        runtime_user_pair=dict(status='PAIRED_RUNTIME_USER_SUPPORTED'))
    job = dict(status='PASS', limited_token=True, results=[dict(exit_code=0, command=['python', '-m', 'scripts.study_operator.cpu_restoration'])])
    return inventory, state, intent, proof, job


def test_archived_real_restoration_allows_only_failed_unattached_disk():
    args = fixture()
    before = copy.deepcopy(args)
    plan = failed_disk_plan(*args)
    assert [r['id'] for r in plan['removals']] == ['8'] and plan['failed_cause_not_reclassified'] is True
    assert args == before


def test_generated_plan_runs_through_shared_executor_and_verifies_absence(tmp_path):
    inventory, state, intent, proof, job = fixture()
    plan = failed_disk_plan(inventory, state, intent, proof, job)
    (tmp_path/'STATE.json').write_text(json.dumps(state))
    class FakeCloud:
        def __init__(self):
            self.rows = copy.deepcopy(inventory['resources'])
            self.deleted = []

        def command(self, argv, **kwargs):
            groups = {'instances': 'vms', 'disks': 'disks', 'snapshots': 'snapshots'}
            group = groups[argv[1]]
            if argv[2] == 'list':
                name = argv[-1].split('=', 2)[-1]
                return [row for row in self.rows[group] if row['name'] == name]
            assert argv[1:3] == ['disks', 'delete'] and argv[3] == intent['disk_name']
            self.deleted.append(argv)
            self.rows['disks'] = [row for row in self.rows['disks'] if row['name'] != argv[3]]

    cloud = FakeCloud()
    result = execute(tmp_path, cloud, inventory, plan, proof)
    assert result['status'] == 'REDUNDANT_RETENTION_ABSENCE_VERIFIED' and len(cloud.deleted) == 1
    assert json.loads((tmp_path/'STATE.json').read_bytes())['resources'][0]['absence_verified'] is True
    assert execute(tmp_path, cloud, inventory, plan, proof)['results'] == result['results']
    assert len(cloud.deleted) == 1


@pytest.mark.parametrize('defect', ['foreign_id', 'attached', 'creation_present', 'gpu_active', 'wrong_snapshot',
    'wrong_marker', 'disposed', 'unapproved', 'synthetic_proof', 'failed_job', 'missing_snapshot', 'tampered_inventory'])
def test_disposal_fail_closed_without_live_ownership_and_qualified_restoration(defect):
    inventory, state, intent, proof, job = fixture()
    rows = inventory['resources']
    if defect == 'foreign_id':
        state['resources'][0]['id'] = 'other'
    elif defect == 'attached':
        rows['disks'][0]['users'] = ['vm/other']
    elif defect == 'creation_present':
        rows['vms'].append(dict(name=intent['name'], id='9', status='TERMINATED'))
    elif defect == 'gpu_active':
        rows['vms'][0].update(status='RUNNING', machineType='types/g2-standard-4')
    elif defect == 'wrong_snapshot':
        rows['disks'][0]['sourceSnapshotId'] = 'other'
    elif defect == 'wrong_marker':
        rows['disks'][0]['description'] = 'foreign'
    elif defect == 'disposed':
        state['resources'][0]['disposed'] = True
    elif defect == 'unapproved':
        state['resources'][0]['disposable'] = False
    elif defect == 'synthetic_proof':
        proof['synthetic'] = True
    elif defect == 'failed_job':
        job['status'] = 'FAILED'
    elif defect == 'missing_snapshot':
        rows['snapshots'] = []
    if defect != 'tampered_inventory':
        inventory['listing_sha256'] = digest(rows)
    else:
        rows['disks'][0]['name'] += '-changed'
    with pytest.raises((ValueError, RuntimeError)):
        failed_disk_plan(inventory, state, intent, proof, job)
