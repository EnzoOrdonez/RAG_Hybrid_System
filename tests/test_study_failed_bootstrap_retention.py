import copy

import pytest

from scripts.study_operator.failed_bootstrap_retention import failed_disk_plan
from scripts.study_operator.retention import digest


def fixture():
    disk = dict(id='8', name='cloudrag-i5-primary-fixture-boot', zone='zones/us-west1-a',
                users=[], description='CloudRAG-I5-fixture', sourceSnapshotId='3')
    rows = dict(vms=[dict(name='original', id='1', status='TERMINATED')], disks=[disk],
                snapshots=[dict(id='3', status='READY')])
    inventory = dict(resources=rows, listing_sha256=digest(rows), protected_ids=dict(vm='1', disk='2'))
    owner = dict(type='disk', id='8', name=disk['name'], zone='us-west1-a', disposable=True, ownership_marker=disk['description'])
    state = dict(resources=[owner])
    intent = dict(name='cloudrag-i5-primary-fixture', disk_name=disk['name'], zone='us-west1-a',
                  source_snapshot_id='3', ownership_marker=disk['description'])
    proof = dict(status='CPU_RESTORATION_VERIFIED', synthetic=False, source_snapshot_id='3',
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
