import copy
import json

import pytest

from scripts.study_operator.retention import digest, plan
from scripts.study_operator.retention_cleanup import execute


def fixture():
    original = dict(id='1', name='cloudrag-study-l4-20261002', zone='zones/us-central1-a',
        status='TERMINATED', deletionProtection=True, disks=[dict(autoDelete=False)])
    alternate = dict(id='7', name='cloudrag-i4-test', zone='zones/us-central1-b',
        status='TERMINATED', deletionProtection=True, disks=[dict(autoDelete=False)])
    rows = dict(vms=[original, alternate, dict(id='6', name='cloudrag-i5-cpu', status='TERMINATED')],
        disks=[dict(id='2', name='cloudrag-study-l4-20261002', zone='zones/us-central1-a'),
               dict(id='5', name='cloudrag-i5-cpu-boot', sourceSnapshotId='3'),
               dict(id='8', name='cloudrag-i4-test-boot', zone='zones/us-central1-b', users=['vm/cloudrag-i4-test'])],
        snapshots=[dict(id='3', name='cloudrag-i4-final', status='READY'),
                   dict(id='4', name='cloudrag-i4-old', status='READY')])
    inventory = dict(resources=rows, listing_sha256=digest(rows), protected_ids=dict(vm='1', disk='2'))
    proof = dict(status='CPU_RESTORATION_VERIFIED', synthetic=False, source_snapshot_id='3',
        all_expected_files_verified=True, image_config_verified=True, model_manifest_and_blobs_verified=True,
        restored_disk_id='5', cpu_vm_id='6', source=dict(files=19), artifacts=dict(files=79))
    inherited = [dict(type=kind, id=r['id'], name=r['name'], inherited=True)
        for group, kind in [('vms', 'vm'), ('disks', 'disk'), ('snapshots', 'snapshot')]
        for r in rows[group] if r['name'].startswith('cloudrag-i4-')]
    removal = plan(inventory, '3', proof, inherited_resources=inherited)
    return inventory, proof, removal, inherited


class Cloud:
    def __init__(self, rows):
        self.rows, self.calls = copy.deepcopy(rows), []
        self.lost_response = False

    def command(self, argv, **options):
        assert options['private_output']
        self.calls.append(argv)
        group = {'instances': 'vms', 'disks': 'disks', 'snapshots': 'snapshots'}[argv[1]]
        if argv[2] == 'list':
            name = argv[-1].split('=', 2)[-1]
            return [r for r in self.rows[group] if r['name'] == name]
        target = next(r for r in self.rows[group] if r['name'] == argv[3])
        if argv[2] == 'update':
            assert target['id'] == '7' and '--no-deletion-protection' in argv
            target['deletionProtection'] = False
        else:
            assert argv[2] == 'delete'
            assert target['id'] in {'7', '8', '4'}
            if group == 'vms':
                assert '--keep-disks=all' in argv
                self.rows['disks'][-1]['users'] = []
            self.rows[group].remove(target)
            if self.lost_response:
                self.lost_response = False
                raise RuntimeError('SIMULATED_LOST_DELETE_RESPONSE')


def owner(tmp_path, inherited):
    (tmp_path/'STATE.json').write_text(json.dumps(dict(resources=inherited)))


def test_retention_deletes_exact_redundancy_in_dependency_order_and_resumes(tmp_path):
    inventory, proof, removal, inherited = fixture()
    owner(tmp_path, inherited)
    cloud = Cloud(inventory['resources'])
    result = execute(tmp_path, cloud, inventory, removal, proof)
    assert [r['resource_id'] for r in result['results']] == ['7', '8', '4']
    assert all(r['disposed'] and r['absence_verified'] for r in json.loads((tmp_path/'STATE.json').read_bytes())['resources']
               if r['id'] in {'7', '8', '4'})
    effects = [a for a in cloud.calls if a[2] in {'update', 'delete'}]
    execute(tmp_path, cloud, inventory, removal, proof)
    assert [a for a in cloud.calls if a[2] in {'update', 'delete'}] == effects
    assert cloud.rows['vms'][0]['deletionProtection']


def test_lost_delete_response_reconciles_absence_without_second_delete(tmp_path):
    inventory, proof, removal, inherited = fixture()
    owner(tmp_path, inherited)
    cloud = Cloud(inventory['resources'])
    cloud.lost_response = True
    with pytest.raises(RuntimeError, match='LOST_DELETE'):
        execute(tmp_path, cloud, inventory, removal, proof)
    assert not list(tmp_path.glob('retention-delete-7-receipt.json'))
    execute(tmp_path, cloud, inventory, removal, proof)
    assert sum(a[1:4] == ['instances', 'delete', 'cloudrag-i4-test'] for a in cloud.calls) == 1


@pytest.mark.parametrize('change', ['replacement', 'protected_original', 'attached_disk', 'unaccounted_missing'])
def test_identity_or_protection_defects_stop_deletion(tmp_path, change):
    inventory, proof, removal, inherited = fixture()
    owner(tmp_path, inherited)
    cloud = Cloud(inventory['resources'])
    if change == 'replacement':
        cloud.rows['vms'][1]['id'] = 'different'
    elif change == 'protected_original':
        cloud.rows['vms'][0]['deletionProtection'] = False
    elif change == 'attached_disk':
        cloud.rows['disks'][-1]['users'] = ['foreign-vm']
        removal['removals'] = removal['removals'][1:]
    else:
        cloud.rows['vms'].pop(1)
    with pytest.raises(ValueError):
        execute(tmp_path, cloud, inventory, removal, proof)
    assert not any(a[2] == 'delete' for a in cloud.calls)
