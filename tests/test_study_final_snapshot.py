import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_operator import final_snapshot


def fixture(tmp_path):
    folder = tmp_path/'linux-build-final02-evidence'
    folder.mkdir()
    common = dict(image_id='sha256:'+'a'*64, commit='b'*40)
    contents = {
        'receipt.json':dict(common, status='PASS', vm_id='1'),
        'image-id.json':common,
        'host-code-inventory.json':dict(common, files={'scripts/study_operator/deployment.py':'c'*64}),
    }
    rows = []
    for name, value in contents.items():
        data = json.dumps(value).encode()
        (folder/name).write_bytes(data)
        rows.append(dict(object='technical/'+name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    proof = dict(common, status='TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED', cpu_stopped_verified=True, files=rows)
    (tmp_path/'linux-build-final02-download-proof.json').write_text(json.dumps(proof))
    resource = dict(type='vm', name='cloudrag-i5-restore-bootstrap01', id='1', zone='us-central1-a',
                    disposable=True, ownership_marker='owned-vm')
    disk = dict(name='owned-disk', id='2', selfLink='zones/us-central1-a/disks/owned-disk', description='owned-disk', sizeGb='100')
    state = dict(status='ACTIVE', deadline_utc='2099-01-01T00:00:00Z', closure_reserved_utc='2098-12-31T21:00:00Z',
                 independent_closure_verified=True, cloud_cutoff_usd=90,
                 cost=dict(estimated_spend_usd=12, reserved_retention_and_closure_usd=4),
                 resources=[resource, dict(type='disk', id='2', name='owned-disk', disposable=True, ownership_marker='owned-disk')],
                 build_jobs=dict(final02=dict(status=proof['status'], commit=common['commit'], vm_id='1')))
    (tmp_path/'STATE.json').write_text(json.dumps(state))
    (tmp_path/'cpu-restoration-bootstrap01-proof.json').write_text(json.dumps(dict(status='CPU_RESTORATION_VERIFIED', synthetic=False, cpu_vm_id='1')))
    vm = dict(id='1', status='TERMINATED', description='owned-vm', disks=[dict(source=disk['selfLink'], autoDelete=False)])
    return proof, state, vm, disk


def save(tmp_path, name, value):
    (tmp_path/name).write_text(json.dumps(value))


def test_complete_inputs_bind_image_source_and_cpu(tmp_path):
    proof, state, *_ = fixture(tmp_path)
    found, job, host = final_snapshot.inputs(tmp_path, 'final02')
    assert found == proof and job['vm_id'] == '1' and host['image_id'] == proof['image_id']


@pytest.mark.parametrize('case', ['failed', 'not-stopped', 'source', 'host-tamper', 'missing'])
def test_incomplete_or_changed_build_rejected_before_cloud(tmp_path, case):
    proof, state, *_ = fixture(tmp_path)
    if case == 'failed':
        proof['status'] = 'TECHNICAL_BUILD_FAILED_DOWNLOADED_VERIFIED'
    elif case == 'not-stopped':
        proof['cpu_stopped_verified'] = False
    elif case == 'source':
        state['build_jobs']['final02']['commit'] = 'd'*40
    elif case == 'host-tamper':
        (tmp_path/'linux-build-final02-evidence/host-code-inventory.json').write_text('{}')
    else:
        proof['files'] = proof['files'][:-1]
    save(tmp_path, 'linux-build-final02-download-proof.json', proof)
    save(tmp_path, 'STATE.json', state)
    with pytest.raises(ValueError):
        final_snapshot.inputs(tmp_path, 'final02')


@pytest.mark.parametrize('change', [dict(status='RUNNING'), dict(id='replacement'), dict(description='foreign')])
def test_live_cpu_ownership_and_stop_required_before_paid_effect(tmp_path, change):
    _, _, vm, _ = fixture(tmp_path)
    calls = []
    class Cloud:
        def command(self, args, **kwargs):
            calls.append(args)
            return dict(vm, **change)
    with pytest.raises(ValueError, match='observed stopped'):
        final_snapshot.preserve(tmp_path, Cloud(), 'final02', dict(host_infrastructure_sha256={'scripts/study_operator/deployment.py':'d'*64}))
    assert len(calls) == 1 and calls[0][:3] == ['compute', 'instances', 'describe']


def test_foreign_source_disk_rejected_before_snapshot_or_quote(tmp_path):
    _, _, vm, disk = fixture(tmp_path)
    calls = []
    class Cloud:
        def command(self, args, **kwargs):
            calls.append(args)
            return copy.deepcopy(vm if args[1] == 'instances' else dict(disk, id='replacement'))
    with pytest.raises(ValueError, match='source disk'):
        final_snapshot.preserve(tmp_path, Cloud(), 'final02', dict(host_infrastructure_sha256={'scripts/study_operator/deployment.py':'d'*64}))
    assert len(calls) == 2 and all(a[2] == 'describe' for a in calls)


def test_active_new_build_blocks_stale_snapshot_before_cloud(tmp_path):
    _, state, *_ = fixture(tmp_path)
    state['build_jobs']['final03'] = dict(status='RUNNING', vm_id='1', commit='d'*40)
    save(tmp_path, 'STATE.json', state)
    with pytest.raises(ValueError, match='Stale collector'):
        final_snapshot.preserve(tmp_path, None, 'final02', dict(host_infrastructure_sha256={'scripts/study_operator/deployment.py':'d'*64}))


def test_restoration_input_derived_from_verified_image_without_mutating_template(tmp_path):
    proof, *_ = fixture(tmp_path)
    host = final_snapshot.inputs(tmp_path, 'final02')[2]
    template = dict(host_infrastructure_sha256={'scripts/study_operator/deployment.py':'d'*64}, asset_root='retained-assets')
    original = copy.deepcopy(template)
    result = final_snapshot.restoration_config(template, 'final02', proof, host)
    assert template == original and result['asset_root'] == 'retained-assets'
    assert result['image_id'] == proof['image_id'] and result['host_code'].endswith('/code-final02')
    assert result['host_infrastructure_sha256']['scripts/study_operator/deployment.py'] == 'c'*64


def test_snapshot_has_intent_and_cost_before_create_and_never_claims_qualification(tmp_path, monkeypatch):
    proof, state, vm, disk = fixture(tmp_path)
    state['deadline_utc'] = (datetime.now(timezone.utc)+timedelta(hours=2)).isoformat()
    save(tmp_path, 'STATE.json', state)
    monkeypatch.setattr(final_snapshot, 'snapshot_quote', lambda folder:dict(
        usd_per_usage_unit='.05', usage_unit='GiBy.mo', catalog_receipt_sha256='e'*64))
    created = []
    class Cloud:
        def command(self, args, **kwargs):
            if args[1] == 'instances':
                return copy.deepcopy(vm)
            if args[1] == 'disks':
                return copy.deepcopy(disk)
            if args[2] == 'list':
                return created
            if args[2] == 'create':
                before = json.loads((tmp_path/'STATE.json').read_bytes())
                assert before['final_snapshot_intents']['final02']['source_disk_id'] == '2'
                assert before['open_exposures']['final-snapshot-final02']['not_billed_spend']
                assert 'BEFORE final snapshot' in (tmp_path/'COST_LEDGER.md').read_text()
                created.append(dict(id='4', name=args[3], status='READY', sourceDiskId='2',
                    storageLocations=['us-central1'], description=next(a.split('=', 1)[1] for a in args if a.startswith('--description='))))
            if args[2] == 'describe':
                return created[0]
    config = dict(host_infrastructure_sha256={'scripts/study_operator/deployment.py':'d'*64})
    cloud = Cloud()
    result = final_snapshot.preserve(tmp_path, cloud, 'final02', config)
    assert result['status'] == 'FINAL_IMAGE_SNAPSHOT_READY_CPU_RESTORATION_PENDING'
    assert result['live_gpu_acceptance_not_inferred']
    stored = json.loads((tmp_path/'STATE.json').read_bytes())
    assert stored['resources'][-1]['candidate_not_qualified']
    assert final_snapshot.preserve(tmp_path, cloud, 'final02', config) == result
    assert len(created) == 1
