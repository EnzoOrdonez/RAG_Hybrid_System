"""ID-pinned, resumable disposal of expressly authorized redundant I4 resources."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.retention import census, digest, plan
from scripts.study_operator.run_control import require_limited
from src.ui.components.session_storage import atomic_json

GROUP = {'vms': 'instances', 'disks': 'disks', 'snapshots': 'snapshots'}
TYPE = {'vms': 'vm', 'disks': 'disk', 'snapshots': 'snapshot'}


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError('Immutable retention evidence differs; no replay')
    else:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2)


def selected(cloud, kind, row):
    rows = cloud.command(['compute', GROUP[kind], 'list', '--filter=name='+row['name']], private_output=True)
    if len(rows) > 1 or rows and str(rows[0]['id']) != str(row['id']):
        raise ValueError('Live deletion identity differs; never delete by name alone')
    if rows and kind != 'snapshots' and rows[0]['zone'].split('/')[-1] != row['zone'].split('/')[-1]:
        raise ValueError('Live deletion zone differs')
    return rows[0] if rows else None


def protections(cloud, inventory, removal_plan):
    """Check original and the qualified recovery snapshot before every effect."""
    for kind, identity in [('vms', removal_plan['preserved_ids']['vm']),
                           ('disks', removal_plan['preserved_ids']['disk']),
                           ('snapshots', removal_plan['qualified_snapshot_id'])]:
        row = next(r for r in inventory['resources'][kind] if str(r['id']) == identity)
        live = selected(cloud, kind, row)
        if live is None:
            raise ValueError('Protected reconstruction resource missing')
        if kind == 'vms' and (live['status'] != 'TERMINATED' or not live['deletionProtection']
                              or live['disks'][0]['autoDelete'] is not False):
            raise ValueError('Original VM protections changed')
        if kind == 'snapshots' and live['status'] != 'READY':
            raise ValueError('Qualified snapshot no longer READY')


def execute(root, cloud, inventory, removal_plan, restoration):
    root = Path(root)
    if (removal_plan['listing_sha256'] != inventory['listing_sha256']
            or digest(inventory['resources']) != inventory['listing_sha256']
            or digest(restoration) != removal_plan['restoration_receipt_sha256']):
        raise ValueError('Pinned inventory/restoration changed')
    results = []
    for row in removal_plan['removals']:
        kind = row['type']
        if kind not in GROUP or str(row['id']) in {
                *removal_plan['preserved_ids'].values(), removal_plan['qualified_snapshot_id'],
                removal_plan['cpu_vm_id'], removal_plan['restored_disk_id']}:
            raise ValueError('Removal intersects protected IDs')
        if not row['name'].startswith('cloudrag-i4-'):
            raise ValueError('Removal outside authorized I4 namespace')
        intent = root/('retention-delete-'+str(row['id'])+'-intent.json')
        receipt_path = root/('retention-delete-'+str(row['id'])+'-receipt.json')
        protections(cloud, inventory, removal_plan)
        live = selected(cloud, kind, row)
        if live is None and not intent.exists():
            raise ValueError('Unaccounted missing resource; no deletion success inferred')
        if not intent.exists():
            immutable(intent, dict(resource=row, listing_sha256=inventory['listing_sha256'],
                restoration_receipt_sha256=digest(restoration), at=datetime.now(timezone.utc).isoformat(),
                authorization='Iteration5 clauses0.4/70; synthetic CPU reconstruction passed'))
            with (root/'DESTRUCTION_LOG.md').open('a', encoding='utf-8') as stream:
                stream.write('\nBEFORE deletion: '+json.dumps(row, sort_keys=True)+'; intent='+intent.name+'\n')
        elif json.loads(intent.read_bytes())['resource'] != row:
            raise ValueError('Prior deletion intent identity differs')
        zone = ['--zone='+row['zone'].split('/')[-1]] if kind != 'snapshots' else []
        if live is not None:
            if receipt_path.exists():
                raise ValueError('Previously deleted resource reappeared')
            if kind == 'vms':
                if live['status'] != 'TERMINATED' or live['disks'][0]['autoDelete'] is not False:
                    raise ValueError('Disposable VM must be stopped with retained disk')
                if live['deletionProtection']:
                    cloud.command(['compute', 'instances', 'update', row['name'], *zone,
                                   '--no-deletion-protection'], private_output=True)
                    live = selected(cloud, kind, row)
                    if live is None or live['deletionProtection'] or live['status'] != 'TERMINATED':
                        raise ValueError('Disposable protection update not verified')
            elif kind == 'disks' and live.get('users'):
                raise ValueError('Disposable disk remains attached')
            protections(cloud, inventory, removal_plan)
            cloud.command(['compute', GROUP[kind], 'delete', row['name'], *zone,
                           *(['--keep-disks=all'] if kind == 'vms' else [])], private_output=True, timeout=600)
        if selected(cloud, kind, row) is not None:
            raise ValueError('Resource absence not verified after delete')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
        else:
            receipt = dict(status='RESOURCE_ABSENCE_VERIFIED', resource_id=str(row['id']),
                           type=TYPE[kind], name=row['name'], at=datetime.now(timezone.utc).isoformat(),
                           intent_sha256=digest(json.loads(intent.read_bytes())))
            immutable(receipt_path, receipt)
        with FileLock(str(root/'state.lock'), timeout=10):
            state_path = root/'STATE.json'
            state = json.loads(state_path.read_bytes())
            matches = [r for r in state['resources'] if r['type'] == TYPE[kind] and str(r['id']) == str(row['id'])]
            if len(matches) != 1:
                raise ValueError('Disposed resource missing from owner state')
            matches[0].update(disposed=True, absence_verified=True, disposal_receipt=str(receipt_path))
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(state_path, state)
        results.append(receipt)
    protections(cloud, inventory, removal_plan)
    return dict(status='REDUNDANT_RETENTION_ABSENCE_VERIFIED', results=results,
                preserved_ids=removal_plan['preserved_ids'], qualified_snapshot_id=removal_plan['qualified_snapshot_id'],
                original_buckets_and_file_evidence_not_deleted=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('package', 'sdk', 'proof', 'proof-job', 'inventory', 'plan'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require_limited()
    root = Path(args.package)
    with FileLock(str(root/'retention-cleanup.lock'), timeout=0):
        proof = json.loads(Path(args.proof).read_bytes())
        job = json.loads(Path(args.proof_job).read_bytes())
        if job['status'] != 'PASS' or not job['limited_token'] or not any(
                'scripts.study_operator.cpu_restoration' in r['command'] and r['exit_code'] == 0
                for r in job['results']):
            raise ValueError('Successful actual CPU-controller job required')
        state = json.loads((root/'STATE.json').read_bytes())
        cloud = Cloud(args.sdk, 'pure-loop-474323-a8', root/'retention-cleanup-api')
        if Path(args.inventory).exists():
            inventory = json.loads(Path(args.inventory).read_bytes())
        else:
            inventory = census(cloud, state)
            immutable(args.inventory, inventory)
        removal_plan = plan(inventory, proof['source_snapshot_id'], proof, inherited_resources=state['resources'])
        immutable(args.plan, removal_plan)
        if args.execute:
            result = execute(root, cloud, inventory, removal_plan, proof)
            target = root/('retention-cleanup-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
            immutable(target, result)
            print(json.dumps(dict(status=result['status'], receipt=str(target), disposed=len(result['results']))))
        else:
            print(json.dumps(dict(status='DRY_RUN_NO_DELETION', planned=len(removal_plan['removals']))))


if __name__ == '__main__':
    main()
