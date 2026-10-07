"""Retire declared I5 test resources after the final snapshot's real CPU proof."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.retention import census, digest
from scripts.study_operator.retention_cleanup import execute, immutable
from scripts.study_operator.run_control import require_limited


def controller_job(job):
    if job.get('status') != 'PASS' or job.get('limited_token') is not True:
        return False
    for row in job.get('results', []):
        args = row.get('command', [])
        if row.get('exit_code') != 0:
            continue
        if '-m' in args and args[args.index('-m')+1] == 'scripts.study_operator.cpu_restoration':
            return True
        if '-c' in args:
            try:
                nodes = ast.parse(args[args.index('-c')+1]).body
            except (SyntaxError, IndexError):
                continue
            imported = any(isinstance(n, ast.ImportFrom) and n.module == 'scripts.study_operator.cpu_restoration'
                and any(a.name == 'main' and a.asname is None for a in n.names) for n in nodes)
            invoked = any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id == 'main'
                and n.value.args and isinstance(n.value.args[0], ast.List) for n in nodes)
            if imported and invoked:
                return True
    return False


def owned_plan(inventory, proof, resources, final):
    rows = inventory['resources']
    if digest(rows) != inventory['listing_sha256']:
        raise ValueError('Full retention inventory changed')
    if (proof.get('status') != 'CPU_RESTORATION_VERIFIED' or proof.get('synthetic') is not False
            or not all(proof.get(key) for key in ('all_expected_files_verified', 'image_config_verified', 'model_manifest_and_blobs_verified'))
            or proof.get('source', {}).get('files') != 19 or proof.get('artifacts', {}).get('files') != 79
            or proof.get('runtime_user_pair', {}).get('status') != 'PAIRED_RUNTIME_USER_SUPPORTED'
            or proof.get('image_id') != final.get('image_id')
            or str(proof.get('source_snapshot_id')) != str(final['snapshot']['id'])):
        raise ValueError('Actual final-image restoration and supported runtime pair required')
    snapshot_id = str(proof['source_snapshot_id'])
    snapshots = [r for r in rows['snapshots'] if str(r['id']) == snapshot_id]
    cpus = [r for r in rows['vms'] if str(r['id']) == proof['cpu_vm_id']]
    disks = [r for r in rows['disks'] if str(r['id']) == proof['restored_disk_id']]
    if (len(snapshots) != 1 or snapshots[0]['status'] != 'READY'
            or snapshots[0].get('storageLocations') != ['us-central1']
            or len(cpus) != 1 or cpus[0]['status'] != 'TERMINATED'
            or cpus[0]['machineType'].split('/')[-1] != 'e2-standard-2'
            or len(disks) != 1 or str(disks[0].get('sourceSnapshotId')) != snapshot_id
            or cpus[0]['disks'][0]['source'] != disks[0]['selfLink']
            or cpus[0]['disks'][0].get('autoDelete') is not False):
        raise ValueError('Live stopped CPU, restored disk and READY final snapshot required')
    protected = inventory['protected_ids']
    keep = {*protected.values(), snapshot_id}
    owners = {(r['type'], str(r['id'])): r for r in resources}
    if len(owners) != len(resources):
        raise ValueError('Duplicate owner identities')
    markers, removals = {}, []
    for group, kind in [('vms', 'vm'), ('disks', 'disk'), ('snapshots', 'snapshot')]:
        for row in rows[group]:
            identity = str(row['id'])
            if identity in keep:
                continue
            owner = owners.get((kind, identity))
            if not owner or owner.get('disposed') or owner['name'] != row['name']:
                raise ValueError('Unaccounted live resource; no deletion plan')
            own = owner.get('disposable') is True and row['name'].startswith('cloudrag-i5-')
            inherited = owner.get('inherited') is True and row['name'].startswith('cloudrag-i4-')
            if own:
                marker = owner.get('ownership_marker', '')
                if not marker.startswith('CloudRAG-I5-') or row.get('description') != marker:
                    raise ValueError('Declared own resource marker differs')
                markers[identity] = marker
            elif not inherited:
                raise ValueError('Resource is not authorized redundant retention')
            if group != 'snapshots' and owner.get('zone') != row['zone'].split('/')[-1]:
                raise ValueError('Recorded disposal zone differs')
            if group == 'vms' and (row['status'] != 'TERMINATED' or row['disks'][0].get('autoDelete') is not False):
                raise ValueError('Test VM must be stopped with retained disk')
            removals.append(dict(resource_kind=group, **row))
    removable_vms = {r['name'] for r in removals if r['resource_kind'] == 'vms'}
    for row in removals:
        if row['resource_kind'] == 'disks' and any(u.rsplit('/', 1)[-1] not in removable_vms for u in row.get('users', [])):
            raise ValueError('Test disk attached outside the stopped removal set')
    return dict(status='PLANNED_NOT_DELETED', preserved_ids=protected, qualified_snapshot_id=snapshot_id,
                cpu_vm_id=proof['cpu_vm_id'], restored_disk_id=proof['restored_disk_id'],
                retire_cpu_probe=True, owned_resource_markers=markers, removals=removals,
                listing_sha256=inventory['listing_sha256'], restoration_receipt_sha256=digest(proof),
                buckets_and_file_evidence_never_deleted=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('package', 'sdk', 'proof', 'proof-job', 'final', 'inventory', 'plan'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require_limited()
    root = Path(args.package)
    with FileLock(str(root/'retention-cleanup.lock'), timeout=0):
        proof = json.loads(Path(args.proof).read_bytes())
        if not controller_job(json.loads(Path(args.proof_job).read_bytes())):
            raise ValueError('Successful Limited CPU-controller job required')
        cloud = Cloud(args.sdk, 'pure-loop-474323-a8', root/'retention-disposable-api')
        state = json.loads((root/'STATE.json').read_bytes())
        inventory = json.loads(Path(args.inventory).read_bytes()) if Path(args.inventory).exists() else census(cloud, state)
        immutable(args.inventory, inventory)
        if Path(args.plan).exists():
            plan = json.loads(Path(args.plan).read_bytes())
            if plan['listing_sha256'] != inventory['listing_sha256'] or plan['restoration_receipt_sha256'] != digest(proof):
                raise ValueError('Pinned disposal plan differs; no replay')
        else:
            plan = owned_plan(inventory, proof, state['resources'], json.loads(Path(args.final).read_bytes()))
        immutable(args.plan, plan)
        if not args.execute:
            print(json.dumps(dict(status='DRY_RUN_NO_DELETION', planned=len(plan['removals']))))
            return
        result = execute(root, cloud, inventory, plan, proof)
        target = root/('retention-disposable-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
        immutable(target, result)
        print(json.dumps(dict(status=result['status'], receipt=str(target), disposed=len(result['results']))))


if __name__ == '__main__':
    main()
