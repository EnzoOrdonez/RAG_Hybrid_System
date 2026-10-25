"""Inventory and retire only an unattached failed bootstrap disk, preserving cause."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.cloud_client import Cloud, no_other_gpu
from scripts.study_operator.evidence import verify as verify_claims
from scripts.study_operator.lifecycle import Operator
from scripts.study_operator.retention import census, digest
from scripts.study_operator.retention_cleanup import execute, immutable
from scripts.study_operator.retention_disposable import controller_job
from scripts.study_operator.run_control import require_limited


def failed_disk_plan(inventory, state, intent, proof, job):
    rows = inventory['resources']
    if digest(rows) != inventory['listing_sha256'] or not controller_job(job):
        raise ValueError('Pinned actual inventory and successful Limited CPU job required')
    snapshot_id = str(proof['source_snapshot_id'])
    if (proof.get('status') != 'CPU_RESTORATION_VERIFIED' or proof.get('synthetic') is not False
            or proof.get('source', {}).get('files') != 19 or proof.get('artifacts', {}).get('files') != 79
            or not all(proof.get(k) for k in ('all_expected_files_verified', 'image_config_verified', 'model_manifest_and_blobs_verified'))
            or proof.get('runtime_user_pair', {}).get('status') != 'PAIRED_RUNTIME_USER_SUPPORTED'
            or not proof.get('cpu_vm_id') or not proof.get('restored_disk_id')
            or intent['source_snapshot_id'] != snapshot_id):
        raise ValueError('Verified final-image restoration required before disposal')
    snapshots = [r for r in rows['snapshots'] if str(r['id']) == snapshot_id]
    if len(snapshots) != 1 or snapshots[0]['status'] != 'READY':
        raise ValueError('Qualified snapshot must remain READY')
    no_other_gpu(rows['vms'], selected_id='none')
    if any(vm['name'] == intent['name'] for vm in rows['vms']):
        raise ValueError('Creation exists; reconcile and stop instead of disposing its disk')
    disks = [r for r in rows['disks'] if r['name'] == intent['disk_name']]
    if len(disks) != 1:
        raise ValueError('One live failed-creation disk required')
    disk = disks[0]
    owners = [r for r in state['resources'] if r['type'] == 'disk' and str(r['id']) == str(disk['id']) and not r.get('disposed')]
    if (len(owners) != 1 or owners[0].get('disposable') is not True
            or not disk['name'].startswith('cloudrag-i5-primary-') or disk.get('users')
            or disk['zone'].split('/')[-1] != intent['zone'] or owners[0]['zone'] != intent['zone']
            or str(disk.get('sourceSnapshotId')) != snapshot_id
            or disk.get('description') != intent['ownership_marker']
            or owners[0]['ownership_marker'] != intent['ownership_marker']
            or not intent['ownership_marker'].startswith('CloudRAG-I5-')):
        raise ValueError('Only the ID-bound own unattached synthetic bootstrap disk is disposable')
    protected = inventory['protected_ids']
    if str(disk['id']) in {*protected.values(), snapshot_id}:
        raise ValueError('Protected resource cannot be disposed')
    return dict(status='PLANNED_NOT_DELETED', preserved_ids=protected, qualified_snapshot_id=snapshot_id,
        cpu_vm_id=proof['cpu_vm_id'], restored_disk_id=proof['restored_disk_id'],
        listing_sha256=inventory['listing_sha256'], restoration_receipt_sha256=digest(proof),
        owned_resource_markers={str(disk['id']): intent['ownership_marker']},
        removals=[dict(disk, resource_kind='disks')], failed_cause_not_reclassified=True,
        buckets_and_file_evidence_never_deleted=True)


def pinned_proof(root, proof_name, job_name):
    verify_claims(root)
    claims = json.loads((root/'claims.json').read_bytes())
    for name in (proof_name, job_name):
        if not any(row['certainty'] == 'VERIFICADO' and any(e['path'] == name for e in row['evidence']) for row in claims):
            raise ValueError('Restoration proof and job must be hash-bound verified claims')
    return json.loads((root/proof_name).read_bytes()), json.loads((root/job_name).read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('package', 'sdk', 'owner', 'proof', 'proof-job', 'inventory', 'plan'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    require_limited()
    root, owner = Path(args.package).resolve(), Path(args.owner).resolve()
    inventory_path, plan_path = Path(args.inventory).resolve(), Path(args.plan).resolve()
    if inventory_path.parent != root or plan_path.parent != root or owner.name != 'operator-iteration5':
        raise ValueError('Own package and operator5 scope required')
    with FileLock(str(root/'retention-cleanup.lock'), timeout=0), FileLock(str(owner/'operator.lock'), timeout=0):
        proof, job = pinned_proof(root, args.proof, args.proof_job)
        cloud = Cloud(args.sdk, 'pure-loop-474323-a8', root/'failed-bootstrap-retention-api')
        state = json.loads((root/'STATE.json').read_bytes())
        operator = Operator(owner, cloud)
        active = operator.state
        if active.get('primary_bootstrap_complete') or active.get('ready_verified'):
            raise ValueError('This disposal is for a failed pre-primary attempt only')
        inventory = json.loads(inventory_path.read_bytes()) if inventory_path.exists() else census(cloud, state)
        immutable(inventory_path, inventory)
        plan = failed_disk_plan(inventory, state, active['primary_creation_intent'], proof, job)
        immutable(plan_path, plan)
        if not args.execute:
            print(json.dumps(dict(status='DRY_RUN_NO_DELETION', disk_id=plan['removals'][0]['id'])))
            return
        result = execute(root, cloud, inventory, plan, proof)
        immutable(plan_path.with_name(plan_path.stem+'-receipt.json'), result)
        disk_id = str(plan['removals'][0]['id'])
        owned = [r for r in active['audit_resources'] if r['type'] == 'disk' and str(r['id']) == disk_id]
        if len(owned) != 1:
            raise ValueError('Disposal completed but operator reconciliation missing; preserve receipt')
        owned[0].update(disposed=True, absence_verified=True,
                        absence_verified_utc=datetime.now(timezone.utc).isoformat())
        active['failed_primary_disk_disposal_receipt'] = str(plan_path.with_name(plan_path.stem+'-receipt.json'))
        # Preserve the unknown create intent for reconciliation and safety.
        operator.persist()
        print(json.dumps(dict(status=result['status'], removed=len(result['results']), cause_not_reclassified=True)))


if __name__ == '__main__':
    main()
