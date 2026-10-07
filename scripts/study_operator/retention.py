"""Read-only inventory and fail-closed retention plans; never infer restorability."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.run_control import require_limited

ORIGINAL = 'cloudrag-study-l4-20261002'
LISTS = {
    'vms': ['compute', 'instances', 'list'],
    'disks': ['compute', 'disks', 'list'],
    'snapshots': ['compute', 'snapshots', 'list'],
    'ips': ['compute', 'addresses', 'list'],
    'firewalls': ['compute', 'firewall-rules', 'list'],
    'subnets': ['compute', 'networks', 'subnets', 'list'],
}
FIELDS = {
    'vms': ('name', 'id', 'zone', 'status', 'machineType', 'creationTimestamp', 'description',
            'labels', 'disks', 'deletionProtection', 'scheduling'),
    'disks': ('name', 'id', 'zone', 'sizeGb', 'type', 'sourceSnapshot', 'sourceSnapshotId',
              'sourceImage', 'sourceImageId', 'creationTimestamp', 'labels', 'users', 'selfLink'),
    'snapshots': ('name', 'id', 'status', 'diskSizeGb', 'storageBytes', 'storageLocations',
                  'sourceDisk', 'sourceDiskId', 'creationTimestamp', 'labels', 'description'),
    'ips': ('name', 'id', 'region', 'address', 'status', 'users', 'creationTimestamp'),
    'firewalls': ('name', 'id', 'network', 'disabled', 'sourceRanges', 'targetTags', 'allowed', 'direction'),
    'subnets': ('name', 'id', 'region', 'network', 'ipCidrRange', 'privateIpGoogleAccess'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def census(cloud, inherited_state):
    result = dict(at=datetime.now(timezone.utc).isoformat(), resources={}, api_response_sha256={},
                  inherited_state_sha256=digest(inherited_state), privacy='TECHNICAL_FIELDS_ONLY')
    for kind, command in LISTS.items():
        observed = cloud.command(command, private_output=True)
        if not isinstance(observed, list) or len({str(r['id']) for r in observed}) != len(observed):
            raise ValueError('Unexpected or duplicate API resource inventory')
        result['api_response_sha256'][kind] = digest(observed)
        result['resources'][kind] = [{k: r[k] for k in FIELDS[kind] if k in r} for r in observed]
    original = [r for r in result['resources']['vms'] if r['name'] == ORIGINAL]
    if len(original) != 1 or not original[0]['deletionProtection'] or original[0]['status'] != 'TERMINATED':
        raise ValueError('Original VM must be observed stopped and protected')
    boot = original[0]['disks'][0]
    disks = [r for r in result['resources']['disks'] if r['selfLink'] == boot['source']]
    if len(disks) != 1 or boot['autoDelete'] is not False:
        raise ValueError('Original retained boot disk identity missing')
    result['protected_ids'] = dict(vm=str(original[0]['id']), disk=str(disks[0]['id']))
    result['project_quotas'] = cloud.command(['compute', 'project-info', 'describe'], private_output=True)['quotas']
    result['central_quotas'] = cloud.command(['compute', 'regions', 'describe', 'us-central1'], private_output=True)['quotas']
    result['evidence_reference'] = 'Inherited STATE.json; exact listing and IDs preserved in this census'
    result['listing_sha256'] = digest(result['resources'])
    return result


def plan(inventory, qualified_snapshot_id, restoration, *, inherited_resources):
    """No names, timestamp recency, or synthetic unit result qualifies a snapshot."""
    if digest(inventory['resources']) != inventory['listing_sha256']:
        raise ValueError('Retention listing changed')
    selected = [r for r in inventory['resources']['snapshots'] if str(r['id']) == str(qualified_snapshot_id)]
    if len(selected) != 1 or selected[0]['status'] != 'READY':
        raise ValueError('One live READY snapshot required')
    if (restoration.get('status') != 'CPU_RESTORATION_VERIFIED' or restoration.get('synthetic') is not False
            or str(restoration.get('source_snapshot_id')) != str(qualified_snapshot_id)
            or not restoration.get('all_expected_files_verified') or not restoration.get('image_config_verified')
            or not restoration.get('model_manifest_and_blobs_verified')
            or not restoration.get('restored_disk_id') or not restoration.get('cpu_vm_id')
            or restoration.get('source', {}).get('files') != 19
            or restoration.get('artifacts', {}).get('files') != 79):
        raise ValueError('Real CPU restoration receipt required before deletion')
    preserved = inventory['protected_ids']
    cpu = [r for r in inventory['resources']['vms'] if str(r['id']) == restoration['cpu_vm_id']]
    clone = [r for r in inventory['resources']['disks'] if str(r['id']) == restoration['restored_disk_id']]
    if (len(cpu) != 1 or cpu[0]['status'] != 'TERMINATED' or len(clone) != 1
            or str(clone[0].get('sourceSnapshotId')) != str(qualified_snapshot_id)):
        raise ValueError('Restored CPU and disk must remain observed and stopped')
    allowed = {(r['type'], str(r['id'])): r for r in inherited_resources if r.get('inherited')}
    keep = {preserved['vm'], preserved['disk'], str(qualified_snapshot_id),
            restoration['cpu_vm_id'], restoration['restored_disk_id']}
    removals = []
    for kind in ('vms', 'disks', 'snapshots'):
        for row in inventory['resources'][kind]:
            if str(row['id']) in keep:
                continue
            # Only the expressly inherited test namespace can be considered.
            legacy = allowed.get(({'vms': 'vm', 'disks': 'disk', 'snapshots': 'snapshot'}[kind], str(row['id'])))
            if not legacy or legacy['name'] != row['name'] or not row['name'].startswith('cloudrag-i4-'):
                raise ValueError('Resource outside authorized redundant test inventory')
            if kind == 'vms' and row['status'] != 'TERMINATED':
                raise ValueError('Test VM must be stopped before disposal')
            if kind == 'disks' and any(not any(str(u).endswith('/'+v['name'])
                    for v in inventory['resources']['vms'] if str(v['id']) != preserved['vm'])
                    for u in row.get('users', [])):
                raise ValueError('Disk attached to an unaccounted instance')
            removals.append(dict(type=kind, **row))
    return dict(status='PLANNED_NOT_DELETED', preserved_ids=preserved,
                qualified_snapshot_id=str(qualified_snapshot_id), listing_sha256=inventory['listing_sha256'],
                cpu_vm_id=restoration['cpu_vm_id'], restored_disk_id=restoration['restored_disk_id'],
                restoration_receipt_sha256=digest(restoration), removals=removals,
                buckets_and_file_evidence_never_deleted=True)


def projection(inventory, snapshot_id, *, pd_usd_gib_h, snapshot_usd_gib_h, bucket_usd_day,
               initial_spend_usd, margins_usd, qualification_gpu_hours, gpu_usd_h):
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in
           (pd_usd_gib_h, snapshot_usd_gib_h, bucket_usd_day, initial_spend_usd,
            margins_usd, qualification_gpu_hours, gpu_usd_h)):
        raise ValueError('Finite nonnegative forecast assumptions required')
    disks = [r for r in inventory['resources']['disks'] if str(r['id']) == inventory['protected_ids']['disk']]
    snapshots = [r for r in inventory['resources']['snapshots'] if str(r['id']) == str(snapshot_id)]
    if len(disks) != 1 or len(snapshots) != 1:
        raise ValueError('Protected disk and selected snapshot required')
    idle = int(disks[0]['sizeGb'])*pd_usd_gib_h*24 + int(snapshots[0]['storageBytes'])/2**30*snapshot_usd_gib_h*24 + bucket_usd_day
    # 21 allocations: one pilot plus 20 sessions, 135 minutes including early start.
    sessions_hours = 21*135/60
    gpu_total = (sessions_hours+qualification_gpu_hours)*gpu_usd_h
    period_disk = 100*pd_usd_gib_h*24*24
    # The associated IP also charges while the retained VM is stopped.
    ip = .01*24*3 + .005*24*24
    scenarios = []
    for wait in (0, 30, 60, 90):
        estimate = initial_spend_usd+idle*(wait+24)+gpu_total+period_disk+ip
        scenarios.append(dict(wait_days=wait, estimated_spend_usd=estimate,
                              separate_margins_usd=margins_usd, upper_with_margins_usd=estimate+margins_usd,
                              below_own_cutoff=estimate+margins_usd < 90))
    return dict(estimated_post_cleanup_idle_usd_day=idle, cleanup_not_inferred=True,
                compressed_snapshot_bytes_may_change_after_chain_deletion=True, scenarios=scenarios,
                assumptions=dict(session_period_days=24, session_gpu_hours=sessions_hours,
                                 qualification_gpu_hours=qualification_gpu_hours, additional_disk_gib=100,
                                 associated_IP_session_days=24, no_idle_IP_during_wait=True), not_invoice=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdk', required=True)
    parser.add_argument('--package', required=True)
    parser.add_argument('--inherited-state', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    require_limited()
    cloud = Cloud(args.sdk, 'pure-loop-474323-a8', Path(args.package)/'retention-controller-census-api')
    result = census(cloud, json.loads(Path(args.inherited_state).read_bytes()))
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(status='READ_ONLY_INVENTORY', listing_sha256=result['listing_sha256'],
                          counts={k: len(v) for k, v in result['resources'].items()})))


if __name__ == '__main__':
    main()
