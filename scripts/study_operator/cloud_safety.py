"""Independent own-resource stop and finite paid-effect admission for I5."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.run_control import require_limited
from src.ui.components.session_storage import atomic_json


def admission(state, exposure_usd, *, now=None):
    now = now or datetime.now(timezone.utc)
    limit = datetime.fromisoformat(state['closure_reserved_utc'])
    cost = state['cost']['estimated_spend_usd']
    reserve = state['cost']['reserved_retention_and_closure_usd']
    values = (cost, reserve, exposure_usd)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Finite nonnegative cost and closure reserve required')
    if state['status'] != 'ACTIVE' or now.tzinfo is None or limit.tzinfo is None or now >= limit:
        raise ValueError('Paid work closed; preserve evidence and complete closure')
    if sum(values) >= state['cloud_cutoff_usd']:
        raise ValueError('Exposure reaches own cloud cutoff; no paid operation admitted')
    if not state.get('independent_closure_verified'):
        raise ValueError('Verified independent closure required before paid effect')
    return dict(projected_with_exposure_usd=sum(values), cutoff_usd=state['cloud_cutoff_usd'])


def targets(state, observed):
    known = {str(row['id']): row for row in state.get('resources', []) if row['type'] == 'vm'}
    intents = {row['name']: row for row in state.get('resource_intents', []) if row['type'] == 'vm'}
    selected = []
    for vm in observed:
        recorded = known.get(str(vm['id']))
        intent = intents.get(vm['name'])
        zone = vm['zone'].split('/')[-1]
        if recorded:
            if recorded['name'] != vm['name'] or recorded['zone'] != zone:
                raise ValueError('Recorded resource identity changed')
        elif intent:
            if (not vm['name'].startswith('cloudrag-i5-') or vm.get('description') != intent['ownership_marker']
                    or intent['zone'] != zone or not intent['ownership_marker'].startswith('CloudRAG-I5-')):
                raise ValueError('Unreconciled creation intent does not prove ownership')
        else:
            continue
        if not zone.startswith('us-'):
            raise ValueError('Owned target outside US scope')
        selected.append(vm)
    return selected


def close(root, cloud):
    root = Path(root)
    state_path = root/'STATE.json'
    with FileLock(str(root/'state.lock'), timeout=10):
        state = json.loads(state_path.read_bytes())
        state.update(status='CLOSING', next_action='Independent closure running; do not launch paid work')
        atomic_json(state_path, state)
    # Full metadata/SSH comments remain in memory, never in technical evidence.
    before = cloud.command(['compute', 'instances', 'list'], private_output=True)
    selected = targets(state, before)
    if not selected:
        raise ValueError('No owned VM observed; empty census cannot prove safe closure')
    recorded_ids = {str(row['id']) for row in state.get('resources', []) if row['type'] == 'vm'}
    if not recorded_ids.issubset({str(row['id']) for row in selected}):
        raise ValueError('Recorded VM missing from closure census')
    for vm in selected:
        if vm['status'] not in {'TERMINATED', 'STOPPING'}:
            cloud.command(['compute', 'instances', 'stop', vm['name'], '--zone='+vm['zone'].split('/')[-1]], timeout=600)
    after = cloud.command(['compute', 'instances', 'list'], private_output=True)
    selected_after = targets(state, after)
    original_ids = {str(row['id']) for row in selected}
    if {str(row['id']) for row in selected_after} != original_ids:
        raise ValueError('Closure census changed; native STOP remains necessary')
    all_stopped = all(row['status'] == 'TERMINATED' for row in selected_after)
    def projection(rows):
        return [{key: row[key] for key in ('name', 'id', 'zone', 'status')} for row in rows]
    receipt = dict(status='OWN_VMS_TERMINATED_VERIFIED' if all_stopped else 'STOP_PENDING_NOT_SAFE',
                   at=datetime.now(timezone.utc).isoformat(), before=projection(selected), after=projection(selected_after),
                   disks_buckets_snapshots_not_deleted=True, further_network_and_retention_audit_required=True)
    target = root/('safety-close-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    atomic_json(target, receipt)
    with FileLock(str(root/'state.lock'), timeout=10):
        current = json.loads(state_path.read_bytes())
        current.update(status='CLOSING', phase=6, safety_receipt=str(target),
                       next_action='Verify IP release, temporary rules, resource retention and final auditor package; paid work remains closed')
        atomic_json(state_path, current)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--sdk', required=True)
    args = parser.parse_args(argv)
    require_limited()
    root = Path(args.package)
    cloud = Cloud(args.sdk, 'pure-loop-474323-a8', root/'independent-safety-api')
    result = close(root, cloud)
    print(json.dumps(dict(status=result['status'])))
    return int(result['status'] != 'OWN_VMS_TERMINATED_VERIFIED')


if __name__ == '__main__':
    raise SystemExit(main())
