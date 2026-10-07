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
            if recorded.get('disposed'):
                raise ValueError('Disposed VM reappeared; closure identity is unsafe')
            if recorded['name'] != vm['name'] or recorded['zone'] != zone:
                raise ValueError('Recorded resource identity changed')
        elif any(r['name'] == vm['name'] for r in known.values()):
            raise ValueError('Recorded VM name reappeared with another ID')
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


def close_network(root, cloud, state, stopped):
    """Release only ID-bound own temporary IPs/IAP rules after every VM stops."""
    if any(vm['status'] != 'TERMINATED' for vm in stopped):
        raise ValueError('Network cleanup requires all owned VMs terminated')
    # Reconcile a successful creation whose response was lost. Intent markers
    # were persisted before the effect; a name alone never proves ownership.
    for intent in state.get('resource_intents', []):
        kind = intent['type']
        if kind not in {'address', 'firewall'} or intent.get('disposed'):
            continue
        if any(row['type'] == kind and row['name'] == intent['name'] for row in state['resources']):
            continue
        if (not intent['name'].startswith('cloudrag-i5-')
                or not intent['ownership_marker'].startswith('CloudRAG-I5-')
                or kind == 'firewall' and not intent['name'].endswith('-iap')):
            raise ValueError('Unreconciled network intent outside own scope')
        group = 'addresses' if kind == 'address' else 'firewall-rules'
        rows = cloud.command(['compute', group, 'list', '--filter=name='+intent['name']], private_output=True)
        if len(rows) > 1 or rows and (rows[0].get('description') != intent['ownership_marker']
                or not str(rows[0].get('id')).isdigit()
                or kind == 'address' and rows[0].get('region', '').split('/')[-1] != intent['region']):
            raise ValueError('Lost network creation response lacks verified own identity')
        if rows:
            reconciled = dict(intent, id=str(rows[0]['id']))
            state['resources'].append(reconciled)
            with FileLock(str(Path(root)/'state.lock'), timeout=10):
                path = Path(root)/'STATE.json'
                current = json.loads(path.read_bytes())
                if any(row['type'] == kind and row['name'] == intent['name'] for row in current['resources']):
                    raise ValueError('Network reconciliation state changed')
                current['resources'].append(reconciled)
                atomic_json(path, current)
    results = []
    for row in state.get('resources', []):
        kind = row['type']
        if kind not in {'address','firewall'} or row.get('inherited') or not row.get('disposable'):
            continue
        marker = row.get('ownership_marker','')
        if (not row['name'].startswith('cloudrag-i5-') or not marker.startswith('CloudRAG-I5-')
                or kind == 'firewall' and not row['name'].endswith('-iap')):
            raise ValueError('Network resource lacks authorized own scope')
        group = 'addresses' if kind == 'address' else 'firewall-rules'
        def observed():
            matches = cloud.command(['compute',group,'list','--filter=name='+row['name']], private_output=True)
            if len(matches) > 1 or matches and (str(matches[0]['id']) != str(row['id'])
                    or matches[0].get('description') != marker):
                raise ValueError('Own network resource identity changed; no deletion')
            return matches[0] if matches else None
        live = observed()
        if live and row.get('disposed'):
            raise ValueError('Disposed own network resource reappeared')
        if live:
            if kind == 'address':
                region = row.get('region','')
                if not region.startswith('us-') or live.get('region','').split('/')[-1] != region:
                    raise ValueError('Own IP region changed; no detachment or deletion')
                for vm in stopped:
                    for interface in vm.get('networkInterfaces',[]):
                        for access in interface.get('accessConfigs',[]):
                            if access.get('natIP') == live['address']:
                                cloud.command(['compute','instances','delete-access-config',vm['name'],
                                    '--zone='+vm['zone'].split('/')[-1], '--network-interface='+interface['name'],
                                    '--access-config-name='+access['name']], private_output=True)
            with (Path(root)/'DESTRUCTION_LOG.md').open('a',encoding='utf-8') as stream:
                stream.write('\nBEFORE independent closure: own disposable '+kind+' '+row['name']+
                             ' ID='+str(row['id'])+'; no inherited rule or IP deleted.\n')
            cloud.command(['compute',group,'delete',row['name'],
                           *(['--region='+row['region']] if kind == 'address' else [])],
                          private_output=True,timeout=180)
        if observed() is not None:
            raise ValueError('Network resource absence not verified')
        result = dict(type=kind,name=row['name'],id=str(row['id']),status='OWN_NETWORK_ABSENCE_VERIFIED',
                      at=datetime.now(timezone.utc).isoformat())
        target = Path(root)/('safety-network-'+kind+'-'+str(row['id'])+'.json')
        if not target.exists():
            atomic_json(target,result)
        with FileLock(str(Path(root)/'state.lock'),timeout=10):
            path = Path(root)/'STATE.json'
            current = json.loads(path.read_bytes())
            matches = [item for item in current['resources'] if item['type'] == kind and str(item['id']) == str(row['id'])]
            if len(matches) != 1:
                raise ValueError('Network owner state changed')
            matches[0].update(disposed=True,absence_verified=True,disposal_receipt=str(target))
            for intent in current.get('resource_intents', []):
                if intent['type'] == kind and intent['name'] == row['name']:
                    intent.update(disposed=True, absence_verified=True)
            atomic_json(path,current)
        results.append(json.loads(target.read_bytes()))
    return results


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
    recorded_ids = {str(row['id']) for row in state.get('resources', [])
                    if row['type'] == 'vm' and not row.get('disposed')}
    if any(row.get('disposed') and not row.get('absence_verified') for row in state.get('resources', [])):
        raise ValueError('Disposed resource lacks API absence evidence')
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
    network = close_network(root,cloud,state,selected_after) if all_stopped else []
    def projection(rows):
        return [{key: row[key] for key in ('name', 'id', 'zone', 'status')} for row in rows]
    receipt = dict(status='OWN_VMS_TERMINATED_VERIFIED' if all_stopped else 'STOP_PENDING_NOT_SAFE',
                   at=datetime.now(timezone.utc).isoformat(), before=projection(selected), after=projection(selected_after),
                   disks_buckets_snapshots_not_deleted=True, own_network_receipts=network,
                   further_network_and_retention_audit_required=True)
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
