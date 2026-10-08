"""Reconcile recorded resource intervals; estimates and margins are never invoices."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.evidence import verify
from scripts.study_operator.pricing import disk_quote_archive, quote_archive, unit_rate
from scripts.study_operator.retention import digest, projection
from scripts.study_operator.retention_disposable import controller_job
from scripts.study_operator.run_control import require_limited, utc
from src.ui.components.session_storage import atomic_json


def aware(value):
    value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        raise ValueError('Recorded aware interval required')
    return value.astimezone(timezone.utc)


def nonnegative(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a cost or unit')
    value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise ValueError('Finite nonnegative amount required')
    return value


def interval_cost(rows, begin, end):
    """Charge each identity once, clipping at prior reconciliation and retirement."""
    begin, end = aware(begin), aware(end)
    if end < begin:
        raise ValueError('Reconciliation cannot go backwards')
    seen, total, intervals = set(), Decimal(0), []
    for row in rows:
        identity = row['kind'], str(row['id'])
        if identity in seen:
            raise ValueError('Duplicate resource would double count')
        seen.add(identity)
        created = aware(row['created_utc'])
        retired = aware(row['retired_utc']) if row.get('retired_utc') else end
        if retired < created or retired > end:
            raise ValueError('Retirement must be observed within reconciliation')
        start, stop = max(begin, created), retired
        seconds = max(Decimal(0), Decimal(str((stop-start).total_seconds())))
        value = seconds/Decimal(3600)*nonnegative(row['units'])*nonnegative(row['usd_unit_h'])
        total += value
        intervals.append(dict(kind=identity[0], id=identity[1], started_utc=start.isoformat(),
            ended_utc=stop.isoformat(), interval_s=str(seconds), estimated_usd=str(value),
            units=str(row['units']), usd_unit_h=str(row['usd_unit_h']), not_invoice=True))
    return dict(estimated_increment_usd=str(total), intervals=intervals)


def static_ip_upper(sku):
    """Explicit maximum-tier upper bound; never treat a free tier as paid usage."""
    pricing = max(sku['pricingInfo'], key=lambda row: row['effectiveTime'])
    expression = pricing['pricingExpression']
    if expression['usageUnit'] != 'h':
        raise ValueError('Static IP hourly unit required')
    rates, previous = [], Decimal(-1)
    for tier in expression['tieredRates']:
        start = nonnegative(tier['startUsageAmount'])
        if start <= previous:
            raise ValueError('Strictly increasing IP tiers required')
        previous = start
        money = tier['unitPrice']
        if money['currencyCode'] != 'USD':
            raise ValueError('Explicit USD IP price required')
        rates.append(nonnegative(money.get('units', 0))+nonnegative(money.get('nanos', 0))/10**9)
    if not rates:
        raise ValueError('IP price missing')
    return dict(usd_h_upper=str(max(rates)), sku_id=sku['skuId'],
        method='MAXIMUM_TIER_UPPER_FREE_TIER_NOT_DEDUCTED', not_invoice=True)


def catalog_rows(folder):
    # Existing complete-catalog verifier pins every page and pagination first.
    quote_archive(folder, 'e2-standard-2', 'us-central1')
    receipt = json.loads((folder/'receipt.json').read_bytes())
    return [row for page in receipt['pages']
            for row in json.loads((folder/page['path']).read_bytes())['skus']]


def reconcile(root, inputs, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.parent != root or output.exists():
        raise ValueError('New own-package receipt required; no replay')
    verify(root)
    pins = {}

    def read(key):
        path = (root/inputs[key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Own-package evidence only')
        raw = path.read_bytes()
        pins[key] = dict(path=path.relative_to(root).as_posix(), sha256=hashlib.sha256(raw).hexdigest())
        return json.loads(raw)

    baseline, live, extra = read('baseline'), read('inventory'), read('extra_inventory')
    previous = read('previous_inventory')
    if any(digest(v['resources']) != v['listing_sha256'] for v in (live, previous, extra)):
        raise ValueError('API listing identity changed')
    if (len(live['resources']['vms']) != 1 or live['resources']['vms'][0]['status'] != 'TERMINATED'
            or not live['resources']['vms'][0]['deletionProtection'] or live['resources']['ips']
            or len(live['resources']['disks']) != 1 or len(live['resources']['snapshots']) != 1
            or live['protected_ids'] != previous['protected_ids']):
        raise ValueError('Fresh stopped original, single final snapshot and no IP required')
    final_snapshot = live['resources']['snapshots'][0]
    if final_snapshot['status'] != 'READY':
        raise ValueError('READY qualified final snapshot required')
    transition = None
    if str(final_snapshot['id']) != str(previous['resources']['snapshots'][0]['id']):
        proof, final, job = read('qualification_proof'), read('qualified_final'), read('qualification_job')
        if (not controller_job(job) or proof.get('status') != 'CPU_RESTORATION_VERIFIED'
                or proof.get('synthetic') is not False
                or not all(proof.get(key) for key in ('all_expected_files_verified',
                    'image_config_verified', 'model_manifest_and_blobs_verified'))
                or proof.get('source', {}).get('files') != 19
                or proof.get('artifacts', {}).get('files') != 79
                or proof.get('runtime_user_pair', {}).get('status') != 'PAIRED_RUNTIME_USER_SUPPORTED'
                or proof.get('image_id') != final.get('image_id')
                or str(proof.get('source_snapshot_id')) != str(final_snapshot['id'])
                or str(final.get('snapshot', {}).get('id')) != str(final_snapshot['id'])):
            raise ValueError('Actual new final-image restoration and Limited controller required')
        transition = dict(previous_id=str(previous['resources']['snapshots'][0]['id']),
            qualified_id=str(final_snapshot['id']), image_id=proof['image_id'],
            restoration_pinned=True, gpu_acceptance_not_inferred=True)
    begin, end = baseline['cost']['as_of_utc'], live['at']
    prices = catalog_rows(root/'official-compute-skus')
    snapshot = [r for r in prices if r['description'] == 'Storage PD Snapshot'
                and 'us-central1' in r['serviceRegions']]
    ip = [r for r in prices if r['description'] == 'Static Ip Charge' and 'us-west1' in r['serviceRegions']]
    transfer = [r for r in prices if r['description'] == 'PD snapshot Data Transfer Out within North America'
                and 'global' in r['serviceRegions']]
    if len(snapshot) != 1 or len(ip) != 1 or len(transfer) != 1:
        raise ValueError('Unambiguous applicable official storage and IP SKUs required')
    rate = unit_rate(snapshot[0])
    if rate['usage_unit'] != 'GiBy.mo':
        raise ValueError('Snapshot monthly unit required')
    snapshot_h = nonnegative(rate['usd_per_usage_unit'])/730
    ip_rate = static_ip_upper(ip[0])
    transfer_rate = unit_rate(transfer[0])
    if transfer_rate['usage_unit'] != 'GiBy':
        raise ValueError('Official North America transfer unit required')
    survivors = {(k,str(r['id'])) for k in ('disks','snapshots') for r in live['resources'][k]}
    rows = []
    for kind in ('disks','snapshots'):
        combined = {str(r['id']):r for r in previous['resources'][kind]}
        for row in extra['resources'][kind]:
            identity = str(row['id'])
            if identity in combined and any(combined[identity].get(k) != row.get(k)
                    for k in ('creationTimestamp','sizeGb','sourceSnapshotId','sourceDiskId','storageLocations')):
                raise ValueError('One identity has conflicting API metadata')
            combined[identity] = row
        for identity, row in combined.items():
            retired = None
            if (kind,identity) in survivors:
                current_row = next(r for r in live['resources'][kind] if str(r['id']) == identity)
                if kind == 'snapshots':
                    # Chain cleanup may change stored bytes. Keep the largest
                    # observed size as an interval upper, not a billing claim.
                    row = dict(row,storageBytes=max(int(row['storageBytes']),int(current_row['storageBytes'])))
                elif (row['sizeGb'],row['type']) != (current_row['sizeGb'],current_row['type']):
                    raise ValueError('Original disk changed since prior reconciliation')
            else:
                deletion = read('deleted_'+identity)
                if deletion['resource_id'] != identity or deletion['status'] != 'RESOURCE_ABSENCE_VERIFIED':
                    raise ValueError('Individual API absence receipt required')
                retired = deletion['at']
            if kind == 'disks':
                if row['type'].split('/')[-1] != 'pd-balanced':
                    raise ValueError('Unquoted disk type')
                quote = disk_quote_archive(root/'official-compute-skus',row['zone'].split('/')[-1].rsplit('-',1)[0])
                units, hourly = int(row['sizeGb']), quote['estimated_usd_gib_h']
            else:
                if row['storageLocations'] != ['us-central1']:
                    raise ValueError('Unquoted snapshot region')
                units, hourly = Decimal(row['storageBytes'])/2**30, snapshot_h
            rows.append(dict(kind=kind,id=identity,created_utc=row['creationTimestamp'],
                retired_utc=retired,units=str(units),usd_unit_h=str(hourly)))
    # A terminated CPU clone's creation-to-deletion interval is a conservative
    # upper bound, including its stopped periods. It never implies GPU usage.
    cpu_upper = []
    for vm in extra['resources']['vms']:
        identity = str(vm['id'])
        if identity == str(live['protected_ids']['vm']):
            continue
        if (vm.get('status') != 'TERMINATED' or vm.get('guestAccelerators')
                or vm['machineType'].split('/')[-1] != 'e2-standard-2'
                or not vm['name'].startswith('cloudrag-i5-restore-')
                or not vm.get('description', '').startswith('CloudRAG-I5-restore-')):
            raise ValueError('Only recorded stopped owned CPU clones can be estimated')
        deletion = read('deleted_'+identity)
        if deletion['resource_id'] != identity or deletion['status'] != 'RESOURCE_ABSENCE_VERIFIED':
            raise ValueError('CPU API absence receipt required')
        quote = quote_archive(root/'official-compute-skus', 'e2-standard-2',
            vm['zone'].split('/')[-1].rsplit('-', 1)[0])
        row = dict(kind='cpu_lifetime_upper', id=identity, created_utc=vm['creationTimestamp'],
            retired_utc=deletion['at'], units=1, usd_unit_h=str(quote['usd_per_hour']))
        rows.append(row)
        cpu_upper.append(dict(instance_id=identity, method='CREATION_TO_API_ABSENCE_INCLUDES_STOPPED_TIME',
            official_quote=quote, not_invoice=True))
    ip_start, ip_end = read('ip_start'), read('ip_end')
    if (ip_start['exit_code'] != 0 or ip_end['exit_code'] != 0
            or 'ip-reserve' not in ip_start['command'] or 'ip-release' not in ip_end['command']):
        raise ValueError('Successful original IP reserve/release command bounds required')
    rows.append(dict(kind='static_ip_command_upper',id=inputs['ip_id'],created_utc=ip_start['started_utc'],
        retired_utc=ip_end['ended_utc'],units=1,usd_unit_h=ip_rate['usd_h_upper']))
    charges = interval_cost(rows,begin,end)
    bucket_day = Decimal('.01')  # Preserved conservative estimate; never an API usage measurement.
    bucket_increment = Decimal(str((aware(end)-aware(begin)).total_seconds()))/86400*bucket_day
    idle = bucket_day+sum(nonnegative(r['units'])*nonnegative(r['usd_unit_h'])*24 for r in rows
                          if (r['kind'],r['id']) in survivors)
    prior = baseline['cost']
    previous_disk_ids = {str(r['id']) for r in previous['resources']['disks']}
    added_cross_region = [r for r in extra['resources']['disks'] if str(r['id']) not in previous_disk_ids
                          and not r['zone'].split('/')[-1].startswith('us-central1-')]
    transfer_units = sum(int(r['sizeGb']) for r in added_cross_region)
    transfer_margin = (nonnegative(prior['unreconciled_snapshot_transfer_upper_separate_usd'])
                       +transfer_units*nonnegative(transfer_rate['usd_per_usage_unit']))
    margin = nonnegative(prior['initial_margin_separate_usd'])+nonnegative(prior['image_egress_upper_separate_usd'])+transfer_margin
    spend = nonnegative(prior['estimated_spend_usd'])+nonnegative(charges['estimated_increment_usd'])+bucket_increment
    state = json.loads((root/'STATE.json').read_bytes())
    reserve = max(Decimal(0),Decimal(str((aware(state['deadline_utc'])-aware(end)).total_seconds())))/86400*idle
    cost = dict(estimated_spend_usd=float(spend),as_of_utc=end,current_idle_upper_usd_day=float(idle),
        permanent_idle_after_own_test_cleanup_usd_day=float(idle),reserved_retention_and_closure_usd=float(margin+reserve),
        initial_margin_separate_usd=prior['initial_margin_separate_usd'],
        image_egress_upper_separate_usd=prior['image_egress_upper_separate_usd'],
        unreconciled_snapshot_transfer_upper_separate_usd=float(transfer_margin),not_invoice=True)
    forecast = projection(live,live['resources']['snapshots'][0]['id'],
        pd_usd_gib_h=float(disk_quote_archive(root/'official-compute-skus','us-central1')['estimated_usd_gib_h']),
        snapshot_usd_gib_h=float(snapshot_h),bucket_usd_day=float(bucket_day),initial_spend_usd=float(spend),
        margins_usd=float(margin+5),qualification_gpu_hours=12,
        gpu_usd_h=float(quote_archive(root/'official-compute-skus','g2-standard-4','us-central1')['usd_per_hour']))
    result = dict(status='ESTIMATED_RECONCILIATION_NOT_INVOICE',cost=cost,inputs=pins,
        recorded_intervals=charges,bucket_increment_estimated_usd=str(bucket_increment),
        static_ip_quote=ip_rate,snapshot_quote=rate,estimate_month_hours=730,
        qualified_snapshot_transition=transition,cpu_lifetime_upper_bounds=cpu_upper,
        current_idle_target_met=idle<=Decimal('.45'),forecast_unchanged_assumptions=forecast,
        transfer_margin_basis=dict(added_disk_ids=[str(r['id']) for r in added_cross_region],
            gib_upper=transfer_units,quote=transfer_rate,actual_transferred_bytes_not_measured=True),
        gpu_runtime_not_inferred=True,remaining_operator_reservations_not_spend=True)
    with FileLock(str(root/'state.lock'),timeout=10):
        current = json.loads((root/'STATE.json').read_bytes())
        if current['cost']['as_of_utc'] != begin:
            raise ValueError('Ledger changed; do not double count')
        atomic_json(output,result)
        current.update(cost=cost,cost_reconciliation_receipt=str(output),updated_utc=utc())
        atomic_json(root/'STATE.json',current)
    with (root/'COST_LEDGER.md').open('a',encoding='utf-8') as stream:
        stream.write('\nESTIMADO, NO FACTURA: '+json.dumps(cost)+'; recibo '+output.name+
            '; margenes separados, reservas del operador no facturadas; proyeccion conserva supuestos.\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args(argv)
    require_limited()
    result = reconcile(args.root,json.loads(Path(args.inputs).read_bytes()),args.output)
    print(json.dumps(dict(status=result['status'],cost=result['cost'],idle_target_met=result['current_idle_target_met'])))


if __name__ == '__main__':
    main()
