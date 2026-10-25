"""Preserve a verified Linux build; qualification still requires a fresh CPU restore."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from filelock import FileLock

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.cloud_safety import admission
from scripts.study_operator.cpu_build import Build
from scripts.study_operator.cpu_restoration import Controller
from scripts.study_operator.pricing import quote_archive, unit_rate
from scripts.study_operator.run_control import require_limited
from src.ui.components.session_storage import atomic_json


def inputs(root, label):
    root = Path(root)
    if not re.fullmatch('[a-z][a-z0-9]{0,15}', label):
        raise ValueError('Safe bounded build label required')
    proof = json.loads((root/('linux-build-'+label+'-download-proof.json')).read_bytes())
    state = json.loads((root/'STATE.json').read_bytes())
    job = state.get('build_jobs', {}).get(label, {})
    if (proof.get('status') != 'TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED'
            or proof.get('cpu_stopped_verified') is not True
            or job.get('status') != proof['status'] or job.get('commit') != proof.get('commit')):
        raise ValueError('Complete verified Linux PASS and CPU STOP required')
    folder = root/('linux-build-'+label+'-evidence')
    required = {'receipt.json', 'host-code-inventory.json', 'image-id.json'}
    seen = set()
    for row in proof['files']:
        name = row['object'].rsplit('/', 1)[-1]
        if not re.fullmatch(r'[a-z0-9-]+\.(?:json|stdout|stderr|receipt\.json)', name) or name in seen:
            raise ValueError('Unique technical build paths required')
        data = (folder/name).read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Downloaded build evidence changed')
        seen.add(name)
    if not required.issubset(seen):
        raise ValueError('Final image and host-code inventories missing')
    receipt = json.loads((folder/'receipt.json').read_bytes())
    host = json.loads((folder/'host-code-inventory.json').read_bytes())
    image = json.loads((folder/'image-id.json').read_bytes())
    if (receipt.get('status') != 'PASS' or str(receipt.get('vm_id')) != job.get('vm_id')
            or any(r.get('commit') != proof['commit'] or r.get('image_id') != proof['image_id']
                   for r in (receipt, host, image))):
        raise ValueError('Image, source and CPU build identities differ')
    return proof, job, host


def snapshot_quote(folder):
    verified = quote_archive(folder, 'g2-standard-4', 'us-central1')
    rows = [row for page in sorted(Path(folder).glob('page-*.json'))
            for row in json.loads(page.read_bytes())['skus']
            if row.get('description') == 'Storage PD Snapshot'
            and 'us-central1' in row.get('serviceRegions', [])
            and row.get('category', {}).get('usageType') == 'OnDemand']
    if len(rows) != 1:
        raise ValueError('One official central snapshot storage SKU required')
    rate = unit_rate(rows[0])
    if rate['usage_unit'] != 'GiBy.mo':
        raise ValueError('Monthly snapshot storage unit required')
    return dict(sku_id=rows[0]['skuId'], description=rows[0]['description'], **rate,
                estimated_month_hours=730, catalog_receipt_sha256=verified['catalog_receipt_sha256'])


def restoration_config(installation, label, proof, host):
    result = json.loads(json.dumps(installation))
    files = result['host_infrastructure_sha256']
    expected = {name:host['files'][name] for name in files}
    if not expected or any(not re.fullmatch('[a-f0-9]{64}', str(value)) for value in expected.values()):
        raise ValueError('Verified host infrastructure hashes required')
    result.update(image_id=proof['image_id'], commit=proof['commit'], host_code='/srv/cloudrag/iteration5/code-'+label,
                  host_infrastructure_sha256=expected)
    return result


def preserve(root, cloud, label, installation):
    root = Path(root)
    proof, job, host = inputs(root, label)
    config = restoration_config(installation, label, proof, host)
    build = Build(root, cloud)
    owner_input = dict(instance_id=job['vm_id'], commit=job['commit'],
                       cpu_label=job.get('cpu_label', 'bootstrap01'))
    if 'cpu_restoration_proof_sha256' in job:
        owner_input['cpu_restoration_proof_sha256'] = job['cpu_restoration_proof_sha256']
    resource = build.collection_owner(label, owner_input)
    vm = cloud.command(['compute', 'instances', 'describe', resource['name'], '--zone='+resource['zone']])
    if str(vm['id']) != resource['id'] or vm['status'] != 'TERMINATED' or vm.get('description') != resource['ownership_marker']:
        raise ValueError('Owned build CPU must be observed stopped before snapshot')
    boot = vm['disks'][0]
    disk = cloud.command(['compute', 'disks', 'describe', boot['source'].rsplit('/', 1)[-1], '--zone='+resource['zone']])
    state = json.loads((root/'STATE.json').read_bytes())
    matches = [r for r in state['resources'] if r['type'] == 'disk' and r.get('id') == str(disk['id']) and not r.get('disposed')]
    if (len(matches) != 1 or not matches[0].get('disposable') or boot.get('autoDelete') is not False
            or disk.get('selfLink') != boot['source'] or disk.get('description') != matches[0]['ownership_marker']
            or int(disk['sizeGb']) > 100):
        raise ValueError('Owned retained bounded source disk required')
    name = 'cloudrag-i5-final-'+label+'-'+root.name.rsplit('-', 1)[-1].lower()
    marker = 'CloudRAG-I5-final-'+root.name+'-'+label
    key = 'final-snapshot-'+label
    quote = snapshot_quote(root/'official-compute-skus')
    hours = max(0, (datetime.fromisoformat(state['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()/3600)
    exposure = int(disk['sizeGb'])*float(quote['usd_per_usage_unit'])/730*hours
    ctl = Controller(root, cloud)
    def intend(s):
        prior = s.setdefault('final_snapshot_intents', {}).get(label)
        intent = dict(name=name, source_disk_id=str(disk['id']), source_vm_id=resource['id'],
                      image_id=proof['image_id'], commit=proof['commit'], ownership_marker=marker)
        if prior and prior != intent:
            raise ValueError('Final snapshot intent changed; no adoption')
        opened = s.setdefault('open_exposures', {})
        if key not in opened:
            admission(s, sum(r['maximum_usd'] for r in opened.values())+exposure)
            opened[key] = dict(maximum_usd=exposure, not_billed_spend=True,
                              catalog_receipt_sha256=quote['catalog_receipt_sha256'])
        s['final_snapshot_intents'][label] = intent
    ctl.update(intend)
    with (root/'COST_LEDGER.md').open('a', encoding='utf-8') as f:
        f.write('\nBEFORE final snapshot '+label+': '+json.dumps(quote)+'; full disk byte upper USD '+str(exposure)+
                ' through deadline, 730h/month conversion ESTIMATED; not invoice. Retain candidate until independent CPU restoration.\n')
    found = cloud.command(['compute', 'snapshots', 'list', '--filter=name='+name])
    if not found:
        cloud.command(['compute', 'snapshots', 'create', name, '--source-disk='+disk['name'],
            '--source-disk-zone='+resource['zone'], '--storage-location=us-central1', '--description='+marker], timeout=600)
    elif len(found) != 1 or found[0].get('description') != marker:
        raise ValueError('Existing snapshot outside recorded ownership')
    snapshot = cloud.command(['compute', 'snapshots', 'describe', name])
    if (snapshot.get('description') != marker or snapshot.get('status') != 'READY'
            or str(snapshot.get('sourceDiskId')) != str(disk['id'])
            or snapshot.get('storageLocations') != ['us-central1']):
        raise ValueError('Snapshot source, ownership or READY differs')
    record = dict(type='snapshot', name=name, id=str(snapshot['id']), zone='', disposable=False,
                  ownership_marker=marker, candidate_not_qualified=True, source_disk_id=str(disk['id']))
    def observed(s):
        previous = [r for r in s['resources'] if r['type'] == 'snapshot' and r['name'] == name]
        if previous and previous != [record]:
            raise ValueError('Recorded final snapshot identity changed')
        if not previous:
            s['resources'].append(record)
    ctl.update(observed)
    result = dict(status='FINAL_IMAGE_SNAPSHOT_READY_CPU_RESTORATION_PENDING', snapshot=snapshot,
                  image_id=proof['image_id'], commit=proof['commit'], source_cpu_vm_id=resource['id'],
                  build_proof_sha256=hashlib.sha256((root/('linux-build-'+label+'-download-proof.json')).read_bytes()).hexdigest(),
                  live_gpu_acceptance_not_inferred=True)
    target = root/('final-snapshot-'+label+'-receipt.json')
    if target.exists():
        if json.loads(target.read_bytes()) != result:
            raise ValueError('Existing snapshot receipt differs')
    else:
        atomic_json(target, result)
    config_path = root/('cpu-'+label+'-restoration-installation.json')
    if config_path.exists() and json.loads(config_path.read_bytes()) != config:
        raise ValueError('Restoration-only input changed')
    if not config_path.exists():
        atomic_json(config_path, config)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('package', 'sdk', 'label', 'installation'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    require_limited()
    root = Path(args.package)
    with FileLock(str(root/'cpu-build.lock'), timeout=0):
        result = preserve(root, Cloud(args.sdk, 'pure-loop-474323-a8', root/'final-snapshot-api'), args.label,
                          json.loads(Path(args.installation).read_bytes()))
        print(json.dumps(dict(status=result['status'], snapshot_id=str(result['snapshot']['id']))))


if __name__ == '__main__':
    main()
