"""Bounded, resumable CPU restoration with intents before every paid effect."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from filelock import FileLock

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.cloud_safety import admission
from scripts.study_operator.pricing import quote_archive
from scripts.study_operator.policy import ReadyPending
from scripts.study_operator.run_control import require_limited
from scripts.study_operator.startup import host_key_publication
from src.ui.components.session_storage import atomic_json


def verified_closure(package):
    package = Path(package)
    receipt = json.loads((package/'iteration4-recovered-seal02.json').read_bytes())
    proof = json.loads(Path(receipt['verification_receipt']).read_bytes())
    pin = json.loads((package/'legacy-seal-retry-pin.json').read_bytes())
    if (receipt['status'] != 'SEALED_AND_EXTERNALLY_VERIFIED' or receipt['verifier_exit_code'] != 0
            or proof['status'] != 'PASS' or proof.get('errors') or proof.get('missing') or proof.get('added') or proof.get('links')
            or proof['checked_files'] != pin['files'] or proof['expected_files'] != pin['files']
            or proof['manifest_sha256'] != pin['manifest_sha256'] or proof['audited_directory_modified'] is not False):
        raise ValueError('Inherited closure not externally verified; no paid effect')
    manifest = Path(receipt['root'])/'MANIFEST_SHA256.jsonl'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != pin['manifest_sha256']:
        raise ValueError('Inherited inventory changed; no paid effect')
    return receipt


def startup_text():
    # Override old startup metadata. No app, Ollama, TLS, or terminal study job.
    return ("#!/bin/bash\nset -euo pipefail\npython3 - <<'PY'\n"
        "from pathlib import Path\nimport subprocess\n"+host_key_publication()+
        "r=subprocess.run(['docker','ps','-q'],capture_output=True,check=True)\n"
        "ids=r.stdout.decode().split()\n"
        "if ids:subprocess.run(['docker','stop',*ids],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=120,check=True)\n"
        "subprocess.Popen(['/usr/sbin/shutdown','-h','+110'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\nPY\n")


def create_arguments(name, disk, startup, marker, installation):
    return ['compute', 'instances', 'create', name, '--zone=us-central1-a', '--machine-type=e2-standard-2',
        '--provisioning-model=STANDARD', '--disk=name='+disk+',boot=yes,auto-delete=no',
        '--deletion-protection', '--max-run-duration=2h', '--instance-termination-action=STOP',
        '--no-address', '--no-service-account', '--no-scopes',
        '--network='+installation['network'], '--subnet='+installation['subnet'],
        '--metadata-from-file=startup-script='+str(startup), '--metadata=enable-guest-attributes=TRUE',
        '--tags='+name+'-iap', '--description='+marker]


class Controller:
    def __init__(self, package, cloud, *, label='bootstrap01'):
        if not re.fullmatch('[a-z][a-z0-9-]{0,15}', label):
            raise ValueError('Safe bounded restoration label required')
        self.root, self.cloud = Path(package), cloud
        self.label = label

    def update(self, mutation):
        with FileLock(str(self.root/'state.lock'), timeout=10):
            path = self.root/'STATE.json'
            state = json.loads(path.read_bytes())
            mutation(state)
            state['updated_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(path, state)
        return state

    def intent(self, kind, name, marker, *, source_snapshot_id=None):
        def mutate(state):
            intents = state.setdefault('resource_intents', [])
            row = dict(type=kind, name=name, zone='us-central1-a', ownership_marker=marker,
                       source_snapshot_id=source_snapshot_id, disposable=True)
            prior = next((r for r in intents if r['type'] == kind and r['name'] == name), None)
            if prior and prior != row:
                raise ValueError('Creation intent differs; no replay')
            if not prior:
                intents.append(row)
        self.update(mutate)

    def observed_resource(self, kind, value, marker):
        if value.get('description') != marker:
            raise ValueError('Resource ownership marker differs')
        row = dict(type=kind, name=value['name'], id=str(value['id']), zone='us-central1-a',
                   disposable=True, created_utc=value['creationTimestamp'], ownership_marker=marker)
        def mutate(state):
            resources = state.setdefault('resources', [])
            prior = next((r for r in resources if r['type'] == kind and r['name'] == row['name']), None)
            if prior and prior['id'] != row['id']:
                raise ValueError('Resource ID changed; no replay')
            if not prior:
                resources.append(row)
        self.update(mutate)
        return row

    def prepare(self, inventory, installation, *, snapshot_name='cloudrag-i4-user-recovery-20261006'):
        verified_closure(self.root)
        if (self.root/f'cpu-restoration-{self.label}-proof.json').exists():
            raise ValueError('Proof already exists; reconcile STOP instead of replaying')
        candidates = [r for r in inventory['resources']['snapshots'] if r['name'] == snapshot_name]
        if len(candidates) != 1:
            raise ValueError('Explicit recovery candidate missing')
        snapshot = candidates[0]
        live = self.cloud.command(['compute', 'snapshots', 'describe', snapshot['name']], private_output=True)
        if str(live['id']) != str(snapshot['id']) or live['status'] != 'READY':
            raise ValueError('Recovery snapshot identity differs')
        quote = quote_archive(self.root/'official-compute-skus', 'e2-standard-2', 'us-central1')
        # Keep the test PD reserved through the global closure, not only CPU uptime.
        exposure = 2*float(quote['usd_per_hour']) + 100*.000136986*72
        def reserve(state):
            reservations = state.setdefault('open_exposures', {})
            key = 'cpu-restoration-'+self.label
            if key in reservations and reservations[key]['maximum_usd'] != exposure:
                raise ValueError('Outstanding paid reservation differs')
            total = sum(row['maximum_usd'] for row in reservations.values())
            admission(state, total+(0 if key in reservations else exposure))
            reservations.setdefault(key, dict(maximum_usd=exposure, not_billed_spend=True,
                catalog_receipt_sha256=quote['catalog_receipt_sha256'], at=datetime.now(timezone.utc).isoformat()))
        self.update(reserve)
        marker = 'CloudRAG-I5-restore-'+self.root.name+'-'+self.label
        name = 'cloudrag-i5-restore-'+self.label
        disk = name+'-boot'
        rule_name = name+'-iap'
        declaration = self.root/'DESTRUCTION_LOG.md'
        with declaration.open('a', encoding='utf-8') as stream:
            stream.write(f'\nBEFORE: own disposable CPU VM {name}, cloned PD {disk}, IAP rule {rule_name}, '
                'retained Docker proof containers cloudrag-i5-restore-<CPU_ID>-missing/-user and their 256MiB tmpfs. '
                'Original VM/disk, all snapshot candidates and buckets preserved until explicit qualified retention plan.\n')
        with (self.root/'COST_LEDGER.md').open('a', encoding='utf-8') as stream:
            stream.write(f'\nBEFORE CPU restoration: official catalog {quote["catalog_receipt_sha256"]}; '
                f'e2-standard-2 USD{quote["usd_per_hour"]}/h, native2h; 100GiB balanced PD '
                f'USD0.3287664/day through closure (72h upper), exposureUSD{exposure}; operation margin separately. '
                'No GPU, external IP, new SA, or new credential. Not billed spend.\n')
        self.intent('disk', disk, marker, source_snapshot_id=str(snapshot['id']))
        disks = self.cloud.command(['compute', 'disks', 'list', '--filter=name='+disk], private_output=True)
        if not disks:
            self.cloud.command(['compute', 'disks', 'create', disk, '--zone=us-central1-a', '--size=100GB',
                '--type=pd-balanced', '--source-snapshot='+snapshot['name'], '--description='+marker], private_output=True, timeout=600)
        observed_disk = self.cloud.command(['compute', 'disks', 'describe', disk, '--zone=us-central1-a'], private_output=True)
        if str(observed_disk.get('sourceSnapshotId')) != str(snapshot['id']):
            raise ValueError('Restored disk not derived from qualified candidate')
        self.observed_resource('disk', observed_disk, marker)
        rules = self.cloud.command(['compute', 'firewall-rules', 'list', '--filter=name='+rule_name], private_output=True)
        if not rules:
            self.intent('firewall', rule_name, marker)
            self.cloud.command(['compute', 'firewall-rules', 'create', rule_name,
                '--network='+installation['network'], '--direction=INGRESS', '--allow=tcp:22',
                '--source-ranges=35.235.240.0/20', '--target-tags='+rule_name, '--description='+marker],
                private_output=True)
        observed_rule = self.cloud.command(['compute', 'firewall-rules', 'describe', rule_name], private_output=True)
        if (observed_rule.get('description') != marker or observed_rule.get('sourceRanges') != ['35.235.240.0/20']
                or observed_rule.get('targetTags') != [rule_name]
                or observed_rule.get('allowed') != [{'IPProtocol': 'tcp', 'ports': ['22']}]):
            raise ValueError('Temporary IAP rule scope differs')
        self.observed_resource('firewall', observed_rule, marker)
        startup = self.root/('cpu-restore-'+self.label+'-startup.sh')
        if not startup.exists():
            startup.open('x', encoding='utf-8', newline='\n').write(startup_text())
        elif startup.read_text() != startup_text():
            raise ValueError('Startup source changed since creation')
        self.intent('vm', name, marker)
        vms = self.cloud.command(['compute', 'instances', 'list', '--filter=name='+name], private_output=True)
        if not vms:
            self.cloud.command(create_arguments(name, disk, startup, marker, installation), private_output=True, timeout=600)
        vm = self.cloud.command(['compute', 'instances', 'describe', name, '--zone=us-central1-a'], private_output=True)
        if (vm['machineType'].split('/')[-1] != 'e2-standard-2' or vm.get('guestAccelerators')
                or not vm.get('deletionProtection') or vm['disks'][0]['autoDelete'] is not False
                or vm['disks'][0]['source'] != observed_disk['selfLink'] or vm.get('serviceAccounts')
                or any(i.get('accessConfigs') for i in vm['networkInterfaces'])
                or vm['scheduling'].get('instanceTerminationAction') != 'STOP'
                or vm['scheduling'].get('maxRunDuration', {}).get('seconds') != '7200'):
            raise ValueError('CPU VM protections, scope, disk or native STOP differ')
        self.observed_resource('vm', vm, marker)
        return dict(vm=vm, disk=observed_disk, snapshot=snapshot)

    def stop_owned_after_prepare_failure(self):
        """A lost create response still has its pre-existing ownership intent."""
        state = json.loads((self.root/'STATE.json').read_bytes())
        intents = [r for r in state.get('resource_intents', [])
                   if r['type'] == 'vm' and r['name'] == 'cloudrag-i5-restore-'+self.label]
        if not intents:
            return dict(status='NO_VM_CREATION_INTENT')
        vms = self.cloud.command(['compute', 'instances', 'list', '--filter=name='+intents[0]['name']], private_output=True)
        for vm in vms:
            if vm.get('description') != intents[0]['ownership_marker'] or not vm['zone'].endswith('/us-central1-a'):
                raise ValueError('Foreign VM after failed prepare; never stop it')
            if vm['status'] != 'TERMINATED':
                self.cloud.command(['compute', 'instances', 'stop', vm['name'], '--zone=us-central1-a'], private_output=True, timeout=600)
        return dict(status='OWN_PREPARE_FAILURE_STOP_REQUESTED', observed_ids=[str(r['id']) for r in vms],
                    native_STOP_independent=True)

    def probe_and_stop(self, resources, installation):
        vm = resources['vm']
        baseline = json.loads((self.root/'rag_freeze_baseline.json').read_bytes())
        artifacts = json.loads(Path('C:/CloudRAG/autonomous-run-20261002T203006Z/deployment-artifacts.json').read_bytes())
        spec = dict(cpu_vm_id=str(vm['id']), restored_disk_id=str(resources['disk']['id']),
            source_snapshot_id=str(resources['snapshot']['id']), zone='us-central1-a',
            code_root=installation['host_code'], asset_root=installation['asset_root'],
            source_files=baseline['rag']['modules'], artifact_files=artifacts['files'],
            image_id=installation['image_id'], ollama_models=installation['ollama_models'],
            model_digest=installation['model_digest'])
        source = Path(__file__).with_name('restore_probe.py').read_text(encoding='utf-8')
        payload = (source+'\nimport base64\nspec=json.loads(base64.b64decode('+repr(base64.b64encode(json.dumps(spec).encode()).decode())+'))\n'
                   'print(json.dumps(run(spec)))\n').encode()
        result = None
        try:
            deadline = time.monotonic()+600
            while True:
                try:
                    keys = self.cloud.command(['compute', 'instances', 'get-guest-attributes', vm['name'],
                        '--zone=us-central1-a', '--query-path=hostkeys/'], private_output=True)
                except ReadyPending:
                    keys = []
                if keys:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Guest public host keys did not arrive')
                time.sleep(15)
            raw = self.cloud.command(['compute', 'ssh', vm['name'], '--zone=us-central1-a',
                '--tunnel-through-iap', '--ssh-key-expire-after=10m', '--command=sudo -n python3 -B -'],
                input_data=payload, private_output=True, json_output=False, timeout=900)
            result = json.loads(raw)
            if result.get('status') != 'CPU_RESTORATION_VERIFIED' or result['cpu_vm_id'] != str(vm['id']):
                raise ValueError('Guest proof contract differs')
            atomic_json(self.root/f'cpu-restoration-{self.label}-proof.json', result)
        finally:
            self.cloud.command(['compute', 'instances', 'stop', vm['name'], '--zone=us-central1-a'], private_output=True, timeout=600)
            stopped = self.cloud.command(['compute', 'instances', 'describe', vm['name'], '--zone=us-central1-a'], private_output=True)
            if str(stopped['id']) != str(vm['id']) or stopped['status'] != 'TERMINATED':
                raise ValueError('CPU STOP not verified; native limit remains armed')
            atomic_json(self.root/f'cpu-restoration-{self.label}-stop.json', dict(id=str(vm['id']), status='TERMINATED',
                at=datetime.now(timezone.utc).isoformat(), source_snapshot_preserved=True))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--sdk', required=True)
    parser.add_argument('--installation', required=True)
    parser.add_argument('--inventory')
    parser.add_argument('--snapshot', default='cloudrag-i4-user-recovery-20261006')
    parser.add_argument('--label', default='bootstrap01')
    args = parser.parse_args(argv)
    require_limited()
    root = Path(args.package)
    controller = Controller(root, Cloud(args.sdk, 'pure-loop-474323-a8', root/('cpu-restoration-'+args.label+'-api')), label=args.label)
    installation = json.loads(Path(args.installation).read_bytes())
    try:
        resources = controller.prepare(json.loads(Path(args.inventory or root/'RETENTION_INVENTORY.json').read_bytes()), installation,
                                       snapshot_name=args.snapshot)
    except Exception:
        stopped = controller.stop_owned_after_prepare_failure()
        atomic_json(root/('cpu-prepare-failure-'+args.label+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'), stopped)
        raise
    result = controller.probe_and_stop(resources, installation)
    print(json.dumps(dict(status=result['status'], runtime_user_pair=result['runtime_user_pair']['status'])))


if __name__ == '__main__':
    main()
