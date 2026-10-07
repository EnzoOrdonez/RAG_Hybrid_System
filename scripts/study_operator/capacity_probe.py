"""Bounded central L4 rounds. A successful start is capacity evidence, not READY."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.capacity_order import next_round
from scripts.study_operator.cloud_client import Cloud, checked_vm, no_other_gpu
from scripts.study_operator.cloud_safety import admission
from scripts.study_operator.cpu_restoration import startup_text
from scripts.study_operator.policy import OperatorError
from scripts.study_operator.pricing import quote_archive
from scripts.study_operator.run_control import require_limited
from src.ui.components.session_storage import atomic_json


ZONES = ('us-central1-a','us-central1-b','us-central1-c')


def create_args(name,disk,zone,startup,marker,network,subnet):
    if zone not in ZONES:
        raise ValueError('Central round only; regional relocation needs its own prepared subnet')
    return ['compute','instances','create',name,'--zone='+zone,'--machine-type=g2-standard-4',
        '--provisioning-model=STANDARD','--maintenance-policy=TERMINATE',
        '--disk=name='+disk+',boot=yes,auto-delete=no','--deletion-protection',
        '--max-run-duration=3h','--instance-termination-action=STOP','--no-address',
        '--no-service-account','--no-scopes','--network='+network,'--subnet='+subnet,
        '--metadata-from-file=startup-script='+str(startup),'--metadata=enable-guest-attributes=TRUE',
        '--description='+marker]


class Probe:
    def __init__(self,root,cloud,*,now=lambda:datetime.now(timezone.utc)):
        self.root,self.cloud,self.now = Path(root),cloud,now

    def update(self,change):
        with FileLock(str(self.root/'state.lock'),timeout=10):
            path = self.root/'STATE.json'
            value = json.loads(path.read_bytes())
            change(value)
            value['updated_utc'] = self.now().isoformat()
            atomic_json(path,value)
        return value

    def record(self,kind,row,zone,marker):
        if row.get('description') != marker or row['zone'].split('/')[-1] != zone:
            raise ValueError('Own capacity resource marker or zone differs')
        resource = dict(type=kind,name=row['name'],id=str(row['id']),zone=zone,disposable=True,
                        ownership_marker=marker,created_utc=row['creationTimestamp'])
        def change(state):
            prior = [r for r in state['resources'] if r['type'] == kind and r['name'] == row['name']]
            if prior and (len(prior) != 1 or prior[0]['id'] != resource['id'] or prior[0].get('disposed')):
                raise ValueError('Recorded capacity resource changed; no adoption')
            if not prior:
                state['resources'].append(resource)
        self.update(change)

    def intent(self,kind,name,zone,marker,**extra):
        row = dict(type=kind,name=name,zone=zone,ownership_marker=marker,disposable=True,**extra)
        def change(state):
            prior = [r for r in state.setdefault('resource_intents',[]) if r['type'] == kind and r['name'] == name]
            if prior and prior != [row]:
                raise ValueError('Capacity creation intent differs')
            if not prior:
                state['resource_intents'].append(row)
        self.update(change)

    def original(self,state):
        original = next(r for r in state['resources'] if r['type'] == 'vm' and r.get('inherited')
                        and r['name'] == 'cloudrag-study-l4-20261002')
        vm = self.cloud.command(['compute','instances','describe',original['name'],'--zone='+original['zone']])
        checked_vm(vm,name=original['name'],instance_id=original['id'],zone=original['zone'])
        if vm['status'] != 'TERMINATED' or any(n.get('accessConfigs') for n in vm['networkInterfaces']):
            raise ValueError('Original must be stopped without an external IP before capacity-only probing')
        return vm

    def attempt(self,zone,number,snapshot,network,subnet,quote):
        state = json.loads((self.root/'STATE.json').read_bytes())
        exposure = 3*float(quote['usd_per_hour'])
        if zone != ZONES[0]:
            exposure += 100*.000136986*max(0,(datetime.fromisoformat(state['closure_reserved_utc'])-self.now()).total_seconds())/3600
        key = 'capacity-round-'+str(number)+'-'+zone
        def reserve(value):
            opened = value.setdefault('open_exposures',{})
            admission(value,sum(v['maximum_usd'] for v in opened.values())+exposure,now=self.now())
            if key in opened:
                raise ValueError('Attempt already reserved; reconcile its evidence instead of replaying')
            opened[key] = dict(maximum_usd=exposure,not_billed_spend=True,at=self.now().isoformat(),
                              catalog_receipt_sha256=quote['catalog_receipt_sha256'])
        self.update(reserve)
        startup = self.root/('capacity-'+str(number)+'-'+zone+'-startup.sh')
        with startup.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(startup_text())  # stops Docker; guest shutdown110min, independent of conversation
        marker = 'CloudRAG-I5-capacity-'+self.root.name+'-'+zone
        name = 'cloudrag-i5-capacity-'+zone.removeprefix('us-')
        disk_name = name+'-boot'
        with (self.root/'DESTRUCTION_LOG.md').open('a',encoding='utf-8') as stream:
            stream.write('\nBEFORE capacity '+key+': own test VM '+name+' and disk '+disk_name+
                ' disposable when qualified final snapshot exists. Original VM/disk retained; '
                'no public IP, no participant app, no terminal study jobs. Startup override removed after STOP.\n')
        with (self.root/'COST_LEDGER.md').open('a',encoding='utf-8') as stream:
            stream.write('\nBEFORE '+key+': '+json.dumps(quote)+'; reserved upper USD '+str(exposure)+
                '; disk100GiB pd-balanced .000136986/GiBh through closure if needed. Reservation is not spend.\n')
        no_other_gpu(self.cloud.command(['compute','instances','list']),selected_id='NO_SELECTED_RUNNING_VM')
        if zone == ZONES[0]:
            vm = self.original(state)
            self.cloud.command(['compute','instances','add-metadata',vm['name'],'--zone='+zone,
                                '--metadata-from-file=startup-script='+str(startup)])
        else:
            self.intent('disk',disk_name,zone,marker,source_snapshot_id=str(snapshot['id']))
            disks = self.cloud.command(['compute','disks','list','--filter=name='+disk_name])
            if not disks:
                self.cloud.command(['compute','disks','create',disk_name,'--zone='+zone,'--type=pd-balanced',
                                    '--source-snapshot='+snapshot['selfLink'],'--description='+marker],timeout=600)
                disks = self.cloud.command(['compute','disks','list','--filter=name='+disk_name])
            if len(disks) != 1 or str(disks[0].get('sourceSnapshotId')) != str(snapshot['id']) or disks[0].get('users'):
                # Attached is allowed only after checking the already-owned stopped VM below.
                if len(disks) != 1 or str(disks[0].get('sourceSnapshotId')) != str(snapshot['id']):
                    raise ValueError('Capacity disk source differs')
            self.record('disk',disks[0],zone,marker)
            self.intent('vm',name,zone,marker,source_snapshot_id=None)
            existing = self.cloud.command(['compute','instances','list','--filter=name='+name])
            if not existing and disks[0].get('users'):
                raise ValueError('Capacity disk attached to an unknown instance')
            vm = existing[0] if len(existing) == 1 else None
            if len(existing) > 1:
                raise ValueError('Duplicate capacity VM name')
            if vm:
                self.record('vm',vm,zone,marker)
                checked_vm(vm,name=name,instance_id=vm['id'],zone=zone)
                if vm['status'] != 'TERMINATED' or vm['disks'][0]['source'] != disks[0]['selfLink']:
                    raise ValueError('Capacity VM must be stopped on the owned disk')
                self.cloud.command(['compute','instances','add-metadata',name,'--zone='+zone,
                                    '--metadata-from-file=startup-script='+str(startup)])
        began = self.now()
        operation_error = None
        result = dict(status='TOOL_OR_RESOURCE_ERROR_NOT_CAPACITY',zone=zone)
        try:
            if vm:
                self.cloud.command(['compute','instances','start',vm['name'],'--zone='+zone],timeout=600)
            else:
                self.cloud.command(create_args(name,disk_name,zone,startup,marker,network,subnet),timeout=600)
            live = self.cloud.command(['compute','instances','list','--filter=name='+(vm['name'] if vm else name)])
            if len(live) != 1 or live[0]['status'] != 'RUNNING':
                raise ValueError('Capacity start was not observed RUNNING')
            vm = live[0]
            if zone != ZONES[0]:
                self.record('vm',vm,zone,marker)
            checked_vm(vm,name=vm['name'],instance_id=vm['id'],zone=zone)
            gpu = vm.get('guestAccelerators',[])
            if len(gpu) != 1 or not gpu[0]['acceleratorType'].endswith('/nvidia-l4') or gpu[0]['acceleratorCount'] != 1:
                raise ValueError('Capacity result is not one L4')
            result = dict(status='L4_RUNNING_OBSERVED_NOT_READY',name=vm['name'],id=str(vm['id']),zone=zone)
        except OperatorError as error:
            if 'ZONE_RESOURCE_POOL_EXHAUSTED' not in str(error):
                operation_error = error
                result = dict(status='TOOL_OR_RESOURCE_ERROR_NOT_CAPACITY',zone=zone)
            else:
                result = dict(status='CAPACITY_EXHAUSTED',zone=zone,cloud_error_code='ZONE_RESOURCE_POOL_EXHAUSTED')
        finally:
            live = self.cloud.command(['compute','instances','list','--filter=name='+(vm['name'] if vm else name)])
            if live:
                if len(live) != 1 or vm and str(live[0]['id']) != str(vm['id']):
                    raise ValueError('Post-attempt VM identity differs; native STOP remains necessary')
                if zone != ZONES[0]:
                    self.record('vm',live[0],zone,marker)
                if live[0]['status'] != 'TERMINATED':
                    self.cloud.command(['compute','instances','stop',live[0]['name'],'--zone='+zone],timeout=600)
                stopped = self.cloud.command(['compute','instances','describe',live[0]['name'],'--zone='+zone])
                if stopped['status'] != 'TERMINATED' or str(stopped['id']) != str(live[0]['id']):
                    raise ValueError('Capacity STOP not verified')
                self.cloud.command(['compute','instances','remove-metadata',live[0]['name'],'--zone='+zone,
                                    '--keys=startup-script'])
            end = self.now()
            receipt = dict(**result,started_utc=began.isoformat(),ended_utc=end.isoformat(),
                           stopped_verified=True,not_a_measurement_or_ready=True,
                           compute_interval_upper_s=(end-began).total_seconds() if live else 0)
            atomic_json(self.root/(key+'-receipt.json'),receipt)
        if operation_error:
            raise operation_error
        return receipt

    def central(self,snapshot,network,subnet):
        state = json.loads((self.root/'STATE.json').read_bytes())
        rounds = state.get('capacity_rounds',[])
        number = next_round(rounds,self.now())
        if any(r.get('status') == 'RUNNING' for r in rounds):
            raise ValueError('Prior incomplete round must be reconciled before a new round')
        live = self.cloud.command(['compute','snapshots','describe',snapshot['name']])
        if live['id'] != snapshot['id'] or live['status'] != 'READY':
            raise ValueError('Qualified snapshot identity changed')
        proof = json.loads((self.root/'cpu-restoration-bootstrap01-proof.json').read_bytes())
        if proof['source_snapshot_id'] != str(live['id']) or proof['status'] != 'CPU_RESTORATION_VERIFIED' or proof['synthetic']:
            raise ValueError('Actual restored snapshot required')
        quote = quote_archive(self.root/'official-compute-skus','g2-standard-4','us-central1')
        entry = dict(round=number,at=self.now().isoformat(),status='RUNNING',results=[])
        self.update(lambda s:s.setdefault('capacity_rounds',[]).append(entry))
        try:
            for zone in ZONES:
                receipt = self.attempt(zone,number,live,network,subnet,quote)
                entry['results'].append(receipt)
                self.update(lambda s:s['capacity_rounds'].__setitem__(number-1,dict(entry)))
                if receipt['status'] == 'L4_RUNNING_OBSERVED_NOT_READY':
                    break
            entry['status'] = 'COMPLETED'
        except Exception:
            entry['status'] = 'FAILED_PRESERVED_RECONCILIATION_REQUIRED'
            raise
        finally:
            self.update(lambda s:s['capacity_rounds'].__setitem__(number-1,dict(entry)))
        return entry


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('package','sdk','snapshot','network','subnet'):
        parser.add_argument('--'+name,required=True)
    args = parser.parse_args(argv)
    require_limited()
    root = Path(args.package)
    with FileLock(str(root/'capacity-probe.lock'),timeout=0):
        snapshot = json.loads(Path(args.snapshot).read_bytes())
        probe = Probe(root,Cloud(args.sdk,'pure-loop-474323-a8',root/'capacity-probe-api'))
        print(json.dumps(probe.central(snapshot,args.network,args.subnet)))


if __name__ == '__main__':
    main()
