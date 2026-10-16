"""I7 owner operator. Paid actions are date/budget guarded; defaults are local."""
import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.deployment import checked_config
from scripts.study_operator.lifecycle import Operator as LegacyOperator
from scripts.study_operator.service_gateway import save_state
from scripts.study_operator.stimulus_host_evidence import sha


def reservation_charge(reservations, running, rate):
    """Union allocated and running intervals; consumed reservation is not charged twice."""
    def instant(value):
        result = datetime.fromisoformat(value.replace('Z','+00:00'))
        if result.tzinfo is None:
            raise ValueError('AWARE_COST_CLOCK_REQUIRED')
        return result.timestamp()
    intervals = sorted((instant(a),instant(b)) for a,b in [*reservations,*running])
    total, end = 0., -math.inf
    for start, stop in intervals:
        if stop < start:
            raise ValueError('COST_INTERVAL_REVERSED')
        total += max(0,stop-max(start,end))
        end = max(end,stop)
    return total/3600*rate


def reservation_identity(resource):
    properties = resource['specificReservation']['instanceProperties']
    accelerators = properties['guestAccelerators']
    if (not resource['name'].startswith('cloudrag-i7-') or not str(resource['id']).isdigit()
            or resource['zone'].split('/')[-1] not in ('us-central1-a','us-central1-b','us-central1-c')
            or str(resource['specificReservation']['count']) != '1'
            or properties['machineType'].split('/')[-1] != 'g2-standard-4'
            or len(accelerators) != 1 or accelerators[0]['acceleratorType'].split('/')[-1] != 'nvidia-l4'
            or int(accelerators[0]['acceleratorCount']) != 1):
        raise ValueError('I7_RESERVATION_IDENTITY_REQUIRED')
    return {k:resource[k] for k in ('name','id','zone','description','specificReservation')}


class Operator(LegacyOperator):
    def __init__(self, root, cloud, *, now=None, sleep=time.sleep):
        super().__init__(root,cloud,now=now,sleep=sleep)
        marker = json.loads((self.root/'I7.json').read_bytes())
        if marker != {'iteration':7} or self.config.get('iteration') != 7:
            raise ValueError('SEPARATE_I7_ROOT_REQUIRED')
        checked_config(self.config)
        if (self.config['zone'][:11] != 'us-central1' or self.config['budget_usd'] != 20
                or self.config['closure_reserve_usd'] != 3):
            raise ValueError('I7_SCOPE_OR_BUDGET_CHANGED')

    def persist(self):
        save_state(self.root/'installation.json',self.config)
        save_state(self.state_path,self.state)

    def receipt(self, kind, value):
        value = dict(value,recorded_utc=self.now().isoformat())
        folder = self.root/'receipts'/kind
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/(self.now().strftime('%Y%m%dT%H%M%S%fZ')+'-'+sha(value)+'.json')
        with path.open('x',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,sort_keys=True)
        return value

    def admit_paid(self, exposure):
        now = self.now()
        if now < datetime(2026,11,1,tzinfo=timezone.utc):
            raise ValueError('NO_PAID_EFFECT_BEFORE_NOVEMBER')
        from scripts.study_operator.i7_costs import admit
        admit(self)
        rates = self.config['rates']
        money = (exposure,self.state.get('spent_usd',0),self.state.get('open_exposure_usd',0),rates['compute_usd_h'])
        if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in money) or money[-1] == 0:
            raise ValueError('FINITE_NONNEGATIVE_LEDGER_REQUIRED')
        verified = datetime.fromisoformat(rates['verified_utc'])
        if (verified.tzinfo is None or not 0 <= (now-verified).total_seconds() <= 7*86400
                or not re.fullmatch('[a-f0-9]{64}',rates['price_receipt_sha256'])):
            raise ValueError('FRESH_USD_PRICE_RECEIPT_REQUIRED')
        if not math.isfinite(exposure) or exposure < 0:
            raise ValueError('FINITE_COST_EXPOSURE_REQUIRED')
        current = self.state.get('spent_usd',0)+self.state.get('open_exposure_usd',0)
        if current+exposure+self.config['closure_reserve_usd'] > self.config['budget_usd']+1e-9:
            raise ValueError('I7_BUDGET_STOP')

    def reserve(self, hours):
        if self.state.get('reservation_intent'):
            raise ValueError('UNKNOWN_RESERVATION_RECONCILE_FIRST')
        if not 0 < hours <= 12:
            raise ValueError('RESERVATION_TTL_REQUIRED')
        self.admit_paid(hours*self.config['rates']['compute_usd_h']+.25)
        if self.state.get('reservation') and not self.state['reservation'].get('deleted_verified'):
            raise ValueError('PRIOR_RESERVATION_NOT_CLOSED')
        allocated=self.state.get('reservation_gpu_hours_used',0)+hours
        if allocated>17:
            raise ValueError('I7_CUMULATIVE_RESERVATION_HOURS_EXCEED_COST_PLAN')
        self.state['reservation_gpu_hours_used']=allocated
        name = 'cloudrag-i7-'+self.now().strftime('%m%d%H%M%S')+'-'+self.config['period_id'][:8]
        zone = self.selected()['zone']
        until = self.now()+timedelta(hours=hours)
        intent = {'name':name,'zone':zone,'delete_at_utc':until.isoformat(),'status':'UNKNOWN'}
        self.state['reservation_intent'] = intent
        self.persist()
        # SDK beta exposes native reservation auto-delete. No unbounded fallback.
        self.cloud.command(['beta','compute','reservations','create',name,'--zone='+zone,
            '--machine-type=g2-standard-4','--vm-count=1','--accelerator=type=nvidia-l4,count=1',
            '--require-specific-reservation','--description='+self.config['resource_marker'],
            '--delete-at-time='+until.isoformat()],timeout=300)
        resource = self.cloud.command(['compute','reservations','describe',name,'--zone='+zone])
        reservation_identity(resource)
        if (resource['description'] != self.config['resource_marker']
                or datetime.fromisoformat(resource['deleteAtTime'].replace('Z','+00:00')) != until):
            raise ValueError('RESERVATION_NATIVE_DELETION_NOT_VERIFIED')
        self.state['reservation'] = dict(resource,created_utc=self.now().isoformat())
        self.state['open_exposure_usd'] = hours*self.config['rates']['compute_usd_h']+.25
        self.state.pop('reservation_intent')
        self.persist()
        return self.receipt('reservation',{'status':'SPECIFIC_RESERVATION_HELD',
            'id':str(resource['id']),'name':name,'delete_at_utc':until.isoformat()})

    def observed(self):
        from scripts.study_operator.cloud_client import checked_vm
        selected = self.selected()
        vm = self.cloud.command(['compute','instances','describe',selected['name'],'--zone='+selected['zone']])
        return checked_vm(vm,name=selected['name'],instance_id=selected['id'],zone=selected['zone'],
                          native_limit_s=self.config['native_stop_s'])

    def start(self, purpose='technical'):
        if purpose not in {'technical','smoke'}:
            raise ValueError('I7_PHASE_B_SYNTHETIC_TECHNICAL_ONLY')
        self.admit_paid(0)
        vm = self.observed()
        if vm['status'] != 'TERMINATED' or not self.state.get('reservation'):
            raise ValueError('STOPPED_VM_AND_SPECIFIC_RESERVATION_REQUIRED')
        reservation = self.state['reservation']
        remaining = (datetime.fromisoformat(reservation['deleteAtTime'].replace('Z','+00:00'))-self.now()).total_seconds()
        if remaining < self.config['native_stop_s']+60:
            raise ValueError('RESERVATION_TTL_TOO_SHORT_FOR_BOOT')
        actual = self.cloud.command(['compute','reservations','describe',reservation['name'],
                                     '--zone='+self.selected()['zone']])
        if reservation_identity(actual) != reservation_identity(reservation) or reservation.get('deleted_verified'):
            raise ValueError('SPECIFIC_RESERVATION_CHANGED')
        affinity = vm.get('reservationAffinity',{})
        if (affinity.get('consumeReservationType') != 'SPECIFIC_RESERVATION'
                or affinity.get('values') != [reservation['name']]):
            raise ValueError('SPECIFIC_RESERVATION_NOT_BOUND')
        from scripts.study_operator.startup import write_startup
        now = self.now()
        import uuid
        periods = self.config.setdefault('period_ids',{})
        periods.setdefault(self.config['purpose'],self.config['period_id'])
        periods.setdefault(purpose,uuid.uuid4().hex)
        self.config.update(purpose=purpose,period_id=periods[purpose])
        config = dict(self.config,instance_id=str(vm['id']),purpose=purpose,
            start_requested_utc=now.isoformat(),
            native_deadline_utc=(now+timedelta(seconds=self.config['native_stop_s'])).isoformat())
        # Inputs are installed and hashed on the clean disk in the CPU build.
        config.pop('bootstrap_inputs',None)
        path = self.root/'startup-i7.sh'
        write_startup(path,config)
        self.state.update(purpose=purpose,ready_verified=False,boot_started_utc=now.isoformat())
        self.persist()  # An ambiguous start remains a counted intent.
        self.cloud.command(['compute','instances','add-metadata',vm['name'],'--zone='+self.selected()['zone'],
            '--metadata-from-file=startup-script='+str(path),'--metadata=enable-guest-attributes=TRUE'])
        self.cloud.command(['compute','instances','start',vm['name'],'--zone='+self.selected()['zone']],timeout=300)
        return {'status':'STARTED_NOT_READY'}

    def preflight(self):
        # Keep existing live IAM, bucket, public 443, metadata and external TLS checks.
        self.config['official_rates'] = {'compute_usd_h':self.config['rates']['compute_usd_h']}
        result = super().preflight()
        value = self.bridge({'operation':'i7-preparation-check'},private=True)
        if value.get('status') != 'HOST_PREPARATION_VERIFIED':
            self.state['ready_verified'] = False
            self.persist()
            raise ValueError('I7_PREPARATION_NOT_VERIFIED')
        ready = self.state['ready']
        receipt = dict(ready,environment_identity_sha256=ready['environment_identity_sha256'],
                       source_commit=self.config['commit'],i7_preparation=value)
        folder = self.root/'boots'
        folder.mkdir(exist_ok=True)
        path = folder/(ready['boot_id']+'.json')
        if path.exists():
            if json.loads(path.read_bytes()) != receipt:
                raise ValueError('IMMUTABLE_BOOT_RECEIPT_CHANGED')
        else:
            with path.open('x',encoding='utf-8') as stream:
                json.dump(receipt,stream,sort_keys=True)
        return result

    def stop(self):
        vm = self.cloud.command(['compute','instances','describe',self.selected()['name'],
                                 '--zone='+self.selected()['zone']])
        if str(vm['id']) != str(self.selected()['id']):
            raise ValueError('STOP_RESOURCE_REPLACED')
        if vm['status'] != 'TERMINATED':
            self.cloud.command(['compute','instances','stop',vm['name'],'--zone='+self.selected()['zone']],timeout=600)
            vm = self.cloud.command(['compute','instances','describe',vm['name'],'--zone='+self.selected()['zone']])
        if vm['status'] != 'TERMINATED':
            raise ValueError('STOP_NOT_VERIFIED')
        self.state['ready_verified'] = False
        self.persist()
        return self.receipt('stop',{'status':'TERMINATED_VERIFIED','id':str(vm['id'])})

    def guard(self):
        """Only the owned reservation, after API-confirmed VM stop. No best-effort success."""
        reservation = self.state['reservation']
        if reservation.get('deleted_verified'):
            raise ValueError('RESERVATION_ALREADY_CLOSED')
        vm = self.cloud.command(['compute','instances','describe',self.selected()['name'],
                                 '--zone='+self.selected()['zone']])
        if str(vm['id']) != str(self.selected()['id']) or vm['status'] != 'TERMINATED':
            raise ValueError('RESERVATION_GUARD_REQUIRES_OWN_STOPPED_VM')
        zone = reservation['zone'].split('/')[-1]
        actual = self.cloud.command(['compute','reservations','describe',reservation['name'],'--zone='+zone])
        if (reservation_identity(actual) != reservation_identity(reservation)
                or actual['description'] != self.config['resource_marker']):
            raise ValueError('RESERVATION_GUARD_IDENTITY_CHANGED')
        reservation['delete_intent_sha256'] = sha(reservation_identity(actual))
        self.persist()
        self.cloud.command(['compute','reservations','delete',reservation['name'],'--zone='+zone],timeout=180)
        reservation['delete_api_success'] = True
        self.persist()
        listed = self.cloud.command(['compute','reservations','list'])
        if not isinstance(listed,list) or any(str(r['id']) == str(reservation['id']) or
            (r['name'] == reservation['name'] and r['zone'].split('/')[-1] == zone) for r in listed):
            raise ValueError('RESERVATION_ABSENCE_NOT_VERIFIED')
        reservation.update(deleted_verified=True,deleted_utc=self.now().isoformat())
        if reservation.get('created_utc'):
            interval = (self.now()-datetime.fromisoformat(reservation['created_utc'])).total_seconds()
            self.state['spent_usd'] = self.state.get('spent_usd',0)+max(0,interval)/3600*self.config['rates']['compute_usd_h']
        self.state['open_exposure_usd'] = 0
        self.persist()
        return self.receipt('reservation',{'status':'RESERVATION_DELETED_VERIFIED',
            'id':reservation['id'],'name':reservation['name'],'zone':zone})

    def reconcile(self):
        """Readback only. Unknown absence before the native deadline stays unknown."""
        listed = self.cloud.command(['compute','reservations','list'])
        if not isinstance(listed,list):
            raise TypeError('RESERVATION_LIST_NOT_SUCCESSFUL')
        reservation = self.state.get('reservation')
        if reservation and not reservation.get('deleted_verified'):
            present = [r for r in listed if str(r['id']) == str(reservation['id']) or
                       (r['name'] == reservation['name'] and r['zone'] == reservation['zone'])]
            if present:
                if len(present) != 1 or reservation_identity(present[0]) != reservation_identity(reservation):
                    raise ValueError('RECONCILIATION_IDENTITY_CHANGED')
                return {'status':'RESERVATION_STILL_PRESENT'}
            expired = reservation.get('deleteAtTime') and self.now() >= datetime.fromisoformat(
                reservation['deleteAtTime'].replace('Z','+00:00'))
            if not reservation.get('delete_api_success') and not expired:
                raise ValueError('DELETE_OUTCOME_UNKNOWN_UNTIL_NATIVE_DEADLINE')
            reservation.update(deleted_verified=True,deleted_utc=self.now().isoformat(),
                               closure_method='SUCCESSFUL_LIST_AFTER_DELETE_OR_NATIVE_TTL')
            if reservation.get('created_utc'):
                seconds=max(0,(self.now()-datetime.fromisoformat(reservation['created_utc'])).total_seconds())
                self.state['spent_usd']=self.state.get('spent_usd',0)+seconds/3600*self.config['rates']['compute_usd_h']
            self.state['open_exposure_usd']=0
            self.persist()
            return self.receipt('reservation',{'status':'RESERVATION_ABSENT_RECONCILED',
                                              'id':reservation['id']})
        intent=self.state.get('reservation_intent')
        if intent:
            found=[r for r in listed if r['name']==intent['name'] and r['zone'].split('/')[-1]==intent['zone']]
            if found:
                if len(found)!=1 or found[0]['description']!=self.config['resource_marker']:
                    raise ValueError('UNKNOWN_RESERVATION_IDENTITY_CHANGED')
                resource=found[0]
                reservation_identity(resource)
                if datetime.fromisoformat(resource['deleteAtTime'].replace('Z','+00:00')) != datetime.fromisoformat(intent['delete_at_utc']):
                    raise ValueError('UNKNOWN_RESERVATION_NATIVE_TTL_CHANGED')
                self.state['reservation']=dict(resource,created_utc=resource['creationTimestamp'])
                self.state['open_exposure_usd']=12*self.config['rates']['compute_usd_h']+.25
                self.state.pop('reservation_intent')
                self.persist()
                return {'status':'RESERVATION_FOUND_RECONCILED'}
            if self.now() < datetime.fromisoformat(intent['delete_at_utc']):
                raise ValueError('UNKNOWN_CREATE_OUTCOME_PRESERVE_INTENT')
            self.state.pop('reservation_intent')
            self.state['spent_usd']=self.state.get('spent_usd',0)+12*self.config['rates']['compute_usd_h']+.25
            self.persist()
            return self.receipt('reservation',{'status':'UNKNOWN_CREATE_EXPIRED_AND_ABSENT',
                                              'estimated_at_full_upper_bound':True})
        return {'status':'NO_OPEN_RESERVATION_RECONCILIATION'}

    def collect(self, kind):
        if kind not in {'stimulus','gate'}:
            raise ValueError('FIXED_COLLECTION_KIND_REQUIRED')
        response = self.bridge({'operation':kind+'-evidence'},private=True)
        proof = response['proof']
        if sha(proof) != response['proof_sha256']:
            raise ValueError('DOWNLOAD_HASH_DIFFERS')
        if kind == 'stimulus':
            from scripts.study_operator.stimulus_evidence import verify_boot
            config = json.loads((Path(self.config['local_reviewed_config'])/'study.json').read_bytes())
            verified = verify_boot(proof,config)
        else:
            from scripts.study_operator.i7_gate_host import verify
            verified = verify(proof)
        boot = proof['inventory']['observed']['boot_id']
        if not re.fullmatch('[a-f0-9-]{36}',boot):
            raise ValueError('BOOT_ID_REQUIRED')
        folder = self.root/'private'/kind/'coded-P999'
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/(boot+'.json')
        if path.exists():
            if sha(json.loads(path.read_bytes())) != response['proof_sha256']:
                raise ValueError('EXISTING_DOWNLOAD_CHANGED')
        else:
            with path.open('x',encoding='utf-8') as stream:
                json.dump(proof,stream,ensure_ascii=False,sort_keys=True)
        if sha(json.loads(path.read_bytes())) != response['proof_sha256']:
            raise ValueError('LOCAL_DOWNLOAD_HASH_DIFFERS')
        # ACK follows complete verification, local persistence and readback.
        ack = self.bridge({'operation':kind+'-ack','proof_sha256':response['proof_sha256']},private=True)
        if ack.get('proof_sha256') != response['proof_sha256']:
            raise ValueError('DOWNLOAD_ACK_DIFFERS')
        return self.receipt(kind,{'status':'DOWNLOAD_VERIFIED','proof_sha256':response['proof_sha256'],
            'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'verified':verified,'boot_id':boot})


def tls_three(receipts):
    rows = sorted(receipts,key=lambda r:r['start_requested_utc'])
    if len(rows) < 3:
        raise ValueError('THREE_CONSECUTIVE_BOOTS_REQUIRED')
    last = rows[-3:]
    if (len({r['boot_id'] for r in last}) != 3 or any(r['start_to_ready_s'] > 900 for r in last)
            or len({r['url'] for r in last}) != 1
            or len({r['tls']['certificate_sha256'] for r in last}) != 1
            or len({r['source_commit'] for r in last}) != 1):
        raise ValueError('TLS_BOOT_SEQUENCE_DIFFERS')
    return {'status':'TLS_THREE_BOOTS_VERIFIED','boots':[r['boot_id'] for r in last],
            'certificate_sha256':last[0]['tls']['certificate_sha256']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--sdk')
    parser.add_argument('--live',action='store_true')
    parser.add_argument('operation',choices=('init','status','reserve','reconcile','reconcile-all','technical-cycle','start','preflight','stop','guard','collect',
        'gate-start','stimulus-start','tls-check','restore-check','privacy','series','schedule','close'))
    parser.add_argument('--config',type=Path)
    parser.add_argument('--request',type=Path)
    parser.add_argument('--boot-index',type=int,choices=range(1,13))
    parser.add_argument('--kind',choices=('stimulus','gate'))
    parser.add_argument('--hours',type=float,default=12)
    args = parser.parse_args(argv)
    if args.operation == 'init':
        config = json.loads(args.config.read_bytes())
        checked_config(config)
        args.root.mkdir(parents=True,exist_ok=False)
        save_state(args.root/'installation.json',config)
        save_state(args.root/'I7.json',{'iteration':7})
        save_state(args.root/'active.json',{})
        print(json.dumps({'status':'LOCAL_I7_ROOT_CREATED'}))
        return 0
    if args.operation == 'tls-check':
        from scripts.study_operator.i7_workflows import check_tls_sequence
        result = check_tls_sequence([json.loads(p.read_bytes()) for p in
                                     (args.root/'receipts'/'boot-attempt').glob('*.json')])
    else:
        if not args.live or datetime.now(timezone.utc) < datetime(2026,11,1,tzinfo=timezone.utc):
            parser.error('Live I7 operations require --live and November authorization')
        from scripts.study_operator.cloud_client import Cloud
        if not args.sdk:
            parser.error('--sdk required')
        cloud = Cloud(args.sdk,'pure-loop-474323-a8',args.root/'runs')
        with FileLock(str(args.root/'i7.lock'),timeout=1):
            operator = Operator(args.root,cloud)
            from scripts.study_operator.i7_workflows import verify_tool_source
            verify_tool_source(operator)
            if args.operation == 'reserve':
                result = operator.reserve(args.hours)
            elif args.operation=='reconcile-all':
                from scripts.study_operator.i7_reconcile import reconcile
                result=reconcile(operator)
            elif args.operation=='technical-cycle':
                from scripts.study_operator.i7_technical import cleanup
                result=cleanup(operator)
            elif args.operation in {'start','preflight','stop','guard','status','reconcile'}:
                result = getattr(operator,args.operation)()
            elif args.operation == 'collect':
                result = operator.collect(args.kind)
            elif args.operation in {'gate-start','stimulus-start'}:
                operator.preflight()
                request = {'operation':args.operation}
                if args.operation == 'stimulus-start':
                    request['boot_index'] = args.boot_index
                result = operator.bridge(request,private=True)
            elif args.operation == 'restore-check':
                from scripts.study_operator.i7_restore import probe
                result = probe(lambda r:operator.bridge(r,private=True),json.loads(args.request.read_bytes()))
                operator.receipt('restore',result)
            else:
                from scripts.study_operator.i7_workflows import dispatch
                result = dispatch(operator,args.operation)
    print(json.dumps(result,ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
