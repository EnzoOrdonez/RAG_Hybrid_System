"""Launch a retained CPU build without holding the conversation open."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from filelock import FileLock

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.cloud_safety import admission
from scripts.study_operator.cpu_restoration import Controller,startup_text
from scripts.study_operator.gcs import Storage
from scripts.study_operator.policy import ReadyPending,validate_storage_iam
from scripts.study_operator.pricing import quote_archive
from scripts.study_operator.run_control import require_limited
from src.ui.components.session_storage import atomic_json

SA = 'cloudrag-study-i4@pure-loop-474323-a8.iam.gserviceaccount.com'
TECHNICAL = 'cloudrag-study-103950017681-20261002'
SESSIONS = 'cloudrag-study-i4-103950017681-20261004'


def checked_cpu(vm,resource):
    if (str(vm['id']) != resource['id'] or vm['name'] != resource['name']
            or vm['zone'].split('/')[-1] != resource['zone'] or vm['status'] != 'TERMINATED'
            or not vm.get('deletionProtection') or vm['disks'][0]['autoDelete'] is not False
            or vm['machineType'].split('/')[-1] != 'e2-standard-2' or vm.get('guestAccelerators')
            or vm['scheduling'].get('instanceTerminationAction') != 'STOP'
            or vm['scheduling'].get('maxRunDuration',{}).get('seconds') != '7200'
            or any(n.get('accessConfigs') for n in vm['networkInterfaces'])
            or vm.get('description') != resource['ownership_marker']):
        raise ValueError('Stopped owned CPU identity, retained disk or native bound differs')
    return vm


def stage_payload(job,archive,source):
    """Private stdin only. No bearer token or session content goes into this job."""
    data = base64.b64encode(archive).decode()
    code = base64.b64encode(source).decode()
    encoded_job = base64.b64encode(json.dumps(job).encode()).decode()
    return ("import base64,json,os,hashlib,tarfile,subprocess,urllib.request\nfrom pathlib import Path\n"
        "job=json.loads(base64.b64decode("+repr(encoded_job)+"))\n"
        "r=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'})\n"
        "with urllib.request.urlopen(r,timeout=5) as s:assert s.read().decode()==job['instance_id']\n"
        "root=Path(job['context']).parent;root.mkdir(parents=True,exist_ok=False)\n"
        "data=base64.b64decode("+repr(data)+")\nassert hashlib.sha256(data).hexdigest()==job['context_sha256']\n"
        "archive=root/'context.tar.gz';archive.write_bytes(data);del data\n"
        "context=Path(job['context']);context.mkdir()\n"
        "with tarfile.open(archive) as s:s.extractall(context,filter='data')\n"
        "source=base64.b64decode("+repr(code)+");assert hashlib.sha256(source).hexdigest()==job['guest_source_sha256']\n"
        "script=root/'linux_build.py';script.write_bytes(source)\n"
        "settings=root/'job.json';settings.write_text(json.dumps(job));os.chmod(settings,0o600)\n"
        "child=subprocess.Popen(['python3','-B',str(script),'--job',str(settings)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\n"
        "print(json.dumps(dict(status='CPU_BUILD_LAUNCHED_NOT_ACCEPTED',pid=child.pid,instance_id=job['instance_id'],context_sha256=job['context_sha256'])))\n").encode()


class Build:
    def __init__(self,root,cloud,*,cpu_label='bootstrap01'):
        if not re.fullmatch('[a-z][a-z0-9-]{0,15}',cpu_label):
            raise ValueError('Safe explicit restored CPU label required')
        self.root,self.cloud = Path(root),cloud
        self.state = Controller(root,cloud)
        self.cpu_label = cpu_label

    def resource(self,*,cpu_label=None):
        label = cpu_label or self.cpu_label
        if not re.fullmatch('[a-z][a-z0-9-]{0,15}',label):
            raise ValueError('Safe explicit restored CPU label required')
        state = json.loads((self.root/'STATE.json').read_bytes())
        matches = [r for r in state['resources'] if r['type'] == 'vm' and r.get('disposable')
                   and r['name'] == 'cloudrag-i5-restore-'+label and not r.get('disposed')]
        if len(matches) != 1:
            raise ValueError('One verified own CPU restoration clone required')
        proof = json.loads((self.root/('cpu-restoration-'+label+'-proof.json')).read_bytes())
        disks = [r for r in state['resources'] if r['type'] == 'disk' and r.get('disposable')
                 and r['name'] == 'cloudrag-i5-restore-'+label+'-boot' and not r.get('disposed')]
        if (proof.get('status') != 'CPU_RESTORATION_VERIFIED' or proof.get('synthetic') is not False
                or str(proof.get('cpu_vm_id')) != matches[0]['id'] or matches[0]['zone'] != 'us-central1-a'
                or proof.get('zone') != 'us-central1-a' or len(disks) != 1
                or disks[0]['ownership_marker'] != matches[0]['ownership_marker']
                or str(proof.get('restored_disk_id')) != disks[0]['id']
                or not proof.get('source_snapshot_id') or proof.get('all_expected_files_verified') is not True
                or proof.get('image_config_verified') is not True or proof.get('model_manifest_and_blobs_verified') is not True):
            raise ValueError('Own CPU clone is not the independently verified restoration')
        return matches[0]

    def stopped(self,resource):
        vm = self.cloud.command(['compute','instances','describe',resource['name'],'--zone='+resource['zone']])
        if str(vm['id']) != resource['id'] or vm['name'] != resource['name']:
            raise ValueError('CPU identity changed; refusing to stop a replacement')
        if vm['status'] != 'TERMINATED':
            self.cloud.command(['compute','instances','stop',resource['name'],'--zone='+resource['zone']],timeout=600)
            vm = self.cloud.command(['compute','instances','describe',resource['name'],'--zone='+resource['zone']])
        return checked_cpu(vm,resource)

    def launch(self,label,inventory,source):
        if not re.fullmatch('[a-z][a-z0-9]{0,15}',label):
            raise ValueError('Safe unique build label required')
        resource = self.resource()
        proof_sha = hashlib.sha256((self.root/('cpu-restoration-'+self.cpu_label+'-proof.json')).read_bytes()).hexdigest()
        bundle = Path(inventory['bundle'])
        archive = bundle.read_bytes()
        if (inventory['status'] != 'CLEAN_PUBLISHED_BUILD_CONTEXT_VERIFIED'
                or len(archive) != inventory['bytes'] or len(archive) > 256*2**20
                or hashlib.sha256(archive).hexdigest() != inventory['sha256']):
            raise ValueError('Verified bounded build context differs')
        vm = self.cloud.command(['compute','instances','describe',resource['name'],'--zone='+resource['zone']])
        checked_cpu(vm,resource)
        quote = quote_archive(self.root/'official-compute-skus','e2-standard-2','us-central1')
        exposure = 2*float(quote['usd_per_hour'])+.01
        def reserve(state):
            if state.get('builds_started',0) >= 3 or label in state.get('build_jobs',{}):
                raise ValueError('Build allowance consumed or label already attempted; no replay')
            opened = state.setdefault('open_exposures',{})
            admission(state,sum(r['maximum_usd'] for r in opened.values())+exposure)
            opened['cpu-build-'+label] = dict(maximum_usd=exposure,not_billed_spend=True,
                at=datetime.now(timezone.utc).isoformat(),catalog_receipt_sha256=quote['catalog_receipt_sha256'])
            state.setdefault('build_jobs',{})[label] = dict(status='INTENT',vm_id=resource['id'],
                commit=inventory['commit'],context_sha256=inventory['sha256'],cpu_label=self.cpu_label,
                cpu_restoration_proof_sha256=proof_sha)
        self.state.update(reserve)
        with (self.root/'DESTRUCTION_LOG.md').open('a',encoding='utf-8') as stream:
            stream.write('\nBEFORE CPU build '+label+': own Git transfer, Docker intermediate/cache, pip/apt '
                'temporary files, pytest tmp directories and container tmpfs disposable. --rm=false containers, '
                'source/context, failed logs and final image retained. No session directory read or uploaded.\n')
        with (self.root/'COST_LEDGER.md').open('a',encoding='utf-8') as stream:
            stream.write('\nBEFORE CPU build '+label+': '+json.dumps(quote)+'; upper '+str(exposure)+
                ' USD (2h CPU plus .01 small technical-object storage/operations); existing disk reservation unchanged.\n')
        subnet = self.cloud.command(['compute','networks','subnets','describe','cloudrag-study-central1-20261002','--region=us-central1'])
        if not subnet.get('privateIpGoogleAccess'):
            raise ValueError('CPU subnet needs Private Google Access before creator-only technical evidence upload; no VM started')
        self.cloud.command(['compute','instances','set-service-account',resource['name'],'--zone='+resource['zone'],
                           '--service-account='+SA,'--scopes=https://www.googleapis.com/auth/devstorage.read_write'])
        vm = self.cloud.command(['compute','instances','describe',resource['name'],'--zone='+resource['zone']])
        checked_cpu(vm,resource)
        project_policy = self.cloud.command(['projects','get-iam-policy','pure-loop-474323-a8'])
        session_policy = self.cloud.command(['storage','buckets','get-iam-policy','gs://'+SESSIONS])
        technical_policy = self.cloud.command(['storage','buckets','get-iam-policy','gs://'+TECHNICAL])
        # Only storage IAM is verified here. No app exists to test metadata isolation.
        validate_storage_iam(vm,project_policy,session_policy,technical_policy,sa=SA,bucket=TECHNICAL)
        startup = self.root/('cpu-build-'+label+'-startup.sh')
        with startup.open('x',encoding='utf-8',newline='\n') as stream:
            stream.write(startup_text())
        self.cloud.command(['compute','instances','add-metadata',resource['name'],'--zone='+resource['zone'],
                            '--metadata-from-file=startup-script='+str(startup)])
        job = dict(operation='I5_CPU_BUILD',purpose='technical',label=label,instance_id=resource['id'],
                   cpu_label=self.cpu_label,cpu_restoration_proof_sha256=proof_sha,
                   commit=inventory['commit'],context_sha256=inventory['sha256'],
                   context='/srv/cloudrag/iteration5/builds/'+label+'/context',bucket=TECHNICAL,
                   prefix='iteration4/iteration5/'+self.root.name+'/'+label,
                   guest_source_sha256=hashlib.sha256(source).hexdigest())
        atomic_json(self.root/('cpu-build-'+label+'-job-input.json'),job)
        def attempted(state):
            state['builds_started'] = state.get('builds_started',0)+1  # Conservatively counts launch, even if transport fails.
            state['build_jobs'][label].update(status='LAUNCH_ATTEMPTED',job_input_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest())
        self.state.update(attempted)
        try:
            self.cloud.command(['compute','instances','start',resource['name'],'--zone='+resource['zone']],timeout=600)
            deadline = time.monotonic()+300
            while True:
                try:
                    keys = self.cloud.command(['compute','instances','get-guest-attributes',resource['name'],
                                              '--zone='+resource['zone'],'--query-path=hostkeys/'])
                except ReadyPending:
                    keys = []
                if keys:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Public guest keys unavailable')
                time.sleep(15)
            raw = self.cloud.command(['compute','ssh',resource['name'],'--zone='+resource['zone'],'--tunnel-through-iap',
                '--ssh-key-expire-after=10m','--command=sudo -n python3 -B -'],input_data=stage_payload(job,archive,source),
                private_output=True,json_output=False,timeout=900)
            result = json.loads(raw)
            if result['status'] != 'CPU_BUILD_LAUNCHED_NOT_ACCEPTED' or result['instance_id'] != resource['id'] or result['context_sha256'] != inventory['sha256']:
                raise ValueError('Guest launch acknowledgement differs')
            self.state.update(lambda state:state['build_jobs'][label].update(status='RUNNING',guest_pid=result['pid']))
            atomic_json(self.root/('cpu-build-'+label+'-launch.json'),result)
            return result
        except Exception:
            self.state.update(lambda state:state['build_jobs'][label].update(status='LAUNCH_FAILED_PRESERVED'))
            self.stopped(resource)
            raise

    def collection_owner(self,label,job):
        """A delayed collector must never stop the CPU serving a newer build."""
        cpu_label = job.get('cpu_label','bootstrap01')
        resource = self.resource(cpu_label=cpu_label)
        state = json.loads((self.root/'STATE.json').read_bytes())
        jobs = state.get('build_jobs',{})
        owner = jobs.get(label,{})
        if (owner.get('vm_id') != resource['id'] or job.get('instance_id') != resource['id']
                or owner.get('commit') != job.get('commit')
                or owner.get('cpu_label','bootstrap01') != cpu_label):
            raise ValueError('Collector lacks the recorded build and CPU ownership')
        if 'cpu_restoration_proof_sha256' in job:
            actual = hashlib.sha256((self.root/('cpu-restoration-'+cpu_label+'-proof.json')).read_bytes()).hexdigest()
            if actual != job['cpu_restoration_proof_sha256'] or actual != owner.get('cpu_restoration_proof_sha256'):
                raise ValueError('Collector restoration proof changed')
        active = {'INTENT','LAUNCH_ATTEMPTED','RUNNING'}
        if any(name != label and row.get('vm_id') == resource['id']
               and row.get('status') in active for name,row in jobs.items()):
            raise ValueError('Stale collector cannot stop CPU assigned to another active build')
        return resource

    def collect(self,label):
        job = json.loads((self.root/('cpu-build-'+label+'-job-input.json')).read_bytes())
        self.collection_owner(label,job)
        storage = Storage(TECHNICAL,self.cloud.owner_token)
        name = job['prefix']+'/manifest.json'
        objects = storage.objects(job['prefix']+'/')
        candidates = [r for r in objects if r['name'] == name]
        if not candidates:
            return dict(status='BUILD_NOT_TERMINAL_OR_OWNER_RECOVERY_REQUIRED',label=label)
        if len(candidates) != 1:
            raise ValueError('Duplicate build manifest')
        manifest_data = storage.read(name,str(candidates[0]['generation']))
        if (hashlib.md5(manifest_data).digest() != base64.b64decode(candidates[0]['md5Hash'])
                or len(manifest_data) != int(candidates[0]['size'])):
            raise ValueError('Remote manifest checksum differs')
        manifest = json.loads(manifest_data)
        if manifest['commit'] != job['commit']:
            raise ValueError('Build manifest commit differs')
        destination = self.root/('linux-build-'+label+'-evidence')
        destination.mkdir(exist_ok=True)
        receipts = []
        for row in manifest['files']:
            filename = row['object'].removeprefix(job['prefix']+'/')
            if row['object'] != job['prefix']+'/'+filename or '/' in filename or not re.fullmatch(r'[a-z0-9-]+\.(?:json|stdout|stderr|receipt\.json)',filename):
                raise ValueError('Build manifest references nontechnical path')
            data = storage.read(row['object'],row['generation'])
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('Technical generation SHA256 differs')
            path = destination/filename
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError('Existing downloaded build evidence differs')
            else:
                path.write_bytes(data)
            receipts.append(row)
        required = {'pip-check.receipt.json','linux-full.receipt.json','linux-filtered.receipt.json','posix-durability.receipt.json','isolated-runtime-user.receipt.json','persistent-mount.stdout','posix-durability.stdout'}
        if 'cpu_restoration_proof_sha256' in job:
            required |= {'host-stdlib.receipt.json','host-unit-validation.receipt.json',
                         'host-systemd-version.receipt.json','host-systemd-version.stdout'}
        if manifest['status'] == 'PASS' and (not required.issubset({r['object'].split('/')[-1] for r in receipts})
                or any(json.loads((destination/name).read_bytes())['exit_code'] for name in required if name.endswith('.json'))):
            raise ValueError('PASS lacks complete zero-exit Linux checks')
        if manifest['status'] == 'PASS':
            mount = json.loads((destination/'persistent-mount.stdout').read_bytes())['filesystems']
            if (len(mount) != 1 or not mount[0]['source'].startswith('/dev/')
                    or mount[0]['target'] != '/' or mount[0]['fstype'] != 'ext4'):
                raise ValueError('Durability temporary path is not on the restored persistent root disk')
            if not seven_posix_passed((destination/'posix-durability.stdout').read_text(encoding='utf-8')):
                raise ValueError('All seven POSIX durability tests must pass without exclusions')
        proof = dict(status='TECHNICAL_BUILD_'+manifest['status']+'_DOWNLOADED_VERIFIED',label=label,
            commit=job['commit'],image_id=manifest['image_id'],files=receipts,
            manifest=dict(object=name,generation=str(candidates[0]['generation']),sha256=hashlib.sha256(manifest_data).hexdigest()),
            final_gpu_acceptance_not_inferred=True)
        resource = self.collection_owner(label,job)
        self.stopped(resource)
        proof['cpu_stopped_verified'] = True
        target = self.root/('linux-build-'+label+'-download-proof.json')
        if target.exists():
            if json.loads(target.read_bytes()) != proof:
                raise ValueError('Existing download proof differs; preserving it')
        else:
            atomic_json(target,proof)
        self.state.update(lambda state:state['build_jobs'][label].update(status=proof['status'],download_proof=str(target)))
        return proof


def seven_posix_passed(output):
    """Accept pytest normal and quiet terminal summaries, never progress text."""
    lines = [line.strip().strip('=').strip() for line in output.splitlines() if line.strip()]
    return bool(lines and re.fullmatch(r'7 passed in [0-9]+\.[0-9]+s(?: \([^\r\n]+\))?', lines[-1]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('operation',choices=('launch','collect'))
    for name in ('package','sdk','label'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--inventory')
    parser.add_argument('--cpu-label',default='bootstrap01')
    args = parser.parse_args(argv)
    require_limited()
    root = Path(args.package)
    with FileLock(str(root/'cpu-build.lock'),timeout=0):
        build = Build(root,Cloud(args.sdk,'pure-loop-474323-a8',root/'cpu-build-api'),cpu_label=args.cpu_label)
        if args.operation == 'launch':
            repo = Path(__file__).resolve().parents[2]
            if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],timeout=60).strip():
                raise ValueError('Only clean published build controller can launch')
            head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],timeout=60).decode().strip()
            if subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/fix/interview-readiness'],timeout=120).decode().split()[0] != head:
                raise ValueError('Build controller not published')
            inventory = json.loads(Path(args.inventory).read_bytes())
            if inventory['commit'] != head:
                raise ValueError('Build input is not the current published image source')
        result = (build.launch(args.label,json.loads(Path(args.inventory).read_bytes()),Path(__file__).with_name('linux_build.py').read_bytes())
                  if args.operation == 'launch' else build.collect(args.label))
        print(json.dumps(dict(status=result['status'],label=args.label)))


if __name__ == '__main__':
    main()
