"""One VM cold boot, fixed container, private output and independent systemd bound."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.study_gate_environment import LinuxSampler
from scripts.study_operator.deployment import app_command, assert_isolation
from scripts.study_operator.host_runtime import ROOT, metadata_unreachable
from scripts.study_operator.service_gateway import save_state
from scripts.study_operator.stimulus_host_evidence import admission_receipt, reasons, sha
from scripts.study_operator.stimulus_pipe import PrivatePipe


def owned_names(active, index):
    boot = active['boot_id']
    if (not boot or any(c not in '0123456789abcdef-' for c in boot)
            or type(index) is not int or not 1 <= index <= 12):
        raise ValueError('REGISTERED_COLD_BOOT_REQUIRED')
    return 'cloudrag-i5-stimulus-'+boot+'-'+str(index), 'cloudrag-i5-stimulus-'+boot


def launch_command(active, index):
    """The guest checks the installed flags before accepting this fixed job."""
    name,unit = owned_names(active,index)
    return ['systemd-run','--unit='+unit,'--service-type=exec',
        '--working-directory='+active['config']['host_code'],
        '--property=RuntimeMaxSec=7200','--property=TimeoutStopSec=35',
        '--property=StandardOutput=null','--property=StandardError=null',
        '--property=ExecStopPost=-/usr/bin/docker stop --time 15 '+name,
        '--property=ExecStopPost=/usr/sbin/shutdown -h now',
        sys.executable,'-B','-m','scripts.study_operator.stimulus_host','--boot-index',str(index)]


class HostCollection:
    def __init__(self, active, index, job_root, *, invoke=subprocess.run, launch=subprocess.Popen,
                 sampler=None, clock=time.monotonic, sleep=time.sleep):
        self.active, self.index, self.root = active,index,Path(job_root)
        self.invoke,self.launch,self.clock,self.sleep = invoke,launch,clock,sleep
        self.name,self.unit = owned_names(active,index)
        self.config = active['config']
        self.began = clock()
        self.deadline = self.began+7200
        self.sampler = sampler or LinuxSampler(str(Path(active['boot_root'])/'sockets'/'service-state.json'))
        self.samples,self.calls = [],[]
        self.host = dict(schema_version=1,mode='LIVE',boot_id=active['boot_id'],boot_index=index,
            app_image_id=self.config['image_id'],source_commit=self.config['commit'],native_limit_s=7200,
            preparation_started=self.began,samples=self.samples,calls=self.calls,status='INCOMPLETE_TERMINAL')
        self.assess_config = dict(model_digest=self.config['model_digest'],service_mode='fresh_runner',
                                 service_boot_id=active['boot_id'])
        self.process = None
        self.stopped = False

    def command(self, argv, seconds=3):
        result = self.invoke(argv,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=seconds)
        if result.returncode:
            raise ValueError('OWNED_STIMULUS_HOST_COMMAND_FAILED')
        return result.stdout

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        try:
            self.command(['docker','stop','--time','15',self.name],30)
        finally:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)

    def poll(self, *, force=False):
        now = self.clock()
        if now >= self.deadline:
            raise TimeoutError('COLD_BOOT_NATIVE_LIMIT_REACHED')
        if not force and self.samples and now-self.samples[-1]['monotonic_s'] < 5:
            return
        app = json.loads(self.command(['docker','inspect',self.name]))[0]
        ollama = json.loads(self.command(['docker','inspect',self.active['ollama_container']]))[0]
        if (app['Id'] != self.host['own_container_id'] or app['Image'] != self.config['image_id']
                or not app['State']['Running'] or ollama['Id'] != self.host['ollama_container_id']
                or not ollama['State']['Running'] or ollama['Config']['Image'] != self.config['ollama_image']):
            raise ValueError('OWNED_CONTAINER_IDENTITY_CHANGED')
        owned = {os.getpid()}
        for name in (self.name,self.active['ollama_container']):
            rows = self.command(['docker','top',name,'-eo','pid']).decode().splitlines()[1:]
            owned.update(int(row.strip()) for row in rows)
        row = self.sampler()
        row['owned_pids'] = sorted(owned)
        row['gpu_pids'] = [int(v.strip()) for v in self.command(
            ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().splitlines() if v.strip()]
        self.samples.append(row)
        save_state(self.root/'telemetry'/f'{len(self.samples):06d}.json',row)
        if reasons(self.samples,self.assess_config,self.began):
            raise ValueError('COLD_HOST_TELEMETRY_REJECTED')

    def admit(self, prepared):
        inventory = prepared['inventory']
        if (prepared.get('status') != 'PREPARED_NOT_ADMITTED'
                or inventory['image']['image_id'] != self.config['image_id']
                or inventory['source']['commit'] != self.config['commit']
                or inventory['observed']['boot_id'] != self.active['boot_id']):
            raise ValueError('PREPARED_IDENTITY_DIFFERS')
        self.poll(force=True)
        first = self.samples[-1]['monotonic_s']
        start = len(self.samples)-1
        while self.samples[-1]['monotonic_s']-first < 60:
            if self.clock() >= self.began+900:
                raise TimeoutError('COLD_PREPARATION_900S')
            self.sleep(1)
            self.poll()
        admitted = self.samples[start:]
        if reasons(admitted,self.assess_config,self.began,admission=True):
            raise ValueError('COLD_ADMISSION_REJECTED')
        self.host.update(admission_started=first,admission_ended=admitted[-1]['monotonic_s'],
                         admission_samples=admitted)
        receipt = admission_receipt(self.host)
        save_state(self.root/'admission.json',receipt)
        return sha(receipt)

    def run(self):
        channel = None
        try:
            if self.config['purpose'] != 'technical' or (self.root/'started.json').exists():
                raise ValueError('TECHNICAL_COLD_BOOT_NO_REPLAY_REQUIRED')
            for path in (self.root/'telemetry',self.root/'coded-P999'):
                path.mkdir(mode=0o700,exist_ok=False)
            save_state(self.root/'started.json',dict(boot_id=self.active['boot_id'],boot_index=self.index,
                       started_utc=datetime.now(timezone.utc).isoformat(),replay_allowed=False))
            app = json.loads(self.command(['docker','inspect',self.active['app_container']]))[0]
            if app['State']['Running'] or not (Path(self.active['boot_root'])/'maintenance.json').is_file():
                raise ValueError('MAINTENANCE_REQUIRED')
            argv = app_command(self.config,self.active['boot_root'],self.active['session_root'],self.name,
                               operation='stimulus')
            argv.insert(2,'-i')
            argv[argv.index('--streamlit-socket')+1] = '/web/stimulus.sock'
            argv.extend(['--boot-index',str(self.index)])
            self.process = self.launch(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            # Allow only bounded creation time; pipeline preparation counts from began.
            for _ in range(30):
                try:
                    observed = json.loads(self.command(['docker','inspect',self.name]))[0]
                    if observed['State']['Running']:
                        break
                except ValueError:
                    pass
                if self.process.poll() is not None:
                    raise ValueError('STIMULUS_CONTAINER_LAUNCH_FAILED')
                self.sleep(.1)
            else:
                raise TimeoutError('STIMULUS_CONTAINER_LAUNCH_TIMEOUT')
            assert_isolation(observed,self.config['image_id'])
            isolated = metadata_unreachable(self.name,invoke=self.invoke)
            ollama = json.loads(self.command(['docker','inspect',self.active['ollama_container']]))[0]
            self.host.update(own_container_id=observed['Id'],ollama_container_id=ollama['Id'],
                             isolation_verified=True,metadata_unreachable=isolated['metadata_unreachable'])
            channel = PrivatePipe(self.process,self.stop,self.poll)
            prepared = channel.receive(max(.001,900-(self.clock()-self.began)))
            admission_sha = self.admit(prepared)
            channel.send(dict(operation='admit',receipt_sha256=admission_sha))
            acknowledged = channel.receive(min(30,self.deadline-self.clock()))
            if acknowledged != dict(status='HOST_ADMISSION_ACKNOWLEDGED',receipt_sha256=admission_sha):
                raise ValueError('COLD_ADMISSION_NOT_ACKNOWLEDGED')
            for index in range(1,prepared['slots']+1):
                self.poll(force=True)
                start = self.clock()
                channel.send(dict(index=index))
                response = channel.receive(min(600,self.deadline-start))
                end = self.clock()
                if response.get('status') != 'OBSERVATION' or response['row']['index'] != index:
                    raise ValueError('COLD_CALL_TERMINAL')
                row = response['row']
                self.calls.append(dict(index=index,started=start,ended=end,row_sha256=sha(row)))
                save_state(self.root/'coded-P999'/f'{index:03d}.json',row)
                self.poll(force=True)
                # Technical progress deliberately excludes query and answer content.
                save_state(self.root/'progress.json',dict(index=index,total=prepared['slots'],
                    elapsed_s=self.clock()-self.began,eta_s=(self.clock()-self.began)/index*(prepared['slots']-index)))
            channel.send(dict(operation='complete'))
            final = channel.receive(min(30,self.deadline-self.clock()))
            if final.get('status') != 'COMPLETE_NOT_ACCEPTANCE':
                raise ValueError('COMPLETE_COLD_BOOT_REQUIRED')
            proof = final['proof']
            self.host.update(status='COMPLETE',ended=self.clock())
            proof['host_evidence'] = self.host
            # Verification inside the same fixed image keeps host Python stdlib-only.
            save_state(self.root/'coded-P999'/'complete.json',proof)
            save_state(self.root/'result.json',dict(status='BOOT_COMPLETE_UNANALYZED',boot_id=self.active['boot_id'],
                       boot_index=self.index,calls=len(self.calls),proof_sha256=sha(proof),acceptance_not_inferred=True))
            return proof
        except BaseException as error:
            save_state(self.root/'result.json',dict(status='INCOMPLETE_TERMINAL',boot_id=self.active['boot_id'],
                boot_index=self.index,calls=len(self.calls),error_type=type(error).__name__,replay_allowed=False))
            raise
        finally:
            try:
                if channel:
                    channel.close()
                else:
                    self.stop()
            finally:
                if self.host['status'] != 'COMPLETE':
                    save_state(Path(self.active['boot_root'])/'stop-request.json',dict(stop=True,boot_id=self.active['boot_id']))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--boot-index',required=True,type=int,choices=range(1,13))
    args = parser.parse_args(argv)
    if os.name != 'posix':
        raise ValueError('LINUX_HOST_REQUIRED')
    active = json.loads((ROOT/'active.json').read_bytes())
    if active['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('ACTIVE_BOOT_DIFFERS')
    job_root = Path(active['boot_root'])/'stimulus'
    try:
        HostCollection(active,args.boot_index,job_root).run()
        # Give the owner one bounded minute to download, verify and acknowledge
        # the private proof. The native unit then shuts down independently.
        deadline = time.monotonic()+60
        while time.monotonic() < deadline and not (job_root/'download-ack.json').is_file():
            time.sleep(1)
    finally:
        save_state(Path(active['boot_root'])/'stop-request.json',dict(stop=True,boot_id=active['boot_id']))


if __name__ == '__main__':
    raise SystemExit(main())
