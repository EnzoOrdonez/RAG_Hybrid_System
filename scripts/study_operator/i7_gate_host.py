"""Host-owned isolated gate with an independent native shutdown bound."""
import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from scripts.study_operator.deployment import app_command, assert_isolation
from scripts.study_operator.host_runtime import ROOT, metadata_unreachable
from scripts.study_operator.i7_gate import HOST_LIMIT_S, GateRun, collect
from scripts.study_operator.service_gateway import save_state
from scripts.study_operator.stimulus_host import HostCollection
from scripts.study_operator.stimulus_host_evidence import (
    admission_receipt,
    assessment_config,
    reasons,
    sha,
)
from scripts.study_operator.stimulus_pipe import PrivatePipe


def launch_command(active):
    if not re.fullmatch('[a-f0-9-]{32,36}', active['boot_id']):
        raise ValueError('GATE_BOOT_ID_REQUIRED')
    name = 'cloudrag-i7-gate-'+active['boot_id']
    return ['systemd-run', '--unit='+name, '--service-type=exec',
        '--working-directory='+active['config']['host_code'],
        '--property=RuntimeMaxSec=9000', '--property=TimeoutStopSec=35',
        '--property=StandardOutput=null', '--property=StandardError=null',
        '--property=ExecStopPost=-/usr/bin/docker stop --time 15 '+name,
        '--property=ExecStopPost=/usr/sbin/shutdown -h now',
        sys.executable, '-B', '-m', 'scripts.study_operator.i7_gate_host']


def hot_reasons(samples, config, began, *, reference_time=None):
    from scripts.study_gate_environment import assess
    return sorted(set(reasons(samples,config,began,reference_time=reference_time)) | set(
        assess(samples,config,admission=True,
               allowed_pids={pid for row in samples for pid in row['owned_pids']},reference_time=reference_time)))


def gate_admission_receipt(host):
    return dict(admission_receipt(host),preparation_sha256=host['preparation_sha256'],warm_service_sequence=9)


def verify_preparation(events, inventory):
    kinds=[e['kind'] for e in events]
    if (kinds.count('preparation_started')!=1 or kinds.count('warmup_response')!=2
            or kinds.count('pipeline_prepared')!=2 or kinds.count('preparation_ready')!=1
            or len(events)!=6 or len({e['instance'] for e in events})!=1
            or {e['scope'] for e in events}!={'i7-gate-'+inventory['observed']['boot_id']}):
        raise ValueError('HOT_PREPARATION_CENSUS_DIFFERS')
    ready=next(e for e in events if e['kind']=='preparation_ready')
    if (ready['status']!='ready' or set(ready['systems'])!={'hybrid','no_rag'}
            or ready['identity']['model_digest']!=inventory['ollama']['digest'].removeprefix('sha256:')):
        raise ValueError('HOT_PREPARATION_IDENTITY_DIFFERS')
    for kind in ('warmup_response','pipeline_prepared'):
        if {e['system'] for e in events if e['kind']==kind}!={'hybrid','no_rag'}:
            raise ValueError('HOT_PREPARATION_SYSTEMS_DIFFERS')


def verify(proof):
    """Recompute accounting and host evidence at its actual closing observation."""
    host, inventory = proof['host_evidence'], proof['inventory']
    identity = {'boot_id': inventory['observed']['boot_id'], 'image_id': inventory['image']['image_id'],
                    'software_sha256': proof['software_sha256']}
    from scripts.study_operator.stimulus_evidence import software_projection
    if (proof['software_sha256'] != sha(software_projection(inventory))
            or host['boot_id'] != identity['boot_id'] or host['app_image_id'] != identity['image_id']
            or host['source_commit'] != inventory['source']['commit']
            or host['status'] != 'COMPLETE' or host['mode'] != 'LIVE'
            or host['native_limit_s'] != HOST_LIMIT_S or not host['isolation_verified']
            or not host['metadata_unreachable'] or not host['own_container_id'] or not host['ollama_container_id']):
        raise ValueError('GATE_HOST_IDENTITY_DIFFERS')
    began, first, last, ended = [host[k] for k in
        ('preparation_started','admission_started','admission_ended','ended')]
    if not began <= first <= last <= ended or last-began > 900 or ended-began > HOST_LIMIT_S:
        raise ValueError('GATE_HOST_LIMIT_DIFFERS')
    samples = host['samples']
    admitted = [s for s in samples if first <= s['monotonic_s'] <= last]
    if (admitted != host['admission_samples'] or not admitted or last-first < 60
            or admitted[0]['monotonic_s'] != first or admitted[-1]['monotonic_s'] != last
            or any(not began <= s['monotonic_s'] <= ended for s in samples)
            or any(a['monotonic_s'] >= b['monotonic_s'] for a,b in itertools.pairwise(samples))
            or sha(gate_admission_receipt(host)) != proof['host_admission_sha256']
            or sha(proof['preparation_events'])!=host['preparation_sha256']):
        raise ValueError('GATE_ADMISSION_DIFFERS')
    verify_preparation(proof['preparation_events'],inventory)
    closing = datetime.fromisoformat(samples[-1]['at'])
    if closing.tzinfo is None:
        raise ValueError('GATE_CLOCK_UNAWARE')
    config = assessment_config(inventory)
    if (reasons(samples, config, began, reference_time=closing.timestamp()) or
            hot_reasons(admitted, config, began, reference_time=closing.timestamp())
            or any(s['service_state']['sequence']!=9 for s in admitted)):
        raise ValueError('GATE_TELEMETRY_REJECTED')
    windows=host.get('window_admissions',[])
    if [w['index'] for w in windows]!=[21,81]:
        raise ValueError('TWO_WINDOW_PREFLIGHTS_REQUIRED')
    for window in windows:
        rows=window['samples']
        if (not rows or rows!=[s for s in samples if rows[0]['monotonic_s']<=s['monotonic_s']<=rows[-1]['monotonic_s']]
                or hot_reasons(rows,config,began,reference_time=closing.timestamp())
                or any(s['service_state']['sequence']!=9+4*(window['index']-1) for s in rows)
                or rows[0]['monotonic_s']<host['calls'][window['index']-2]['ended']
                or rows[-1]['monotonic_s']>host['calls'][window['index']-1]['started']
                or window['inventory_sha256']!=sha(inventory)):
            raise ValueError('WINDOW_PREFLIGHT_DIFFERS')
    run = GateRun(proof['tasks'], identity)
    if len(proof['rows']) != 140 or len(host['calls']) != 140:
        raise ValueError('GATE_CENSUS_INCOMPLETE')
    previous = last
    previous_pids = set()
    from scripts.study_operator.stimulus_evidence import OPTIONS
    for index, (row, call) in enumerate(zip(proof['rows'],host['calls'],strict=True),1):
        start, end = call['started'], call['ended']
        state = row['service_after']
        pids = state.get('runner_pids',[])
        if (row.get('generation_options') != OPTIONS or not re.fullmatch('[a-f0-9]{64}',row.get('request_sha256',''))
                or not isinstance(row.get('raw_text'),str) or not row['raw_text'].strip()
                or state.get('boot_id') != identity['boot_id'] or state.get('mode') != 'fresh_runner'
                or state.get('phase') != 'RESIDENT' or state.get('sequence') != 9+4*index
                or not pids or len(set(pids)) != len(pids) or previous_pids.intersection(pids)
                or any(type(pid) is not int or pid <= 0 for pid in pids)):
            raise ValueError('GATE_GENERATION_OR_RUNNER_POLICY_DIFFERS')
        previous_pids.update(pids)
        backup=row.get('backup',{})
        if (backup.get('status')!='complete' or not re.fullmatch('[a-f0-9]{64}',backup.get('sha256',''))
                or set(backup.get('objects',{}))!={'full_session.json','export_manifest.json'}
                or backup['objects']['full_session.json']['sha256']!=backup['sha256']):
            raise ValueError('GATE_DURABLE_BACKUP_REQUIRED')
        for name,receipt in backup['objects'].items():
            if (not str(receipt.get('generation','')).isdigit()
                    or not re.fullmatch('[a-f0-9]{64}',receipt.get('sha256',''))
                    or not receipt.get('object','').endswith('/'+row['session_id']+'/'+name)):
                raise ValueError('GATE_BACKUP_GENERATION_DIFFERS')
        if (call['index'] != index or call['row_sha256'] != sha(row)
                or row.get('identity') != identity or row.get('index') != index
                or not previous <= start < end <= ended
                or not any(0 <= start-s['monotonic_s'] <= 15 for s in samples)
                or not any(0 <= s['monotonic_s']-end <= 15 for s in samples)):
            raise ValueError('GATE_FRONTIER_DIFFERS')
        run.record(index, row, started=start, ended=end)
        previous = end
    if run.decision() != proof['result']:
        raise ValueError('GATE_AGGREGATE_DIFFERS')
    return run.decision()


class GateHost(HostCollection):
    def __init__(self, active, job_root, **kwargs):
        super().__init__(active, 1, job_root, **kwargs)
        self.name = self.unit = 'cloudrag-i7-gate-'+active['boot_id']
        self.deadline = self.began+HOST_LIMIT_S
        self.host['native_limit_s'] = HOST_LIMIT_S

    def idle_admission(self):
        self.poll(force=True)
        first=self.samples[-1]['monotonic_s']
        start=len(self.samples)-1
        while self.samples[-1]['monotonic_s']-first<60:
            self.sleep(1)
            self.poll()
        admitted=self.samples[start:]
        if hot_reasons(admitted,self.assess_config,self.began):
            raise ValueError('HOT_GATE_ADMISSION_REJECTED')
        return admitted

    def admit(self, prepared):
        inventory=prepared['inventory']
        if (prepared.get('status')!='PREPARED_NOT_ADMITTED'
                or inventory['image']['image_id']!=self.config['image_id']
                or inventory['source']['commit']!=self.config['commit']
                or inventory['observed']['boot_id']!=self.active['boot_id']
                or prepared['initial_service_state']['sequence']!=9):
            raise ValueError('HOT_GATE_PREPARED_IDENTITY_DIFFERS')
        verify_preparation(prepared['preparation_events'],inventory)
        admitted=self.idle_admission()
        if admitted[-1]['monotonic_s']-self.began>900:
            raise TimeoutError('GATE_PREPARATION_900S')
        self.host.update(admission_started=admitted[0]['monotonic_s'],
            admission_ended=admitted[-1]['monotonic_s'],admission_samples=admitted,
            preparation_sha256=sha(prepared['preparation_events']))
        receipt=gate_admission_receipt(self.host)
        save_state(self.root/'admission.json',receipt)
        return sha(receipt)

    def revalidate(self, channel, index):
        channel.send({'operation':'revalidate','index':index})
        result=channel.receive(min(900,self.deadline-self.clock()))
        if (result.get('status')!='WINDOW_REVALIDATED' or result.get('index')!=index
                or result.get('inventory_sha256')!=self.inventory_sha):
            raise ValueError('WINDOW_REVALIDATION_FAILED')
        rows=self.idle_admission()
        receipt={'index':index,'samples':rows,'inventory_sha256':self.inventory_sha}
        self.host.setdefault('window_admissions',[]).append(receipt)
        save_state(self.root/f'window-{index}-admission.json',receipt)

    def run(self):
        channel, run = None, None
        try:
            if self.config['purpose'] != 'technical' or (self.root/'started.json').exists():
                raise ValueError('TECHNICAL_GATE_NO_REPLAY_REQUIRED')
            for directory in ('telemetry','coded-P999'):
                (self.root/directory).mkdir(mode=0o700,exist_ok=False)
            save_state(self.root/'started.json',{'boot_id': self.active['boot_id'], 'replay_allowed': False})
            app = json.loads(self.command(['docker','inspect',self.active['app_container']]))[0]
            if app['State']['Running'] or not (Path(self.active['boot_root'])/'maintenance.json').is_file():
                raise ValueError('MAINTENANCE_REQUIRED')
            argv = app_command(self.config,self.active['boot_root'],self.active['session_root'],self.name,
                               operation='i7-gate')
            argv.insert(2,'-i')
            self.process = self.launch(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
            for _ in range(30):
                try:
                    observed = json.loads(self.command(['docker','inspect',self.name]))[0]
                    if observed['State']['Running']:
                        break
                except ValueError:
                    pass
                self.sleep(.1)
            else:
                raise TimeoutError('GATE_CONTAINER_LAUNCH_TIMEOUT')
            assert_isolation(observed,self.config['image_id'])
            isolated = metadata_unreachable(self.name,invoke=self.invoke)
            ollama = json.loads(self.command(['docker','inspect',self.active['ollama_container']]))[0]
            self.host.update(own_container_id=observed['Id'],ollama_container_id=ollama['Id'],
                             isolation_verified=True,metadata_unreachable=isolated['metadata_unreachable'])
            channel = PrivatePipe(self.process,self.stop,self.poll)
            prepared = channel.receive(max(.001,900-(self.clock()-self.began)))
            admission_sha = self.admit(prepared)
            channel.send({'operation': 'admit','receipt_sha256': admission_sha})
            if channel.receive(30) != {'status': 'HOST_ADMISSION_ACKNOWLEDGED','receipt_sha256': admission_sha}:
                raise ValueError('GATE_ADMISSION_NOT_ACKNOWLEDGED')
            inventory = prepared['inventory']
            self.inventory_sha=sha(inventory)
            from scripts.study_operator.stimulus_evidence import software_projection
            software = sha(software_projection(inventory))
            # Read sealed tasks locally; never accept a task list from owner RPC.
            protocol = json.loads((Path(self.active['boot_root'])/'config'/'study.json').read_bytes())
            tasks = protocol['tasks']['T1']+protocol['tasks']['T2']
            run = GateRun(tasks,{'boot_id': self.active['boot_id'],'image_id': self.config['image_id'],
                                    'software_sha256': software})
            result = collect(channel,self,run)
            self.host.update(status='COMPLETE', ended=self.clock(), calls=run.calls)
            proof = {'inventory': inventory,'software_sha256': software,'tasks': tasks,'rows': run.rows,
                         'prerequisites_sha256':sha(json.loads((self.root/'prerequisites.json').read_bytes())),
                         'preparation_events':prepared['preparation_events'],
                         'host_evidence': self.host,'host_admission_sha256': admission_sha,'result': result}
            verify(proof)
            save_state(self.root/'coded-P999'/'complete.json',proof)
            save_state(self.root/'result.json',{'status': 'GATE_COMPLETE_UNANALYZED',
                       'proof_sha256': sha(proof),'calls': 140})
            return proof
        except BaseException as error:
            self.host.update(status='INCOMPLETE_TERMINAL', calls=run.calls if run else [])
            save_state(self.root/'terminal.json',{'host_evidence': self.host,
                'rows': run.rows if run else [],'error_type': type(error).__name__,'replay_allowed': False})
            save_state(self.root/'result.json',{'status': 'INCOMPLETE_TERMINAL',
                'calls': len(run.rows) if run else 0,'error_type': type(error).__name__,'replay_allowed': False})
            raise
        finally:
            channel.close() if channel else self.stop()


def main():
    if os.name != 'posix':
        raise ValueError('LINUX_GATE_HOST_REQUIRED')
    active = json.loads((ROOT/'active.json').read_bytes())
    if active['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('ACTIVE_BOOT_DIFFERS')
    root = Path(active['boot_root'])/'i7-gate'
    host=GateHost(active,root)
    try:
        try:
            host.run()
        except BaseException:
            if not (root/'terminal.json').is_file():
                raise
        # Finalization fits inside the unchanged independent 9000-second bound.
        deadline = min(host.deadline,time.monotonic()+600)
        while time.monotonic() < deadline and not (root/'finalization-ack.json').is_file():
            time.sleep(1)
    finally:
        save_state(Path(active['boot_root'])/'stop-request.json',{'stop': True,'boot_id': active['boot_id']})


if __name__ == '__main__':
    main()
