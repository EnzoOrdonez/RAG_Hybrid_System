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
        '--property=RuntimeMaxSec=30600', '--property=TimeoutStopSec=35',
        '--property=StandardOutput=null', '--property=StandardError=null',
        '--property=ExecStopPost=-/usr/bin/docker stop --time 15 '+name,
        '--property=ExecStopPost=/usr/sbin/shutdown -h now',
        sys.executable, '-B', '-m', 'scripts.study_operator.i7_gate_host']


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
            or sha(admission_receipt(host)) != proof['host_admission_sha256']):
        raise ValueError('GATE_ADMISSION_DIFFERS')
    closing = datetime.fromisoformat(samples[-1]['at'])
    if closing.tzinfo is None:
        raise ValueError('GATE_CLOCK_UNAWARE')
    config = assessment_config(inventory)
    if (reasons(samples, config, began, reference_time=closing.timestamp()) or
            reasons(admitted, config, began, admission=True, reference_time=closing.timestamp())):
        raise ValueError('GATE_TELEMETRY_REJECTED')
    run = GateRun(proof['tasks'], identity)
    if len(proof['rows']) != 140 or len(host['calls']) != 140:
        raise ValueError('GATE_CENSUS_INCOMPLETE')
    previous = last
    for index, (row, call) in enumerate(zip(proof['rows'],host['calls'],strict=True),1):
        start, end = call['started'], call['ended']
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
    try:
        GateHost(active,root).run()
        deadline = time.monotonic()+60
        while time.monotonic() < deadline and not (root/'download-ack.json').is_file():
            time.sleep(1)
    finally:
        save_state(Path(active['boot_root'])/'stop-request.json',{'stop': True,'boot_id': active['boot_id']})


if __name__ == '__main__':
    main()
