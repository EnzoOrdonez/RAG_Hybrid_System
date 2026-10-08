"""Fixed private owner operations for one registered cold VM boot."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys

from scripts.study_operator.managed_stores import recovery_counts
from scripts.study_operator.service_gateway import save_state
from scripts.study_operator.service_transition import cold_state
from scripts.study_operator.stimulus_host import launch_command, owned_names
from scripts.study_operator.stimulus_host_evidence import sha


def dispatch_stimulus(active, request, *, execute, controller, maintenance,
                      now=lambda: datetime.now(timezone.utc)):
    root = Path(active['boot_root'])
    job = root/'stimulus'
    boot = active['boot_id']
    operation = request['operation']
    if active['config']['purpose'] != 'technical':
        raise ValueError('TECHNICAL_STIMULUS_ONLY')
    if operation == 'stimulus-start':
        if set(request) != {'operation','boot_index'}:
            raise ValueError('REGISTERED_STIMULUS_REQUEST_REQUIRED')
        index = request['boot_index']
        name,unit = owned_names(active,index)
        if job.exists():
            raise ValueError('STIMULUS_BOOT_ALREADY_REGISTERED')
        backup = controller(active,dict(operation='backup-check'))
        copies = recovery_counts(active['session_root'],active['config']['purpose'])
        if (backup.get('active_session') or
                any(backup[key] or copies[key] for key in ('session_count','invitation_count'))):
            raise ValueError('EMPTY_TECHNICAL_PERIOD_REQUIRED')
        cold_state(json.loads((root/'sockets'/'service-state.json').read_bytes()),boot)
        deadlines = [datetime.fromisoformat(active[key]) for key in ('guest_deadline_utc','native_deadline_utc')]
        if any(d.tzinfo is None for d in deadlines) or min((d-now()).total_seconds() for d in deadlines) < 125*60:
            raise ValueError('STIMULUS_REQUIRES_125_MINUTES')
        for binary in ('/usr/bin/docker','/usr/sbin/shutdown'):
            if not os.access(binary,os.X_OK):
                raise ValueError('STIMULUS_NATIVE_BINARY_MISSING')
        command = launch_command(active,index)
        job.mkdir(mode=0o700)
        unit_file = job/(unit+'.service')
        exec_args = command[command.index(sys.executable):]
        unit_file.write_text('[Unit]\nDescription=Owned cold stimulus\n[Service]\nType=exec\n'
            +'WorkingDirectory='+active['config']['host_code']+'\n'
            +'RuntimeMaxSec=7200\nTimeoutStopSec=35\nStandardOutput=null\nStandardError=null\n'
            +'ExecStart='+shlex.join(exec_args)+'\n'
            +'ExecStopPost=-/usr/bin/docker stop --time 15 '+name+'\n'
            +'ExecStopPost=/usr/sbin/shutdown -h now\n',encoding='utf-8')
        execute(['systemd-analyze','verify',str(unit_file)])
        save_state(job/'launch-intent.json',dict(boot_id=boot,boot_index=index,unit=unit,container=name,
                    native_limit_s=7200,replay_allowed=False))
        maintenance()
        execute(command,timeout=30)
        return dict(status='STIMULUS_STARTED_NOT_ACCEPTANCE',boot_id=boot,boot_index=index,unit=unit,
                    native_limit_s=7200,replay_allowed=False)
    if operation == 'stimulus-status' and set(request) == {'operation'}:
        result = json.loads((job/'result.json').read_bytes()) if (job/'result.json').exists() else dict(status='RUNNING_OR_LAUNCH_UNCONFIRMED')
        progress = json.loads((job/'progress.json').read_bytes()) if (job/'progress.json').exists() else {}
        return dict(result=result,progress=progress,acceptance_not_inferred=True)
    if operation == 'stimulus-evidence' and set(request) == {'operation'}:
        result = json.loads((job/'result.json').read_bytes())
        if result['status'] != 'BOOT_COMPLETE_UNANALYZED':
            raise ValueError('COMPLETE_STIMULUS_BOOT_REQUIRED')
        proof = json.loads((job/'coded-P999'/'complete.json').read_bytes())
        if sha(proof) != result['proof_sha256']:
            raise ValueError('STIMULUS_PRIVATE_PROOF_CHANGED')
        return dict(status='PRIVATE_STIMULUS_EVIDENCE',proof=proof,proof_sha256=result['proof_sha256'])
    if operation == 'stimulus-ack' and set(request) == {'operation','proof_sha256'}:
        result = json.loads((job/'result.json').read_bytes())
        if result['status'] != 'BOOT_COMPLETE_UNANALYZED' or request['proof_sha256'] != result['proof_sha256']:
            raise ValueError('STIMULUS_DOWNLOAD_HASH_DIFFERS')
        save_state(job/'download-ack.json',dict(proof_sha256=request['proof_sha256'],verified_by_owner=True))
        return dict(status='STIMULUS_DOWNLOAD_ACKNOWLEDGED',proof_sha256=request['proof_sha256'])
    raise ValueError('UNKNOWN_STIMULUS_OPERATION')
