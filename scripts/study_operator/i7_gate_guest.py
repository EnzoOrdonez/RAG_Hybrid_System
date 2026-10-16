"""Allowlisted owner RPC for a single terminal I7 gate cohort."""
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.study_operator.i7_gate import HOST_LIMIT_S
from scripts.study_operator.i7_gate_host import launch_command
from scripts.study_operator.managed_stores import recovery_counts
from scripts.study_operator.service_gateway import save_state
from scripts.study_operator.service_transition import cold_state
from scripts.study_operator.stimulus_host_evidence import sha


def dispatch_gate(active, request, *, execute, controller, maintenance,
                  now=lambda: datetime.now(timezone.utc)):
    root = Path(active['boot_root'])
    job = root/'i7-gate'
    op = request['operation']
    if active['config']['purpose'] != 'technical':
        raise ValueError('TECHNICAL_GATE_ONLY')
    if op == 'gate-start' and set(request) == {'operation'}:
        if job.exists():
            raise ValueError('GATE_COHORT_ALREADY_REGISTERED')
        backup = controller(active,{'operation':'backup-check'})
        copies = recovery_counts(active['session_root'],active['config']['purpose'])
        if backup.get('active_session') or any(backup[k] or copies[k] for k in ('session_count','invitation_count')):
            raise ValueError('EMPTY_TECHNICAL_PERIOD_REQUIRED')
        cold_state(json.loads((root/'sockets'/'service-state.json').read_bytes()),active['boot_id'])
        deadlines = [datetime.fromisoformat(active[k]) for k in ('guest_deadline_utc','native_deadline_utc')]
        if any(d.tzinfo is None or (d-now()).total_seconds() < HOST_LIMIT_S+300 for d in deadlines):
            raise ValueError('GATE_NATIVE_LEASE_INSUFFICIENT')
        job.mkdir(mode=0o700)
        save_state(job/'launch-intent.json',{'boot_id':active['boot_id'],'replay_allowed':False})
        maintenance()
        execute(launch_command(active),timeout=30)
        return {'status':'GATE_STARTED_NOT_GO','boot_id':active['boot_id']}
    if op == 'gate-status' and set(request) == {'operation'}:
        return json.loads((job/'result.json').read_bytes()) if (job/'result.json').exists() else {'status':'RUNNING_OR_UNKNOWN'}
    if op == 'gate-evidence' and set(request) == {'operation'}:
        result = json.loads((job/'result.json').read_bytes())
        if result['status'] != 'GATE_COMPLETE_UNANALYZED':
            raise ValueError('GATE_COMPLETE_REQUIRED')
        proof = json.loads((job/'coded-P999'/'complete.json').read_bytes())
        if sha(proof) != result['proof_sha256']:
            raise ValueError('GATE_HASH_DIFFERS')
        # Owner verifies the full proof after download; host runtime stays stdlib-only.
        return {'proof':proof,'proof_sha256':sha(proof)}
    if op == 'gate-ack' and set(request) == {'operation','proof_sha256'}:
        result = json.loads((job/'result.json').read_bytes())
        if result['status'] != 'GATE_COMPLETE_UNANALYZED' or request['proof_sha256'] != result['proof_sha256']:
            raise ValueError('GATE_DOWNLOAD_HASH_DIFFERS')
        save_state(job/'download-ack.json',{'proof_sha256':request['proof_sha256']})
        return {'status':'GATE_DOWNLOAD_ACKNOWLEDGED','proof_sha256':request['proof_sha256']}
    raise ValueError('UNKNOWN_GATE_OPERATION')
