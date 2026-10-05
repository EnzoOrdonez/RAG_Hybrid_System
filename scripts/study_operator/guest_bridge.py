"""Fixed owner RPC. Session content only in private stdin/stdout, never logs."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from scripts.study_operator.deployment import app_command, assert_isolation  # noqa: E402
from scripts.study_operator.host_runtime import ROOT, certificate, metadata_unreachable  # noqa: E402
from scripts.study_operator.service_gateway import save_state  # noqa: E402


@contextmanager
def owner_lock():
    import fcntl

    with (ROOT/'owner-bridge.lock').open('a') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def execute(argv, request=None, *, timeout=180):
    result = subprocess.run(argv,input=json.dumps(request).encode() if request else None,
                            stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=timeout)
    if result.returncode:
        raise ValueError('OWNED_GUEST_OPERATION_FAILED')
    return json.loads(result.stdout) if result.stdout.strip() else {}


def controller(active, request, *, maintenance=False):
    root = Path(active['boot_root'])
    if maintenance:
        observed = execute(['docker','inspect',active['app_container']])[0]
        if observed['State']['Running'] or not (root/'maintenance.json').is_file():
            raise ValueError('MAINTENANCE_NOT_VERIFIED')
        name = 'cloudrag-i4-maintenance-'+uuid.uuid4().hex
        command = app_command(active['config'],root,active['session_root'],name)
        # A separate retained control container uses the same image and fixed mounts, without networking.
        command.insert(2,'-i')
        index = command.index('-m')
        command[index:] = ['-m','scripts.study_operator.session_control']
        command[command.index('--entrypoint'):command.index('--entrypoint')] = ['-e','CLOUDRAG_MAINTENANCE=1']
        return execute(command,request)
    return execute(['docker','exec','-i',active['app_container'],'python','-m',
                    'scripts.study_operator.session_control'],request)


def dispatch(request):
    active = json.loads((ROOT/'active.json').read_text(encoding='utf-8'))
    root = Path(active['boot_root'])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if (active['boot_id'] != boot or root != ROOT/'boots'/boot or not root.is_dir()):
        raise ValueError('NO_ACTIVE_BOOT')
    operation = request['operation']
    if operation == 'stop':
        backup = controller(active,dict(operation='backup-check'))
        save_state(root/'stop-request.json',dict(stop=True,boot_id=boot))
        return dict(status='STOP_REQUESTED',failover_data_reconciled=not backup['session_count'] and
                    not backup['invitation_count'],unfinished_session_preserved=backup['active_session'])
    if operation == 'preflight':
        ready_path = root/'ready.json'
        if not ready_path.exists() or (root/'maintenance.json').exists():
            raise ValueError('READY_NOT_AVAILABLE')
        observed = execute(['docker','inspect',active['app_container']])[0]
        if not observed['State']['Running']:
            raise ValueError('APP_NOT_RUNNING')
        assert_isolation(observed,active['config']['image_id'])
        metadata_unreachable(active['app_container'])
        execute(['docker','exec',active['app_container'],'python','scripts/cloud_entrypoint.py','verify',
                 '--deployment','/deployment/deployment.json'])
        backup = controller(active,dict(operation='backup-check'))
        ready = json.loads(ready_path.read_text())
        if certificate(active['config']['hostname'])['certificate_sha256'] != ready['tls']['certificate_sha256']:
            raise ValueError('LIVE_CERTIFICATE_CHANGED')
        return dict(status='PREFLIGHT',ready=ready,boot_id=boot,metadata_unreachable=True,
                    guest_deadline_utc=active['guest_deadline_utc'],native_deadline_utc=active['native_deadline_utc'],
                    active_session=backup['active_session'],session_count=backup['session_count'],
                    invitation_count=backup['invitation_count'])
    if operation == 'invite':
        if (root/'maintenance.json').exists():
            raise ValueError('MAINTENANCE_ACTIVE')
        return execute(['docker','exec','-i',active['app_container'],'python','scripts/cloud_entrypoint.py','invite',
            '--deployment','/deployment/deployment.json','--request','-'],request)
    if operation == 'revoke':
        return controller(active,request)
    if operation == 'maintenance':
        result = subprocess.run(['docker','stop','--time','30',active['app_container']],
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45)
        if result.returncode or execute(['docker','inspect',active['app_container']])[0]['State']['Running']:
            raise ValueError('APP_STOP_NOT_VERIFIED')
        save_state(root/'maintenance.json',dict(status='ADMISSION_CLOSED',boot_id=boot))
        return dict(status='MAINTENANCE',admission_closed=True)
    if operation in {'inventory','download-disk','clean-disk','export','restore'}:
        return controller(active,request,maintenance=True)
    if operation == 'technical-evidence':
        # Explicit allowlist. Never upload the deployment tree, sessions, Caddy storage or logs.
        files = {}
        for path in [root/name for name in ('ready.json','failure.json','stopped.json')]+list((root/'meta').glob('*')):
            if path.is_file() and path.name in {'ready.json','failure.json','stopped.json','environment_identity.json',
                'host-runtime.json','image-receipt.json','service-policy.json','deployment-packages.json'}:
                files[path.name] = json.loads(path.read_text(encoding='utf-8'))
        return dict(status='TECHNICAL_EVIDENCE',files=files)
    raise ValueError('UNKNOWN_OWNER_OPERATION')


def main():
    with open(os.devnull,'w') as discard,redirect_stdout(discard),redirect_stderr(discard):
        try:
            request = json.load(sys.stdin)
            with owner_lock():
                result = dispatch(request)
        except Exception:
            result = dict(status='ERROR',reason='GUEST_OPERATION_REJECTED',
                          next_action='Verifica status, READY y mantenimiento; conserva los recibos antes de repetir.')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
