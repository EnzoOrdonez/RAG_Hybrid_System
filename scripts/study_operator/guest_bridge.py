"""Fixed owner RPC. Session content only in private stdin/stdout, never logs."""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from scripts.study_operator.deployment import app_command, assert_isolation  # noqa: E402
from scripts.study_operator.host_runtime import ROOT, certificate, metadata_unreachable  # noqa: E402
from scripts.study_operator.managed_stores import (  # noqa: E402
    collection, managed_stores, recovery_counts, recovery_key, scoped_downloads, snapshot_safe,
)
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


def stores(active):
    return managed_stores(active['session_root'],active['config']['purpose'])


def disk_dispatch(active, request):
    operation = request['operation']
    roots = stores(active)
    if operation == 'restore':
        folder = Path(active['session_root']).parent/'recoveries'/recovery_key(request)
        for directory in (folder.parent,folder,folder/'sessions',folder/'private-inventory'):
            if directory.is_symlink():
                raise ValueError('RECOVERY_SYMLINK_REJECTED')
            directory.mkdir(mode=0o700,exist_ok=True)
        target = folder/'sessions'
        marker = target/'_i4_root.json'
        expected = dict(schema_version=1,purpose=active['config']['purpose'])
        if marker.exists() and json.loads(marker.read_text()) != expected:
            raise ValueError('RECOVERY_PURPOSE_CHANGED')
        if not marker.exists():
            save_state(marker,expected)
        for directory in (target,folder/'private-inventory'):
            os.chown(directory,10001,10001)
        clone = dict(active,session_root=str(target))
        result = controller(clone,request,maintenance=True)
        return dict(result,recovery_store_id=folder.name,new_app_instance=True,original_store_unchanged=True)
    if operation == 'inventory':
        plans = [dict(store_id=key,plan=controller(dict(active,session_root=str(path)),request,
                      maintenance=True)['plan']) for key,path in roots.items()]
        return dict(status='INVENTORIED_ALL_APP_COPIES',plan=collection(plans))
    if operation == 'export':
        # Research export uses primary sessions; restored copies are verification copies.
        return controller(active,request,maintenance=True)
    plan = request['plan']
    if (plan.get('schema_version') != 2 or {item['store_id'] for item in plan['stores']} != set(roots)
            or len(plan['stores']) != len(roots) or collection(plan['stores']) != plan):
        raise ValueError('MANAGED_STORE_INVENTORY_CHANGED')
    files, results = [], []
    for item in plan['stores']:
        key = item['store_id']
        payload = dict(request,plan=item['plan'])
        if operation == 'clean-disk':
            payload['verified_downloads'] = scoped_downloads(request['verified_downloads'],key)
        result = controller(dict(active,session_root=str(roots[key])),payload,maintenance=True)
        if operation == 'download-disk':
            files.extend(dict(row,store_id=key) for row in result['files'])
        else:
            results.append(dict(store_id=key,result=result['result']))
    if operation == 'download-disk':
        return dict(status='DISK_DOWNLOADED_PRIVATE',files=files)
    return dict(status='ALL_APP_COPIES_CLEANED',result=dict(empty=all(row['result']['empty'] for row in results),
                                                         stores=results))


def current_boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def technical_files(root):
    # A failed freeze has no active.json. Read only this boot's allowlisted receipts.
    files = {}
    allowed = {'ready.json','failure.json','stopped.json','environment_identity.json','host-runtime.json',
               'image-receipt.json','service-policy.json','deployment-packages.json'}
    candidates = [root/name for name in ('ready.json','failure.json','stopped.json')]
    candidates += list((root/'meta').glob('*'))+list(root.glob('command-*.json'))
    for path in candidates:
        if (not path.is_symlink() and path.is_file() and
                (path.name in allowed or re.fullmatch(r'command-[0-9]{4}\.json',path.name))):
            files[path.name] = json.loads(path.read_text(encoding='utf-8'))
    return dict(status='TECHNICAL_EVIDENCE',files=files,session_content_excluded=True)


def dispatch(request):
    if request.get('operation') == 'snapshot-safety':
        return snapshot_safe(ROOT/'periods')
    boot = current_boot()
    current = ROOT/'boots'/boot
    if request.get('operation') == 'technical-evidence':
        return dict(technical_files(current),boot_id=boot)
    if request.get('operation') == 'preflight':
        if (current/'failure.json').is_file():
            return dict(status='ERROR',reason='BOOTSTRAP_FAILED',
                        next_action='Ejecuta diagnostics, conserva sus recibos y después stop; corrige la causa antes de repetir start.')
        if not (current/'ready.json').is_file():
            return dict(status='WAITING',reason='BOOTSTRAP_PENDING')
    active = json.loads((ROOT/'active.json').read_text(encoding='utf-8'))
    root = Path(active['boot_root'])
    if (active['boot_id'] != boot or root != ROOT/'boots'/boot or not root.is_dir()):
        raise ValueError('NO_ACTIVE_BOOT')
    operation = request['operation']
    if operation == 'stop':
        backup = controller(active,dict(operation='backup-check'))
        copies = recovery_counts(active['session_root'],active['config']['purpose'])
        save_state(root/'stop-request.json',dict(stop=True,boot_id=boot))
        return dict(status='STOP_REQUESTED',failover_data_reconciled=not backup['session_count'] and
                    not backup['invitation_count'] and not copies['session_count'] and not copies['invitation_count'],
                    unfinished_session_preserved=backup['active_session'])
    if operation == 'preflight':
        ready_path = root/'ready.json'
        if not ready_path.exists() and not (root/'failure.json').exists() and not (root/'maintenance.json').exists():
            return dict(status='WAITING',reason='READY_PENDING')
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
        copies = recovery_counts(active['session_root'],active['config']['purpose'])
        ready = json.loads(ready_path.read_text())
        if certificate(active['config']['hostname'])['certificate_sha256'] != ready['tls']['certificate_sha256']:
            raise ValueError('LIVE_CERTIFICATE_CHANGED')
        return dict(status='PREFLIGHT',ready=ready,boot_id=boot,metadata_unreachable=True,
                    guest_deadline_utc=active['guest_deadline_utc'],native_deadline_utc=active['native_deadline_utc'],
                    active_session=backup['active_session'],session_count=backup['session_count']+copies['session_count'],
                    invitation_count=backup['invitation_count']+copies['invitation_count'])
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
        return disk_dispatch(active,request)
    raise ValueError('UNKNOWN_OWNER_OPERATION')


def main():
    with open(os.devnull,'w') as discard,redirect_stdout(discard),redirect_stderr(discard):
        try:
            request = json.load(sys.stdin)
            with owner_lock():
                result = dispatch(request)
        except Exception as error:
            reason = str(error) if isinstance(error,ValueError) and re.fullmatch('[A-Z][A-Z0-9_]{0,80}',str(error)) else 'GUEST_OPERATION_REJECTED'
            result = dict(status='ERROR',reason=reason,
                          next_action='Verifica status, READY y mantenimiento; conserva los recibos antes de repetir.')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
