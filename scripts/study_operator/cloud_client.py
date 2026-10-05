"""Owner CLI operations with receipts; secrets/private downloads stay in memory."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.study_operator.policy import OperatorError, ReadyPending
from scripts.study_operator.service_gateway import save_state


class Cloud:
    def __init__(self, sdk, project, run_root, *, invoke=subprocess.run):
        if project != 'pure-loop-474323-a8':
            raise OperatorError('Proyecto fuera del ámbito. Usa la instalación revisada para pure-loop-474323-a8.')
        self.sdk, self.project, self.root, self.invoke = sdk, project, Path(run_root), invoke
        self.root.mkdir(parents=True, exist_ok=True)
        self.sequence = max((int(p.name.split('-')[0]) for p in self.root.glob('*-intent.json')
                             if p.name.split('-')[0].isdigit()), default=0)

    def command(self, arguments, *, private_output=False, input_data=None, timeout=180, json_output=True):
        self.sequence += 1
        stem = self.root / f'{self.sequence:04d}'
        argv = [self.sdk, *arguments, '--project=' + self.project, '--quiet']
        if json_output:
            argv.append('--format=json')
        environment = dict(os.environ, CLOUDSDK_CORE_DISABLE_FILE_LOGGING='1', CLOUDSDK_CORE_DISABLE_PROMPTS='1',
                           CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED='False', PYTHONUTF8='1',
                           CLOUDSDK_SSH_PUTTY_FORCE_CONNECT='False')
        before, began = datetime.now(timezone.utc).isoformat(), time.monotonic()
        save_state(str(stem) + '-intent.json', dict(command=argv, started_utc=before,
            output_policy='MEMORY_ONLY_PRIVATE' if private_output else 'TECHNICAL_JSON',
            stdin_policy='MEMORY_ONLY_NOT_LOGGED', status='STARTED'))
        transport = 'SDK'
        private_rpc = os.name == 'nt' and input_data is not None and arguments[:2] == ['compute', 'ssh']
        try:
            actual = argv
            if private_rpc:
                from scripts.study_operator.windows_ssh import api_host_key_flags, sdk_argv

                zones = [arg for arg in arguments if arg.startswith('--zone=')]
                if len(zones) != 1 or zones[0].split('=', 1)[1] not in {'us-central1-a', 'us-central1-b', 'us-central1-c'}:
                    raise ValueError('Managed SSH zone required')
                keys = self.command(['compute', 'instances', 'get-guest-attributes', arguments[2],
                    zones[0], '--query-path=hostkeys/'], timeout=min(timeout, 60))
                if isinstance(keys, list) and not keys:
                    raise ReadyPending('Aún faltan claves públicas del invitado. Espera y repite preflight dentro de 15 minutos; no aceptes una clave desconocida.')
                # Keep -batch first after the proxy field for the exact SDK
                # display inverse. PuTTY ignores SDK OpenSSH known_hosts options.
                if any(arg.startswith('--ssh-flag=') for arg in arguments):
                    raise ValueError('Managed RPC SSH flags cannot be overridden')
                pinned = [*arguments, '--ssh-flag=-batch', *api_host_key_flags(keys)]

                # dry-run returns before renewing the expiring metadata key.
                # Authenticate a fixed no-op first, with no private stdin and
                # no automatic host-key acceptance, then send the real bytes.
                authenticate = [arg for arg in pinned if not arg.startswith('--command=')]
                self.command([*authenticate, '--command=true'],
                    json_output=False, timeout=min(timeout, 90))
                dry = self.command([*pinned, '--dry-run'],
                    json_output=False, timeout=max(1, min(60, timeout-(time.monotonic()-began))))
                actual = sdk_argv(dry, Path(self.sdk).parent/'sdk/plink.exe')
                expected_pins = [arg.removeprefix('--ssh-flag=') for arg in pinned
                                 if arg.startswith('--ssh-flag=SHA256:')]
                actual_pins = [actual[index + 1] for index, arg in enumerate(actual[:-1]) if arg == '-hostkey']
                if actual_pins != expected_pins:
                    raise ValueError('SDK discarded authenticated host key pins')
                transport = 'VALIDATED_SDK_DRYRUN_PLINK'
            remaining = max(1, timeout-(time.monotonic()-began))
            result = self.invoke(actual, input=input_data, capture_output=True, timeout=remaining, env=environment)
        except subprocess.TimeoutExpired:
            save_state(str(stem) + '-receipt.json', dict(command=argv, exit_code=124, started_utc=before,
                ended_utc=datetime.now(timezone.utc).isoformat(), duration_s=time.monotonic()-began))
            raise OperatorError('Venció el límite de la herramienta. Conserva el recibo y verifica status antes de reintentar.') from None
        except ReadyPending:
            save_state(str(stem) + '-receipt.json', dict(command=argv, exit_code=1, started_utc=before,
                ended_utc=datetime.now(timezone.utc).isoformat(), duration_s=time.monotonic()-began,
                transport=transport, reason='PUBLIC_HOST_KEYS_PENDING'))
            raise
        except (OperatorError, ValueError, OSError, KeyError, TypeError):
            save_state(str(stem) + '-receipt.json', dict(command=argv, exit_code=1, started_utc=before,
                ended_utc=datetime.now(timezone.utc).isoformat(), duration_s=time.monotonic()-began,
                transport=transport, reason='SSH_TRANSPORT_REJECTED'))
            if private_rpc:
                raise OperatorError('El transporte SSH no pudo verificarse. Conserva runs, revisa la clave del host y status; no repitas start ni aceptes una clave cambiada.') from None
            raise OperatorError('No se pudo ejecutar Google Cloud. Conserva runs y verifica la instalación del SDK antes de repetir; no crees recursos a ciegas.') from None
        stdout, stderr = result.stdout, result.stderr
        if isinstance(stdout, str):
            stdout = stdout.encode('utf-8')
        if isinstance(stderr, str):
            stderr = stderr.encode('utf-8')
        if not private_output:
            Path(str(stem) + '.stdout').write_bytes(stdout)
            Path(str(stem) + '.stderr').write_bytes(stderr)
        receipt = dict(command=argv, started_utc=before, ended_utc=datetime.now(timezone.utc).isoformat(),
            duration_s=time.monotonic()-began, exit_code=result.returncode,
            stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_sha256=hashlib.sha256(stderr).hexdigest(),
            private_output_not_persisted=private_output, transport=transport)
        save_state(str(stem) + '-receipt.json', receipt)
        if result.returncode:
            if (arguments[:3] == ['compute', 'instances', 'get-guest-attributes']
                    and '--query-path=hostkeys/' in arguments
                    and b'HTTPError 404' in stderr and b"'hostkeys/'" in stderr
                    and b'Guest Attribute' in stderr):
                raise ReadyPending('Aún faltan claves públicas del invitado. Espera y repite preflight dentro de 15 minutos; no aceptes una clave desconocida.')
            # Capacity is a resource error, not a measured gate failure.
            if b'ZONE_RESOURCE_POOL_EXHAUSTED' in stderr or b'does not have enough resources' in stderr:
                raise OperatorError('ZONE_RESOURCE_POOL_EXHAUSTED: ejecuta failover a us-central1-b o us-central1-c; si ambas fallan, reprograma según el runbook.')
            raise OperatorError('Google Cloud rechazó la operación. Revisa su recibo, verifica status y corrige la causa antes de repetir.')
        if json_output:
            try:
                return json.loads(stdout.decode('utf-8-sig')) if stdout.strip() else None
            except (ValueError, UnicodeError):
                raise OperatorError('La salida no coincide con el contrato JSON. Conserva el recibo; no repitas recursos a ciegas.') from None
        return stdout

    def owner_token(self):
        value = self.command(['auth', 'print-access-token'], private_output=True, json_output=False, timeout=60)
        token = value.decode().strip()
        if not token or '\n' in token:
            raise OperatorError('No hay sesión válida de gcloud. Enzo debe revisar su autenticación; no se crean credenciales.')
        return token


def checked_vm(observed, *, name, instance_id, zone):
    if zone not in {'us-central1-a', 'us-central1-b', 'us-central1-c'}:
        raise OperatorError('Zona fuera del ámbito. Solo se admite us-central1.')
    if (observed.get('name') != name or str(observed.get('id')) != str(instance_id)
            or observed.get('zone', '').split('/')[-1] != zone
            or observed.get('machineType', '').split('/')[-1] != 'g2-standard-4'
            or not observed.get('deletionProtection')
            or not observed.get('disks') or observed['disks'][0].get('autoDelete') is not False
            or observed.get('scheduling', {}).get('instanceTerminationAction') != 'STOP'
            or observed.get('scheduling', {}).get('maxRunDuration', {}).get('seconds') != '10800'):
        raise OperatorError('VM, disco o límite nativo distintos de la instalación. No se enciende; revisa identidad y protecciones.')
    return observed


def no_other_gpu(instances, *, selected_id):
    for item in instances:
        if str(item.get('id')) != str(selected_id) and item.get('guestAccelerators') and item.get('status') != 'TERMINATED':
            raise OperatorError('Hay otra VM con GPU activa. Detén y verifica esa VM antes de encender otra.')


def readiness(receipt, *, image_id, url, boot_id):
    if not isinstance(receipt, dict) or receipt.get('status') != 'READY':
        raise OperatorError('El invitado aún no está READY. Espera dentro del límite de 15 minutos y repite preflight; no repitas start.')
    if (receipt.get('image_id') != image_id or receipt.get('url') != url or receipt.get('boot_id') != boot_id
            or not receipt.get('identity_verified') or not receipt.get('metadata_unreachable')
            or not receipt.get('backup_clear') or not receipt.get('tls_verified')):
        raise OperatorError('READY no acredita identidad, aislamiento, respaldo o TLS. Mantén cerrada la admisión y revisa el recibo del invitado.')
    return receipt
