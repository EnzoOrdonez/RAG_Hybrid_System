"""Owner CLI operations with receipts; secrets/private downloads stay in memory."""
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time

from scripts.study_operator.policy import OperatorError, ReadyPending
from scripts.study_operator.service_gateway import save_state
from scripts.study_operator.region_scope import US_L4_ZONES


def error_categories(stderr):
    """Keep allowlisted technical classifications, never private error messages."""
    names = ('ConnectionError', 'ConnectionResetError', 'RemoteDisconnected', 'SSLError',
             'ReadTimeout', 'ConnectTimeout', 'ProxyError', 'TimeoutError', 'JSONDecodeError',
             'INVALID_ARGUMENT', 'PERMISSION_DENIED', 'UNAUTHENTICATED', 'QUOTA_EXCEEDED',
             'RESOURCE_EXHAUSTED', 'NOT_FOUND', 'UNAVAILABLE', 'INTERNAL')
    categories = [name for name in names if re.search(rb'\b'+name.encode()+rb'\b', stderr)]
    if re.search(rb'\b(?:unrecognized arguments|Invalid choice|expected one argument)\b', stderr, re.I):
        categories.append('SDK_ARGUMENT_ERROR')
    http_statuses = sorted({int(value) for value in re.findall(rb'\bHTTPError[ :]+([45][0-9]{2})\b', stderr)})
    return dict(error_categories=categories, http_error_statuses=http_statuses,
                private_error_message_not_persisted=True)


class Cloud:
    def __init__(self, sdk, project, run_root, *, invoke=subprocess.run):
        if project != 'pure-loop-474323-a8':
            raise OperatorError('Proyecto fuera del ámbito. Usa la instalación revisada para pure-loop-474323-a8.')
        self.sdk, self.project, self.root, self.invoke = sdk, project, Path(run_root), invoke
        self.root.mkdir(parents=True, exist_ok=True)
        self.sequence = max((int(p.name.split('-')[0]) for p in self.root.glob('*-intent.json')
                             if p.name.split('-')[0].isdigit()), default=0)

    def command(self, arguments, *, private_output=True, input_data=None, timeout=180, json_output=True):
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
                from scripts.study_operator.windows_ssh import api_host_key_flags, native_argv, native_run, sdk_argv

                zones = [arg for arg in arguments if arg.startswith('--zone=')]
                if len(zones) != 1 or zones[0].split('=', 1)[1] not in US_L4_ZONES:
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
                    json_output=False, private_output=True, timeout=min(timeout, 90))
                dry = self.command([*pinned, '--dry-run'],
                    json_output=False, private_output=True,
                    timeout=max(1, min(60, timeout-(time.monotonic()-began))))
                actual = sdk_argv(dry, Path(self.sdk).parent/'sdk/plink.exe')
                expected_pins = [arg.removeprefix('--ssh-flag=') for arg in pinned
                                 if arg.startswith('--ssh-flag=SHA256:')]
                actual_pins = [actual[index + 1] for index, arg in enumerate(actual[:-1]) if arg == '-hostkey']
                if actual_pins != expected_pins:
                    raise ValueError('SDK discarded authenticated host key pins')
                vm = self.command(['compute', 'instances', 'describe', arguments[2], zones[0]],
                    private_output=True, timeout=max(1, min(60, timeout-(time.monotonic()-began))))
                if (vm['name'] != arguments[2]
                        or vm['zone'].split('/')[-1] != zones[0].split('=', 1)[1]):
                    raise ValueError('Authenticated VM identity differs')
                actual = native_argv(actual, keys, sdk=self.sdk,
                    target=dict(name=vm['name'], id=str(vm['id']), zone=vm['zone'].split('/')[-1]),
                    known_hosts=Path(str(stem)+'-public-hostkeys'))
                transport = 'WINDOWS_OPENSSH_IAP_PINNED'
            remaining = max(1, timeout-(time.monotonic()-began))
            invoke = native_run if private_rpc and self.invoke is subprocess.run else self.invoke
            result = invoke(actual, input=input_data, capture_output=True, timeout=remaining, env=environment)
        except subprocess.TimeoutExpired as error:
            # A timed-out public tool can still explain the transport failure.
            # Private RPC/token output must stay in memory even on timeout.
            if not private_output:
                for suffix, partial in (('stdout', error.stdout), ('stderr', error.stderr)):
                    partial = partial or b''
                    if isinstance(partial, str):
                        partial = partial.encode('utf-8')
                    Path(str(stem) + '.' + suffix).write_bytes(partial)
            save_state(str(stem) + '-receipt.json', dict(command=argv, exit_code=124, started_utc=before,
                ended_utc=datetime.now(timezone.utc).isoformat(), duration_s=time.monotonic()-began,
                partial_output_policy='PRIVATE_NOT_PERSISTED' if private_output else 'PUBLIC_PRESERVED'))
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
        if result.returncode:
            receipt.update(error_categories(stderr))
        if (result.returncode and len(arguments) > 2 and arguments[:2] == ['compute', 'instances']
                and arguments[2] in {'start', 'create'}):
            match = re.search(rb'\b(ZONE_RESOURCE_POOL_EXHAUSTED(?:_WITH_DETAILS)?)\b', stderr)
            if match:
                receipt['cloud_error_code'] = match[0].decode('ascii')
                # Capacity errors are technical evidence, but an SDK footer
                # containing identity/credentials must still remain in memory.
                if (re.fullmatch(rb'[\t\r\n\x20-\x7e]*', stderr)
                        and not re.search(rb'@|\b(?:Bearer|token|Authorization|ya29)\b|\b\d{1,3}(?:\.\d{1,3}){3}\b', stderr, re.I)):
                    receipt['capacity_error_text'] = stderr.decode('utf-8', errors='strict')
        save_state(str(stem) + '-receipt.json', receipt)
        if result.returncode:
            if (arguments[:3] == ['compute', 'instances', 'get-guest-attributes']
                    and '--query-path=hostkeys/' in arguments
                    and b'HTTPError 404' in stderr and b"'hostkeys/'" in stderr
                    and b'Guest Attribute' in stderr):
                raise ReadyPending('Aún faltan claves públicas del invitado. Espera y repite preflight dentro de 15 minutos; no aceptes una clave desconocida.')
            # Capacity is a resource error, not a measured gate failure.
            if b'ZONE_RESOURCE_POOL_EXHAUSTED' in stderr or b'does not have enough resources' in stderr:
                raise OperatorError('ZONE_RESOURCE_POOL_EXHAUSTED: respeta las rondas de us-central1 y el orden de regiones medido; usa el runbook de contingencia, identidad y smoke antes de una sesión.')
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
    if zone not in US_L4_ZONES:
        raise OperatorError('Zona fuera del ámbito L4 verificado de EE.UU. Revisa el catálogo antes de encender.')
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
        gpu = bool(item.get('guestAccelerators')) or item.get('machineType', '').split('/')[-1].startswith('g2-')
        if str(item.get('id')) != str(selected_id) and gpu and item.get('status') != 'TERMINATED':
            raise OperatorError('Hay otra VM con GPU activa. Detén y verifica esa VM antes de encender otra.')


def readiness(receipt, *, image_id, url, boot_id):
    if not isinstance(receipt, dict) or receipt.get('status') != 'READY':
        raise OperatorError('El invitado aún no está READY. Espera dentro del límite de 15 minutos y repite preflight; no repitas start.')
    if (receipt.get('image_id') != image_id or receipt.get('url') != url or receipt.get('boot_id') != boot_id
            or not receipt.get('identity_verified') or not receipt.get('metadata_unreachable')
            or not receipt.get('backup_clear') or not receipt.get('tls_verified')):
        raise OperatorError('READY no acredita identidad, aislamiento, respaldo o TLS. Mantén cerrada la admisión y revisa el recibo del invitado.')
    return receipt
