"""Reviewed deployment layout and argv. No credentials or session content."""
import ipaddress
import re
from pathlib import PurePosixPath

from scripts.study_operator.policy import OperatorError


def checked_config(value):
    if (value.get('schema_version') != 1 or value.get('project') != 'pure-loop-474323-a8'
            or value.get('zone') not in {'us-central1-a', 'us-central1-b', 'us-central1-c'}
            or value.get('machine_type') != 'g2-standard-4'
            or value.get('sessions_bucket') != 'cloudrag-study-i4-103950017681-20261004'
            or value.get('technical_bucket') != 'cloudrag-study-103950017681-20261002'
            or value.get('purpose') not in {'study', 'technical', 'smoke', 'rehearsal', 'pilot'}):
        raise OperatorError('Instalación fuera del ámbito. Revisa el archivo de instalación de iteración 4.')
    for key, pattern in [('image_id', r'sha256:[a-f0-9]{64}'), ('commit', r'[a-f0-9]{40}'),
                         ('model_digest', r'[a-f0-9]{64}'), ('period_id', r'[a-f0-9]{32}')]:
        if not re.fullmatch(pattern, str(value.get(key, ''))):
            raise OperatorError('Identidad de instalación incompleta. Ejecuta la preparación revisada antes de start.')
    ip = str(ipaddress.IPv4Address(value['static_ip']))
    if value.get('hostname') != ip + '.sslip.io':
        raise OperatorError('El nombre TLS no corresponde a la IP reservada. Ejecuta ip-reserve y tls-prepare.')
    for key in ('ollama_image', 'caddy_image'):
        if not re.fullmatch(r'[^\s]+@sha256:[a-f0-9]{64}', value.get(key, '')):
            raise OperatorError('Imagen auxiliar sin digest. Usa el recibo de imágenes verificadas.')
    for key in ('asset_root', 'ollama_models', 'host_code'):
        path = PurePosixPath(value[key])
        if not path.is_absolute() or '..' in path.parts or not str(path).startswith('/srv/cloudrag/'):
            raise OperatorError('Ruta de despliegue no gestionada. Conserva los datos y revisa la instalación.')
    return value


def bind(source, destination, readonly=True):
    return ['--mount', 'type=bind,source=' + str(source) + ',target=' + destination + (',readonly' if readonly else '')]


def app_command(config, boot_root, session_root, name, *, operation='serve', request=None, output=None):
    checked_config(config)
    root = PurePosixPath(boot_root)
    argv = ['docker', 'run', '--rm=false', '--name', name, '--network', 'none', '--gpus', 'all',
            '--user', '10001:10001', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--read-only', '--log-driver', 'none', '--tmpfs', '/tmp:rw,nosuid,nodev,size=512m',
            '-e', 'HOME=/tmp', '-e', 'USER=cloudrag', '-e', 'CLOUDRAG_IMAGE_ID=' + config['image_id'],
            '-e', 'CLOUDRAG_ISOLATED_APP=1', '-e', 'CLOUDRAG_ISOLATED_SERVICE=1',
            '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'CLOUDRAG_DEMO_GPU=1',
            *bind(config['asset_root'] + '/data/models', '/opt/cloudrag/repository/data/models'),
            *bind(config['asset_root'] + '/data/indices', '/opt/cloudrag/repository/data/indices'),
            *bind(root / 'embeddings-initialization', '/opt/cloudrag/repository/data/embeddings'),
            *bind(root / 'config', '/reviewed'), *bind(root / 'meta', '/deployment', operation != 'freeze'),
            *bind(root / 'sockets', '/service'), *bind(root / 'web', '/web', False),
            *bind(PurePosixPath(session_root).parent / 'private-inventory', '/private-inventory', False),
            *bind(session_root, '/sessions', False), '--entrypoint', 'python', config['image_id'],
            '-m', 'scripts.study_operator.app_runtime', operation, '--deployment', '/deployment/' +
            ('deployment-seed.json' if operation == 'freeze' else 'deployment.json'),
            '--ollama-socket', '/service/generation.sock', '--streamlit-socket', '/web/streamlit.sock']
    if request:
        argv += ['--request', request]
    if output:
        argv += ['--output', output]
    return argv


def assert_isolation(observed, expected_image):
    host = observed['HostConfig']
    if (observed['Image'] != expected_image or host.get('NetworkMode') != 'none'
            or host.get('PidMode') not in ('', None) or not host.get('ReadonlyRootfs')
            or set(host.get('CapDrop', [])) != {'ALL'}
            or 'no-new-privileges' not in [x.split('=')[0] for x in host.get('SecurityOpt', [])]
            or host.get('LogConfig', {}).get('Type') != 'none'
            or observed['Config'].get('User') != '10001:10001'
            or 'USER=cloudrag' not in observed['Config'].get('Env', [])):
        raise OperatorError('Contenedor sin aislamiento mínimo. Mantén cerrada la admisión y revisa el arranque.')
    mounts = {m['Destination']: m for m in observed['Mounts']}
    if any(m.get('Destination') == '/var/run/docker.sock' for m in observed['Mounts']):
        raise OperatorError('Docker socket expuesto a la app. Detén el despliegue y corrige los montajes.')
    for path in ('/service', '/deployment', '/reviewed', '/opt/cloudrag/repository/data/models',
                 '/opt/cloudrag/repository/data/indices', '/opt/cloudrag/repository/data/embeddings'):
        if path not in mounts or mounts[path]['RW']:
            raise OperatorError('Montaje de solo lectura cambiado. Detén la admisión y revisa el recibo.')
    cache = PurePosixPath('/opt/cloudrag/repository/data/llm_cache')
    covered = [PurePosixPath(path) for path in (host.get('Tmpfs') or {})]
    covered.extend(PurePosixPath(m['Destination']) for m in observed['Mounts'])
    if any(path.is_relative_to(cache) or cache.is_relative_to(path) for path in covered):
        raise OperatorError('Un montaje oculta archivos versionados de caché. Mantén el checkout de solo lectura, sin superponer esa ruta; cache_enabled permanece False.')
    if '/tmp' not in (host.get('Tmpfs') or {}):
        raise OperatorError('Falta el directorio temporal en memoria. Mantén el checkout de solo lectura y revisa tmpfs de /tmp.')
    return {'container_isolation_verified': True}


def caddyfile(hostname):
    ipaddress.IPv4Address(hostname.removesuffix('.sslip.io'))
    if not hostname.endswith('.sslip.io'):
        raise ValueError('Fixed deployment hostname required')
    return '''{
    admin off
    persist_config off
    auto_https disable_redirects
    log default {
        output discard
    }
}
HOST_NAME {
    log {
        output discard
    }
    tls {
        issuer acme {
            dir https://acme-v02.api.letsencrypt.org/directory
            email 20221789@aloe.ulima.edu.pe
            disable_http_challenge
        }
    }
    reverse_proxy unix//web/streamlit.sock {
        header_up -X-Forwarded-For
        header_up -X-Real-IP
        header_up -Forwarded
        header_up -User-Agent
    }
}
'''.replace('HOST_NAME', hostname)
