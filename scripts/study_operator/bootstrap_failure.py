"""Immutable failed-boot metadata, readable by the owner after native STOP."""
import hashlib
import json
from pathlib import Path
import re

REASONS = {'HOST_FAILED','HOST_COMMAND_FAILED','SESSION_PURPOSE_CHANGED','OLLAMA_VERSION_CHANGED',
           'OLLAMA_START_TIMEOUT','CLOUD_IDENTITY_CHANGED','IMAGE_CHANGED','PREREGISTRATION_ANCHOR_CHANGED',
           'PRIVATE_AGENT_FAILED','PRIVATE_AGENT_TIMEOUT','FREEZE_IMAGE_CHANGED','METADATA_ISOLATION_NOT_VERIFIED',
           'APP_NOT_HEALTHY','READY_DEADLINE_900S','PRIVATE_AGENT_EXITED'}
ERROR_TYPES = {'UnknownError','ValueError','OperatorError','OSError','FileNotFoundError','PermissionError',
               'TimeoutExpired','TimeoutError','KeyError','AssertionError','RuntimeError','TypeError'}
OPERATIONS = {'ARM_SHUTDOWN','LIST_CONTAINERS','STOP_CONTAINER','VERIFY_IMAGE','VERIFY_CONTAINER',
              'FREEZE_RUNTIME_IDENTITY','START_ISOLATED_APP','START_OLLAMA','TLS_CONFIGURATION',
              'CONTAINER_CHECK','HOST_COMMAND'}


def operation(argv):
    prefix = tuple(argv[:2])
    known = {('docker','ps'):'LIST_CONTAINERS',('docker','stop'):'STOP_CONTAINER',
             ('docker','image'):'VERIFY_IMAGE',('docker','inspect'):'VERIFY_CONTAINER',
             ('docker','exec'):'CONTAINER_CHECK'}
    if argv[0] == 'systemd-run':
        return 'ARM_SHUTDOWN'
    if prefix == ('docker','run'):
        if 'scripts.study_operator.app_runtime' in argv:
            return 'FREEZE_RUNTIME_IDENTITY' if 'freeze' in argv else 'START_ISOLATED_APP'
        if 'OLLAMA_HOST=127.0.0.1:11434' in argv:
            return 'START_OLLAMA'
        if 'caddy' in argv:
            return 'TLS_CONFIGURATION'
    return known.get(prefix,'HOST_COMMAND')

def failure_summary(root, *, instance_id, image_id, commit):
    root = Path(root)
    failure = json.loads((root/'failure.json').read_text(encoding='utf-8'))
    commands = []
    for path in sorted(root.glob('command-*.json')):
        if path.is_symlink() or not re.fullmatch(r'command-\d{4}.json', path.name):
            raise ValueError('UNSAFE_TECHNICAL_RECEIPT')
        row = json.loads(path.read_text(encoding='utf-8'))
        commands.append(dict(sequence=int(path.stem.split('-')[-1]),exit_code=row['exit_code'],
            duration_s=row['duration_s'],started_utc=row['started_utc'],
            operation=operation(row['command']),
            argv_sha256=hashlib.sha256(json.dumps(row['command'],sort_keys=True).encode()).hexdigest()))
    value = dict(schema_version=1,status='FAILED_BOOT_TECHNICAL_METADATA',boot_id=root.name,
        instance_id=str(instance_id),image_id=image_id,commit=commit,
        failure=dict(reason=failure.get('reason') if failure.get('reason') in REASONS else 'HOST_FAILED',
                     error_type=failure.get('error_type') if failure.get('error_type') in ERROR_TYPES else 'UnknownError'),
        commands=commands,session_content_excluded=True)
    validate_summary(value,instance_id=instance_id,image_id=image_id,commit=commit)
    return value


def validate_summary(value, *, instance_id, image_id, commit):
    keys={'schema_version','status','boot_id','instance_id','image_id','commit','failure','commands','session_content_excluded'}
    if (set(value) != keys or value['schema_version'] != 1
            or value['status'] != 'FAILED_BOOT_TECHNICAL_METADATA'
            or value['instance_id'] != str(instance_id) or value['image_id'] != image_id or value['commit'] != commit
            or value['session_content_excluded'] is not True
            or not re.fullmatch(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}',value['boot_id'])
            or set(value['failure']) != {'reason','error_type'}):
        raise ValueError('FAILED_BOOT_IDENTITY_OR_SCHEMA_CHANGED')
    if value['failure']['reason'] not in REASONS or value['failure']['error_type'] not in ERROR_TYPES:
        raise ValueError('UNSAFE_FAILURE_TEXT')
    for row in value['commands']:
        if (set(row) != {'sequence','exit_code','duration_s','started_utc','argv_sha256','operation'}
                or row['operation'] not in OPERATIONS
                or not isinstance(row['exit_code'],int) or not isinstance(row['sequence'],int)
                or not isinstance(row['duration_s'],(int,float)) or row['duration_s'] < 0
                or not re.fullmatch(r'[a-f0-9]{64}',row['argv_sha256'])
                or not re.fullmatch(r'[0-9T:+.Z-]{20,40}',row['started_utc'])):
            raise ValueError('UNSAFE_COMMAND_METADATA')
    return value
