"""Private stdin/stdout maintenance API inside the isolated app image."""
import base64
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

from scripts.study_operator.policy import OperatorError, participant_code
from scripts.study_operator.session_data import clean, export_by_code, inventory
from src.ui.components.session_storage import atomic_json


def read_plan(plan):
    result = []
    for row in plan['files']:
        data = Path(row['path']).read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise OperatorError('Archivo cambió tras inventariar. Conserva todas las copias y congela la admisión.')
        result.append(dict(row,content_base64=base64.b64encode(data).decode()))
    # Shared admissions/replacements are retained privately, not deleted as whole files in withdrawal.
    for name in ('_admissions.json','_replacements.json'):
        path = Path(plan['root'])/name
        if path.exists():
            data = path.read_bytes()
            result.append(dict(path=str(path),relative=name,root=plan['root'],kind='shared',bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),content_base64=base64.b64encode(data).decode()))
    return result


def restore_closed(store, export_data, manifest_data, objects, *, allow_replay=False):
    """Restore a verified closed session to a new app store and reproduce its export."""
    from src.ui.components.study_sessions import StudySession

    payload, manifest = json.loads(export_data), json.loads(manifest_data)
    sid = payload['session_id']
    code = participant_code(payload['assignment']['participant_id'])
    digest = hashlib.sha256(export_data).hexdigest()
    if (not re.fullmatch('[a-f0-9]{32}',sid) or payload['stage'] not in ('complete','abandoned')
            or payload['purpose'] != store.purpose or payload['protocol_fingerprint'] != store.protocol['fingerprint']
            or payload.get('protocol_hashes') != store.protocol['hashes']
            or manifest.get('files') != {'full_session.json':digest}
            or objects.get('full_session.json',{}).get('sha256') != digest
            or objects.get('export_manifest.json',{}).get('sha256') != hashlib.sha256(manifest_data).hexdigest()
            or any(not str(row.get('generation','')).isdigit() for row in objects.values())):
        raise OperatorError('Respaldo cerrado o identidad inválidos. No restaures sobre otra configuración.')
    store.freeze()
    if allow_replay and (store.root/sid/'full_session.json').is_file():
        folder = store.root/sid
        admissions = store._read()
        checkpoint = {key:payload[key] for key in StudySession(store,sid,payload['assignment']).data}
        if json.loads(folder.joinpath('study_checkpoint.json').read_text()) != checkpoint:
            raise OperatorError('Checkpoint restaurado alterado. Conserva la copia y revisa antes de repetir.')
        if (folder.joinpath('full_session.json').read_bytes() != export_data
                or folder.joinpath('export_manifest.json').read_bytes() != manifest_data
                or json.loads(folder.joinpath('backup_state.json').read_text()).get('objects') != objects
                or admissions.get('active') is not None
                or any(not row.get('revoked') for row in admissions['invitations'].values())
                or len(list(store.root.glob('*/study_checkpoint.json'))) != 1):
            raise OperatorError('Copia de recuperación previa distinta. Conserva ambos respaldos; no se sobrescribe.')
        bucket = os.environ.pop('CLOUDRAG_BACKUP_BUCKET',None)
        try:
            actual = StudySession.load(store,sid).export().read_bytes()
        finally:
            if bucket is not None:
                os.environ['CLOUDRAG_BACKUP_BUCKET'] = bucket
        if actual != export_data:
            raise OperatorError('Checkpoint restaurado alterado. Conserva la copia y revisa antes de repetir.')
        return dict(status='RESTORED_VERIFIED',session_id=sid,participant_code=code,export_sha256=digest,
                    invitation_revoked=True,replayed=True)
    if store.path.exists() or any(store.root.glob('*/study_checkpoint.json')) or (store.root/sid).exists():
        raise OperatorError('La restauración exige una instancia de app vacía. Conserva su almacenamiento actual.')
    with store.lock:
        checkpoint = {key:payload[key] for key in StudySession(store,sid,payload['assignment']).data}
        folder = store.root/sid
        folder.mkdir(mode=0o700)
        atomic_json(folder/'study_checkpoint.json',checkpoint)
        invitation_hash = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        atomic_json(store.path,dict(invitations={invitation_hash:dict(participant_id=code,session_id=sid,
            assignment=payload['assignment'],issued_at=0,expires_at=0,revoked=True)},active=None))
        for name, data in [('full_session.json',export_data),('export_manifest.json',manifest_data)]:
            with (folder/name).open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        atomic_json(folder/'backup_state.json',dict(status='complete',sha256=digest,objects=objects,
                                                   restored_from_generation_verified=True))
        restored = StudySession.load(store,sid)
        # Re-export verification is local; the original backup receipt remains authoritative.
        bucket = os.environ.pop('CLOUDRAG_BACKUP_BUCKET',None)
        try:
            actual = restored.export().read_bytes()
        finally:
            if bucket is not None:
                os.environ['CLOUDRAG_BACKUP_BUCKET'] = bucket
        if hashlib.sha256(actual).hexdigest() != digest:
            raise OperatorError('La exportación restaurada difiere. Conserva ambos archivos para revisión.')
        return dict(status='RESTORED_VERIFIED',session_id=sid,participant_code=code,export_sha256=digest,
                    invitation_revoked=True)


def dispatch(request, deployment, *, maintenance=False):
    from src.ui.components.study_sessions import StudyStore
    from src.ui.components.study_protocol import verify_draw

    operation = request['operation']
    store = StudyStore(deployment['session_root'],verify_draw(deployment['config_dir']),deployment['purpose'])
    store.check()
    if operation == 'revoke':
        store.revoke(participant_code(request['participant_id']))
        return dict(status='REVOKED')
    if operation == 'backup-check':
        store.check_backups()
        admissions = store._read()
        return dict(status='BACKUPS_CLEAR',session_count=len(list(store.root.glob('*/study_checkpoint.json'))),
                    invitation_count=len(admissions['invitations']),active_session=admissions.get('active') is not None)
    if operation not in {'inventory','download-disk','clean-disk','export','restore'} or not maintenance:
        raise OperatorError('Operación sin mantenimiento verificado. Detén la app antes de inventariar o borrar.')
    if operation == 'inventory':
        plan = inventory(store.root,code=request.get('code'),private_inventory=deployment['private_inventory'],
                         synthetic_only=request.get('synthetic_only',True))
        return dict(status='INVENTORIED',plan=plan)
    if operation == 'download-disk':
        plan = request['plan']
        actual = inventory(store.root,code=plan['code'],private_inventory=deployment['private_inventory'],
                           synthetic_only=plan['synthetic_only'])
        if actual != plan:
            raise OperatorError('Inventario cambió antes de descargar. Mantén cerrada la admisión.')
        return dict(status='DISK_DOWNLOADED_PRIVATE',files=read_plan(plan))
    if operation == 'clean-disk':
        plan = request['plan']
        if Path(plan['root']).resolve() != store.root or plan['private_inventory'] != str(Path(deployment['private_inventory']).resolve()):
            raise OperatorError('Ámbito de limpieza distinto del despliegue. No se borra.')
        return dict(status='DISK_CLEANED',result=clean(plan,request['verified_downloads']))
    if operation == 'export':
        return dict(status='EXPORTED_PRIVATE',export=export_by_code([
            json.loads(path.read_text(encoding='utf-8')) for path in store.root.glob('*/full_session.json')]))
    return restore_closed(store,base64.b64decode(request['full_session_base64'],validate=True),
        base64.b64decode(request['manifest_base64'],validate=True),request['objects'],allow_replay=True)


def main():
    request = json.load(sys.stdin)
    deployment = json.loads(Path('/deployment/deployment.json').read_text(encoding='utf-8'))
    # No library output, error payload, token, question or IP reaches stdout/stderr.
    with open(os.devnull,'w') as discard, redirect_stdout(discard), redirect_stderr(discard):
        try:
            from scripts.cloud_entrypoint import configure

            configure(deployment)
            result = dispatch(request,deployment,maintenance=os.environ.get('CLOUDRAG_MAINTENANCE') == '1')
        except Exception:
            result = dict(status='ERROR',reason='MAINTENANCE_OPERATION_REJECTED',
                          next_action='Conserva recibos, verifica identidad e inventario; no repitas un borrado a ciegas.')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
