"""Enumerate every mutable app copy in a period, including recovery instances."""
import hashlib
import json
from pathlib import Path
import re

from scripts.study_operator.policy import OperatorError


def managed_stores(session_root, purpose):
    root = Path(session_root)
    if root.is_symlink() or root.name != 'sessions' or not root.is_dir():
        raise OperatorError('Raíz activa no gestionada. Conserva todas las copias y revisa el inventario.')
    stores = [('primary', root)]
    recoveries = root.parent/'recoveries'
    if recoveries.exists():
        if recoveries.is_symlink() or not recoveries.is_dir():
            raise OperatorError('Directorio de recuperación enlazado. No se inventaría ni se borra.')
        for directory in sorted(recoveries.iterdir()):
            if directory.is_symlink() or not directory.is_dir() or not re.fullmatch('[a-f0-9]{64}',directory.name):
                raise OperatorError('Copia de recuperación desconocida. Conserva el servidor para revisión.')
            if any(path.name not in {'sessions','private-inventory'} or path.is_symlink() for path in directory.iterdir()):
                raise OperatorError('Archivos de recuperación fuera del inventario. No se borra.')
            stores.append((directory.name,directory/'sessions'))
    for _, folder in stores:
        marker = folder/'_i4_root.json'
        if (folder.is_symlink() or not folder.is_dir() or not marker.is_file() or marker.is_symlink()
                or json.loads(marker.read_text(encoding='utf-8')) != dict(schema_version=1,purpose=purpose)):
            raise OperatorError('Copia sin identidad de propósito. No se borra ni se abandona al conmutar.')
    return dict(stores)


def recovery_key(request):
    import base64

    return hashlib.sha256(base64.b64decode(request['full_session_base64'],validate=True)).hexdigest()


def collection(plans):
    """Store identity disambiguates identical in-container paths across copies."""
    return dict(schema_version=2,stores=plans,
                files=[dict(row,store_id=item['store_id']) for item in plans for row in item['plan']['files']],
                session_count=sum(len(item['plan']['session_codes']) for item in plans))


def scoped_downloads(rows, store_id):
    return [{key:value for key,value in row.items() if key != 'store_id'}
            for row in rows if row.get('store_id') == store_id]


def recovery_counts(session_root, purpose):
    sessions = invitations = 0
    for key, root in managed_stores(session_root,purpose).items():
        if key == 'primary':
            continue
        admissions = root/'_admissions.json'
        value = json.loads(admissions.read_text()) if admissions.exists() else dict(invitations={},active=None)
        if value.get('active') is not None or any(not row.get('revoked') for row in value['invitations'].values()):
            raise OperatorError('Recuperación con admisión activa. Mantén cerrada la admisión y revisa sus copias.')
        invitations += len(value['invitations'])
        for checkpoint in root.glob('*/study_checkpoint.json'):
            payload = json.loads(checkpoint.read_text())
            receipt = json.loads((checkpoint.parent/'backup_state.json').read_text())
            full = (checkpoint.parent/'full_session.json').read_bytes()
            if (payload.get('stage') not in {'complete','abandoned'} or receipt.get('status') != 'complete'
                    or hashlib.sha256(full).hexdigest() != receipt.get('sha256')):
                raise OperatorError('Recuperación incompleta o alterada. Conserva todas sus copias y revisa el respaldo.')
            sessions += 1
    return dict(session_count=sessions,invitation_count=invitations)


def snapshot_safe(periods_root):
    """Never make an immutable snapshot containing iteration4 study data."""
    root = Path(periods_root)
    if not root.exists():
        return dict(status='ALL_I4_PERIODS_EMPTY',period_count=0)
    if root.is_symlink():
        raise OperatorError('Raíz de periodos enlazada. No prepares una instantánea.')
    count = 0
    allowed = {'_i4_root.json','_study_protocol.json','_admissions.lock','_inference.lock','_maintenance.json'}
    for period in root.iterdir():
        if period.is_symlink() or not period.is_dir() or not re.fullmatch('[a-f0-9]{32}',period.name):
            raise OperatorError('Periodo desconocido. No prepares una instantánea con datos sin inventariar.')
        purpose = json.loads((period/'sessions'/'_i4_root.json').read_text())['purpose']
        for _, folder in managed_stores(period/'sessions',purpose).items():
            count += 1
            for item in folder.iterdir():
                if item.name == '_preparation' and item.is_dir() and not item.is_symlink() and not any(item.iterdir()):
                    continue
                if item.is_symlink() or item.is_dir() or item.name not in allowed:
                    raise OperatorError('Persisten datos o invitaciones en un periodo. Purga y verifica antes de preparar contingencia.')
            private = folder.parent/'private-inventory'
            if private.exists() and (private.is_symlink() or any(private.iterdir())):
                raise OperatorError('Persisten inventarios privados. Purga todas las copias antes de preparar contingencia.')
    return dict(status='ALL_I4_PERIODS_EMPTY',app_store_count=count)
