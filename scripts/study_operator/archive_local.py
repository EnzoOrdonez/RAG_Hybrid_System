"""Archive closed, generation-verified disk copies while retaining canonical GCS backups."""
import hashlib
import json
from pathlib import Path

from scripts.study_operator.policy import OperatorError


def verified_closed_copies(disk_downloads, remote_downloads):
    available = {(row['object'],str(row['generation']),row['sha256']) for row in remote_downloads}
    rows = {(row['store_id'],row['relative']):row for row in disk_downloads if row['kind'] in {'session','shared'}}
    count = 0
    for (store_id,relative), row in rows.items():
        if relative == '_admissions.json':
            if json.loads(Path(row['local_path']).read_bytes()).get('active') is not None:
                raise OperatorError('Una sesión sigue activa. Respalda su cierre antes de archivar el disco.')
        if not relative.endswith('/study_checkpoint.json'):
            continue
        checkpoint = json.loads(Path(row['local_path']).read_bytes())
        if checkpoint.get('stage') not in {'complete','abandoned'}:
            raise OperatorError('Una sesión no está cerrada. No se elimina su copia de disco.')
        sid = relative.split('/')[0]
        receipt_row = rows.get((store_id,sid+'/backup_state.json'))
        if not receipt_row:
            raise OperatorError('Falta un respaldo de generación verificada. No archives el disco.')
        receipt = json.loads(Path(receipt_row['local_path']).read_bytes())
        if receipt.get('status') != 'complete':
            raise OperatorError('Respaldo pendiente. No archives el disco.')
        for name in ('full_session.json','export_manifest.json'):
            local = rows.get((store_id,sid+'/'+name))
            remote = receipt.get('objects',{}).get(name,{})
            if (not local or hashlib.sha256(Path(local['local_path']).read_bytes()).hexdigest() != remote.get('sha256')
                    or (remote.get('object'),str(remote.get('generation')),remote.get('sha256')) not in available):
                raise OperatorError('La copia cerrada no coincide con su generación en GCS. Conserva ambas; no archives.')
        count += 1
    return dict(closed_backups_verified=count)
