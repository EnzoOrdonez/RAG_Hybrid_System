"""Two-stage deletion: download and verify every generation before deleting any."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from scripts.study_operator.policy import OperatorError


def inventory(storage, prefix):
    if not prefix or not prefix.endswith('/') or '..' in PurePosixPath(prefix).parts:
        raise OperatorError('Prefijo de borrado inválido. Usa el inventario del periodo de sesiones.')
    storage.verify_zero_retention()
    objects = storage.objects(prefix, versions=True)
    result = []
    for item in objects:
        name = item['name']
        relative = PurePosixPath(name.removeprefix(prefix))
        if not name.startswith(prefix) or relative.is_absolute() or '..' in relative.parts or '\\' in str(relative):
            raise OperatorError('El inventario escapa del ámbito. Conserva los datos y revisa las rutas.')
        if relative.name not in {'full_session.json', 'export_manifest.json'}:
            raise OperatorError('Objeto desconocido en sesiones. Añádelo al inventario explícito antes de borrar.')
        result.append({'object': name, 'generation': str(item['generation']), 'size': int(item['size']),
                       'relative': relative.as_posix()})
    return sorted(result, key=lambda row: (row['object'], row['generation']))


def private_directory(value):
    """Check the supplied path before resolving links, including Windows junctions."""
    path = Path(value).absolute()
    for candidate in (path, *path.parents):
        if candidate.is_symlink() or (candidate.exists() and
                getattr(candidate.lstat(), 'st_file_attributes', 0) & 0x400):
            raise OperatorError('Descarga en un enlace no admitida. Usa un directorio privado real.')
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _save(path, value):
    temporary = path.with_suffix('.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def execute(storage, prefix, local_download, *, dry_run=True, disk_cleanup=None, retain_remote=False):
    planned = inventory(storage, prefix)
    if dry_run:
        return {'status': 'DRY_RUN', 'scope_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
                'objects': planned, 'deleted': 0}
    if disk_cleanup is None:
        raise OperatorError('Falta el agente de limpieza del disco. No se borra ninguna copia.')
    root = private_directory(local_download)
    scope = hashlib.sha256((storage.bucket if hasattr(storage, 'bucket') else 'fixture').encode()
                           + b'\0' + prefix.encode()).hexdigest()
    state_path = root / ('deletion-' + scope + '.json')
    if state_path.is_symlink() or (state_path.exists() and
            getattr(state_path.lstat(), 'st_file_attributes', 0) & 0x400):
        raise OperatorError('Recibo de borrado enlazado. Usa un directorio privado real.')
    previous = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else None
    if previous and previous.get('retain_remote',False) != retain_remote:
        raise OperatorError('Propósito de transacción distinto. Conserva el recibo; no mezcles archivo local y purga.')
    if previous and previous['stage'] == 'COMPLETE':
        if planned:
            raise OperatorError('Hay datos nuevos tras un borrado completo. Usa una transacción nueva; no reutilices su recibo.')
        for item in previous['verified_downloads']:
            path = root/item['relative']/(item['generation']+'.download')
            if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root)
                    or path.stat().st_size != item['size']
                    or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']):
                raise OperatorError('Descarga verificada alterada. Conserva el recibo; no se acredita el borrado.')
        if storage.objects(prefix):
            raise OperatorError('Listado normal no vacío. Conserva el recibo y congela la admisión.')
        disk = disk_cleanup(previous['verified_downloads'])
        if not disk.get('empty'):
            raise OperatorError('Borrado no completo en disco. Mantén bloqueada la admisión.')
        return {'status': 'DELETED_VERIFIED', 'scope_sha256': scope, 'deleted': 0,
                'previous_transaction': str(state_path), 'remote_versions_empty': True,
                'normal_objects_empty': True, 'verified_downloads': previous['verified_downloads'],
                'zero_retention_policy_verified': True, 'policy_verification': storage.policy_verification,
                'disk': disk}
    if previous:
        original = previous['objects']
        if any(item not in original for item in planned):
            raise OperatorError('Cambió el inventario al reanudar. Conserva las descargas y congela la admisión.')
        if previous['stage'] == 'DOWNLOADING' and planned != original:
            raise OperatorError('Faltan objetos antes de verificar todas las descargas. Conserva el recibo y revisa el servidor.')
    else:
        previous = dict(schema_version=1, scope_sha256=scope, stage='DOWNLOADING',
                        objects=planned, verified_downloads=[], deleted_generations=[],retain_remote=retain_remote)
        _save(state_path, previous)
    receipts = []
    for item in previous['objects']:
        # Keep each generation distinct even if an unexpected old version exists.
        path = root / item['relative'] / (item['generation'] + '.download')
        if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise OperatorError('Destino de descarga no seguro. No se ha borrado nada.')
        path.parent.mkdir(parents=True, exist_ok=True)
        recorded = next((row for row in previous['verified_downloads']
                         if row['object'] == item['object'] and row['generation'] == item['generation']), None)
        if recorded:
            if not path.exists() or len(path.read_bytes()) != item['size'] or hashlib.sha256(path.read_bytes()).hexdigest() != recorded['sha256']:
                raise OperatorError('Descarga verificada alterada. No continúes el borrado; recupera el respaldo local.')
            digest = recorded['sha256']
        else:
            data = storage.read(item['object'], item['generation'])
            if len(data) != item['size']:
                raise OperatorError('Descarga incompleta. Conserva todas las copias y repite la verificación.')
            digest = hashlib.sha256(data).hexdigest()
            if path.exists():
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise OperatorError('Descarga previa distinta. Conserva ambas y revisa el inventario.')
            else:
                with path.open('xb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise OperatorError('SHA-256 local distinto. No se ha borrado ninguna copia remota.')
        receipts.append(dict(item, sha256=digest))
        if not recorded:
            previous['verified_downloads'] = receipts
            _save(state_path, previous)
    # Caller holds admission/inference freeze across plan, download, deletion and final listing.
    if inventory(storage, prefix) != planned:
        raise OperatorError('Cambió el inventario durante la descarga. No se borra; vuelve a congelar la admisión.')
    previous['verified_downloads'] = receipts
    if retain_remote:
        previous['stage'] = 'ARCHIVE_DISK_CLEANUP'
        _save(state_path,previous)
        disk = disk_cleanup(receipts)
        if inventory(storage,prefix) != planned or not disk.get('empty'):
            raise OperatorError('Archivo local no completo. Conserva las copias; no se borró ningún objeto del bucket.')
        previous['stage'] = 'COMPLETE_RETAINED_REMOTE'
        previous['disk'] = disk
        _save(state_path,previous)
        return dict(status='ARCHIVED_LOCAL_VERIFIED',verified_downloads=receipts,deleted=0,disk=disk,
                    remote_objects_retained=len(planned),scope_sha256=scope,transaction=str(state_path))
    previous['stage'] = 'DELETING'
    previous['policy_verification_before_delete'] = storage.policy_verification
    _save(state_path, previous)
    for item in planned:
        storage.verify_zero_retention()
        storage.delete(item['object'], item['generation'])
        previous['deleted_generations'].append([item['object'], item['generation']])
        _save(state_path, previous)
    previous['stage'] = 'DISK_CLEANUP'
    _save(state_path, previous)
    disk = disk_cleanup(receipts)
    storage.verify_zero_retention()
    if storage.objects(prefix) or storage.objects(prefix, versions=True) or not disk.get('empty'):
        raise OperatorError('Borrado no completo. Mantén bloqueada la admisión y revisa el recibo de limpieza.')
    previous['stage'] = 'COMPLETE'
    previous['disk'] = disk
    previous['policy_verification_after_delete'] = storage.policy_verification
    _save(state_path, previous)
    return {'status': 'DELETED_VERIFIED', 'scope_sha256': scope, 'transaction': str(state_path),
            'verified_downloads': receipts, 'deleted': len(receipts), 'disk': disk,
            'remote_versions_empty': True, 'normal_objects_empty': True,
            'zero_retention_policy_verified': True, 'policy_verification': storage.policy_verification}
