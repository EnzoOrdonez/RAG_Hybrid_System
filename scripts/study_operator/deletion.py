"""Two-stage deletion: download and verify every generation before deleting any."""
import hashlib
from pathlib import Path, PurePosixPath

from scripts.study_operator.policy import OperatorError


def inventory(storage, prefix):
    if not prefix or not prefix.endswith('/') or '..' in PurePosixPath(prefix).parts:
        raise OperatorError('Prefijo de borrado inválido. Usa el inventario del periodo de sesiones.')
    objects = storage.objects(prefix, versions=True)
    soft = storage.objects(prefix, soft_deleted=True)
    if soft:
        raise OperatorError('Hay copias soft-deleted. No se puede prometer borrado definitivo; revisa la política del bucket.')
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
    return result


def execute(storage, prefix, local_download, *, dry_run=True, disk_cleanup=None):
    planned = inventory(storage, prefix)
    if dry_run:
        return {'status': 'DRY_RUN', 'scope_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
                'objects': planned, 'deleted': 0}
    if disk_cleanup is None:
        raise OperatorError('Falta el agente de limpieza del disco. No se borra ninguna copia.')
    root = Path(local_download).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise OperatorError('Descarga en un enlace no admitida. Usa un directorio privado real.')
    receipts = []
    for item in planned:
        # Keep each generation distinct even if an unexpected old version exists.
        path = root / item['relative'] / (item['generation'] + '.download')
        if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
            raise OperatorError('Destino de descarga no seguro. No se ha borrado nada.')
        data = storage.read(item['object'], item['generation'])
        if len(data) != item['size']:
            raise OperatorError('Descarga incompleta. Conserva todas las copias y repite la verificación.')
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise OperatorError('Descarga previa distinta. Conserva ambas y revisa el inventario.')
        else:
            with path.open('xb') as stream:
                stream.write(data)
                stream.flush()
                import os
                os.fsync(stream.fileno())
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise OperatorError('SHA-256 local distinto. No se ha borrado ninguna copia remota.')
        receipts.append(dict(item, sha256=digest))
    # Caller holds admission/inference freeze across plan, download, deletion and final listing.
    if inventory(storage, prefix) != planned:
        raise OperatorError('Cambió el inventario durante la descarga. No se borra; vuelve a congelar la admisión.')
    for item in receipts:
        storage.delete(item['object'], item['generation'])
    disk = disk_cleanup(receipts)
    if storage.objects(prefix, versions=True) or storage.objects(prefix, soft_deleted=True) or not disk.get('empty'):
        raise OperatorError('Borrado no completo. Mantén bloqueada la admisión y revisa el recibo de limpieza.')
    return {'status': 'DELETED_VERIFIED', 'scope_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
            'verified_downloads': receipts, 'deleted': len(receipts), 'disk': disk,
            'remote_versions_empty': True, 'remote_soft_deleted_empty': True}
