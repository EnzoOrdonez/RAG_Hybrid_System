"""Explicit inventory and verified cleanup of iteration4 session roots only."""
import hashlib
import json
from pathlib import Path
import re

from filelock import FileLock

from scripts.study_operator.deletion import private_directory
from scripts.study_operator.policy import OperatorError, participant_code
from src.ui.components.session_storage import atomic_json


def _root(path, *, synthetic_only):
    root = private_directory(path)
    marker = root / '_i4_root.json'
    if not marker.is_file():
        raise OperatorError('Raíz sin inventario de iteración4. No se toca almacenamiento heredado.')
    value = json.loads(marker.read_text(encoding='utf-8'))
    if value.get('schema_version') != 1 or value.get('purpose') not in {'study', 'smoke', 'rehearsal', 'technical', 'pilot'}:
        raise OperatorError('Inventario de raíz inválido. Conserva los datos y revisa la instalación.')
    if synthetic_only and value['purpose'] == 'study':
        raise OperatorError('Esta ejecución solo autoriza borrado sintético; study no se toca.')
    return root


def _file(path, root, kind):
    if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise OperatorError('Ruta de datos enlazada o fuera del inventario. No se borra.')
    content = path.read_bytes()
    return dict(path=str(path), relative=path.relative_to(root).as_posix(),
                root=str(root), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), kind=kind)


def inventory(path, *, code=None, private_inventory=None, synthetic_only=True, _known_sessions=None):
    root = _root(path, synthetic_only=synthetic_only)
    if code is not None:
        participant_code(code)
    admissions_path = root / '_admissions.json'
    admissions = json.loads(admissions_path.read_text(encoding='utf-8')) if admissions_path.exists() else dict(invitations={}, active=None)
    selected = {key: row for key, row in admissions['invitations'].items() if code is None or row['participant_id'] == code}
    sessions = {row['session_id']: row['participant_id'] for row in selected.values()}
    files = []
    for directory in root.iterdir():
        if directory.is_symlink():
            raise OperatorError('Enlace en raíz de sesiones. No se borra.')
        if not directory.is_dir():
            if directory.name not in {'_i4_root.json', '_study_protocol.json', '_admissions.json',
                                     '_admissions.lock', '_inference.lock', '_replacements.json', '_maintenance.json'} and not re.fullmatch(r'_cleanup-[a-f0-9]{64}\.json', directory.name):
                raise OperatorError('Archivo raíz fuera del inventario. Conserva los datos y declara su finalidad.')
            continue
        if directory.name == '_preparation':
            continue
        if not re.fullmatch('[a-f0-9]{32}', directory.name):
            raise OperatorError('Directorio de sesión desconocido. Declara sus rutas antes de borrar.')
        checkpoint = directory / 'study_checkpoint.json'
        export = directory / 'full_session.json'
        identified = checkpoint if checkpoint.exists() else export if export.exists() else None
        if not identified and directory.name in (_known_sessions or {}):
            observed_code = _known_sessions[directory.name]
            if code is None or code == observed_code:
                sessions[directory.name] = observed_code
        if identified:
            payload = json.loads(identified.read_text(encoding='utf-8'))
            observed_code = participant_code(payload['assignment']['participant_id'])
            if payload['session_id'] != directory.name:
                raise OperatorError('ID de sesión distinto del directorio. Conserva el inventario.')
            if code is None or code == observed_code:
                if directory.name in sessions and sessions[directory.name] != observed_code:
                    raise OperatorError('Código de admisión distinto del checkpoint. No se borra.')
                sessions[directory.name] = observed_code
        if directory.name not in sessions:
            if code is None:
                raise OperatorError('Sesión sin código identificable. Conserva los datos para revisión.')
            continue
        for file in directory.iterdir():
            if file.name not in {'study_checkpoint.json', 'full_session.json', 'export_manifest.json',
                                  'backup_state.json', 'study_checkpoint.json.lock'} and not file.name.startswith('.pending-'):
                raise OperatorError('Archivo de sesión desconocido. Amplía el inventario antes de borrar.')
            files.append(_file(file, root, 'session'))
    if synthetic_only and any(int(value[1:]) < 900 for value in sessions.values()):
        raise OperatorError('Código no sintético. Esta iteración no autoriza borrarlo.')
    preparation = root / '_preparation'
    if preparation.exists():
        for file in preparation.iterdir():
            if code is None or json.loads(file.read_text(encoding='utf-8')).get('scope') in sessions:
                files.append(_file(file, root, 'preparation'))
    if private_inventory:
        private = private_directory(private_inventory)
        for sid in sessions:
            for name in (sid + '.json', sid + '.pending'):
                file = private / name
                if file.exists():
                    files.append(_file(file, private, 'private_inventory'))
    replacements = root / '_replacements.json'
    selected_replacements = {}
    if replacements.exists():
        rows = json.loads(replacements.read_text(encoding='utf-8'))
        selected_replacements = {key: value for key, value in rows.items()
                                 if code is None or key == code or value.get('primary') == code}
    return dict(schema_version=1, root=str(root), code=code, synthetic_only=synthetic_only,
                session_codes=sessions, invitations=selected, replacements=selected_replacements,
                files=sorted(files, key=lambda row: row['path']),
                admissions_sha256=hashlib.sha256(admissions_path.read_bytes()).hexdigest() if admissions_path.exists() else None,
                replacements_sha256=hashlib.sha256(replacements.read_bytes()).hexdigest() if replacements.exists() else None,
                private_inventory=str(Path(private_inventory).resolve()) if private_inventory else None)


def clean(plan, verified_downloads):
    """Caller has stopped the app. Verify all copies before removing any file."""
    root = _root(plan['root'], synthetic_only=plan['synthetic_only'])
    with FileLock(str(root / '_inference.lock'), timeout=0), FileLock(str(root / '_admissions.lock'), timeout=0):
        covered = {(row['path'], row['sha256'], row['bytes']) for row in verified_downloads}
        if any((row['path'], row['sha256'], row['bytes']) not in covered for row in plan['files']):
            raise OperatorError('Faltan descargas SHA-256 del disco. No se borra ninguna copia.')
        scope = hashlib.sha256((str(root) + '\0' + str(plan['code'])).encode()).hexdigest()
        transaction_path = root / ('_cleanup-' + scope + '.json')
        transaction = json.loads(transaction_path.read_text(encoding='utf-8')) if transaction_path.exists() else None
        if transaction and transaction['plan'] != plan:
            raise OperatorError('Transacción de disco distinta. Recupera el recibo original antes de continuar.')
        current = inventory(root, code=plan['code'], private_inventory=plan['private_inventory'], synthetic_only=plan['synthetic_only'],
                            _known_sessions=plan['session_codes'] if transaction else None)
        if not transaction:
            if not current['session_codes'] and not current['files'] and not current['invitations'] and not current['replacements']:
                return dict(empty=True, removed_files=0, replayed=True)
            if current != plan:
                raise OperatorError('Inventario de disco cambió. No se borra; vuelve a descargar y verificar.')
            transaction = dict(schema_version=1, plan=plan, shared={}, removed=[])
            # Durable plan and all verified downloads precede any disk deletion.
            for name in ('_admissions.json', '_replacements.json'):
                shared_path = root / name
                if not shared_path.exists():
                    continue
                value = json.loads(shared_path.read_text(encoding='utf-8'))
                if name == '_admissions.json':
                    value['invitations'] = {k: v for k, v in value['invitations'].items() if k not in plan['invitations']}
                    if value.get('active') in plan['session_codes']:
                        value['active'] = None
                    replacement = value if value['invitations'] else None
                else:
                    replacement = {k: v for k, v in value.items() if k not in plan['replacements']} or None
                transaction['shared'][name] = dict(old_sha256=hashlib.sha256(shared_path.read_bytes()).hexdigest(),
                    replacement=replacement, new_sha256=hashlib.sha256(json.dumps(replacement, ensure_ascii=False, indent=2, allow_nan=False).encode()).hexdigest() if replacement is not None else None)
            atomic_json(transaction_path, transaction)
        # Revoke/remove admissions before unlinking. Interrupted replay stays closed.
        for name, item in transaction['shared'].items():
            shared_path = root / name
            actual = hashlib.sha256(shared_path.read_bytes()).hexdigest() if shared_path.exists() else None
            if actual not in (item['old_sha256'], item['new_sha256']):
                raise OperatorError('Archivo compartido cambió al reanudar. No se toca otro código.')
        for row in plan['files']:
            file = Path(row['path'])
            if file.exists() and hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
                raise OperatorError('Archivo de sesión cambió al reanudar. No se continúa el borrado.')
        # Any new scoped file or code fails closed; absence is allowed only after persisted plan.
        if any(row not in plan['files'] for row in current['files']) or any(
                sid not in plan['session_codes'] for sid in current['session_codes']):
            raise OperatorError('Datos nuevos durante la limpieza. Mantén bloqueada la admisión.')
        for name, item in transaction['shared'].items():
            shared_path = root / name
            if item['replacement'] is not None:
                atomic_json(shared_path, item['replacement'])
            elif shared_path.exists():
                shared_path.unlink()
        deleted_now = 0
        for row in plan['files']:
            file = Path(row['path'])
            if file.exists():
                file.unlink()
                deleted_now += 1
            transaction['removed'].append(row['path'])
            atomic_json(transaction_path, transaction)
        for sid in plan['session_codes']:
            directory = root / sid
            if directory.exists():
                if directory.parent != root or not re.fullmatch('[a-f0-9]{32}', sid):
                    raise OperatorError('Directorio no autorizado. Conserva el recibo de limpieza.')
                directory.rmdir()
        after = inventory(root, code=plan['code'], private_inventory=plan['private_inventory'], synthetic_only=plan['synthetic_only'])
        empty = not after['session_codes'] and not after['files'] and not after['invitations'] and not after['replacements']
        if not empty:
            raise OperatorError('Persisten datos del ámbito. Mantén bloqueada la admisión.')
        transaction_path.unlink()
        return dict(empty=True, removed_files=deleted_now, confirmed_absent_paths=len(plan['files']),
                    removed_invitations=len(plan['invitations']))


def export_by_code(sessions):
    """Pseudonymous research export; free text is private and always reviewed manually."""
    from scripts.study_operator.policy import publication_export

    records, review = [], []
    for session in sessions:
        code = participant_code(session['assignment']['participant_id'])
        attempts = []
        for attempt in session.get('attempts', []):
            item = {key: attempt.get(key) for key in ('analysis_role', 'condition', 'query_id', 'status',
                                                     'elapsed_ms', 'decline_class', 'decline_classifier_version')}
            if attempt.get('analysis_role') == 'free_query':
                review.append(dict(participant_code=code, field='free_query', question=attempt.get('question'),
                                   answer=attempt.get('answer'), publishable=False))
            else:
                item.update(question=attempt.get('question'), answer=attempt.get('answer'))
            attempts.append(item)
        instruments = [{key: row.get(key) for key in ('condition', 'sus', 'sus_score', 'ueq_s', 'ueq_s_scores', 'likert')
                        if key in row}
                       for row in session.get('instruments', [])]
        comparative = session.get('comparative') or {}
        blinding = session.get('blinding') or {}
        review.extend([dict(participant_code=code, field='comparative.C4', text=comparative.get('C4'), publishable=False),
                       dict(participant_code=code, field='blinding.reason', text=blinding.get('reason'), publishable=False)])
        records.append(dict(participant_code=code, purpose=session['purpose'],
                            attempts=attempts, instruments=instruments,
                            comparative={key: comparative.get(key) for key in ('C1', 'C2', 'C3')},
                            blinding_choice=blinding.get('choice')))
    return dict(pseudonymous_by_code=records, private_manual_review=review,
                publication_candidate=publication_export(sessions), automatic_publication_allowed=False)
