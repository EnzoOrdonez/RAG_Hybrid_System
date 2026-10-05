"""Fail-closed policy for the new operator; no credentials or participant data."""
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re


class OperatorError(RuntimeError):
    """A readable refusal with the next action, without an underlying payload."""


def purpose_allowed(purpose, operator_root, *, today=None):
    if purpose not in {'study', 'technical', 'smoke', 'rehearsal', 'pilot'}:
        raise OperatorError('Propósito inválido. Usa study, technical, smoke, rehearsal o pilot.')
    if purpose != 'study':
        return {'purpose': purpose, 'ethics_verified': False}
    root = Path(operator_root).resolve() / 'ethics'
    path = root / 'ethics_approval.json'
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        required = {'committee', 'approval_code', 'date', 'pdf_sha256', 'approved_b4_version'}
        if set(value) != required or not all(isinstance(v, str) and v.strip() for v in value.values()):
            raise ValueError('fields')
        if date.fromisoformat(value['date']) > (today or date.today()):
            raise ValueError('future approval')
        if not re.fullmatch('[a-f0-9]{64}', value['pdf_sha256']):
            raise ValueError('hash')
        if hashlib.sha256((root / 'ethics_approval.pdf').read_bytes()).hexdigest() != value['pdf_sha256']:
            raise ValueError('PDF identity')
    except (OSError, ValueError, TypeError, AttributeError):
        raise OperatorError('study bloqueado: Enzo debe guardar la aprobación ética y su PDF en ethics; revisa el runbook.') from None
    return {'purpose': purpose, 'ethics_verified': True,
            'approval_record_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def session_margin(guest_deadline, native_deadline, *, now=None):
    now = now or datetime.now(timezone.utc)
    try:
        limits = [datetime.fromisoformat(value) for value in (guest_deadline, native_deadline)]
        if now.tzinfo is None or any(value.tzinfo is None for value in limits):
            raise ValueError('aware times required')
        seconds = min((value - now).total_seconds() for value in limits)
    except (TypeError, ValueError):
        raise OperatorError('No hay límites de apagado verificables. Ejecuta preflight antes de invitar.') from None
    if seconds < 70 * 60:
        raise OperatorError('Quedan menos de 70 minutos. Detén y vuelve a encender antes de admitir una sesión.')
    return {'remaining_seconds': seconds, 'minimum_seconds': 4200}


def validate_minimal_iam(vm, project_policy, session_policy, technical_policy, *, sa, bucket, metadata_reachable):
    accounts = vm.get('serviceAccounts', [])
    expected_scope = 'https://www.googleapis.com/auth/devstorage.read_write'
    if len(accounts) != 1 or accounts[0].get('email') != sa or set(accounts[0].get('scopes', [])) != {expected_scope}:
        raise OperatorError('Identidad o scopes de VM inválidos. Asigna la SA dedicada con scope storage-rw.')
    member = 'serviceAccount:' + sa
    if any(member in binding.get('members', []) for binding in project_policy.get('bindings', [])):
        raise OperatorError('La SA de la VM tiene un rol de proyecto. Retira ese binding y repite preflight.')
    roles = {binding['role'] for binding in session_policy.get('bindings', []) if member in binding.get('members', [])}
    if roles != {'roles/storage.objectCreator', 'roles/storage.objectViewer'}:
        raise OperatorError('Permisos del bucket de sesiones inválidos. Solo objectCreator y objectViewer para la SA.')
    bindings = [binding for binding in technical_policy.get('bindings', []) if member in binding.get('members', [])]
    expression = "resource.name.startsWith('projects/_/buckets/" + bucket + "/objects/iteration4/')"
    if len(bindings) != 1 or bindings[0]['role'] != 'roles/storage.objectCreator' or bindings[0].get('condition', {}).get('expression') != expression:
        raise OperatorError('Permiso de evidencia demasiado amplio. Limítalo al prefijo iteration4.')
    if metadata_reachable is not False:
        raise OperatorError('El contenedor alcanza metadatos o no se probó su aislamiento. Revisa network=none y los relays Unix.')
    return {'minimal_iam_verified': True, 'metadata_reachable': False}


def private_session_bucket(observed):
    if (observed.get('location', '').upper() != 'US-CENTRAL1'
            or not observed.get('uniform_bucket_level_access')
            or observed.get('public_access_prevention') != 'enforced'
            or observed.get('versioning_enabled', False)
            or str(observed.get('soft_delete_policy', {}).get('retentionDurationSeconds', '')) != '0'):
        raise OperatorError('Bucket de sesiones no apto. Exige us-central1, UBLA, PAP, sin versiones y soft delete=0.')
    return {'private_bucket_verified': True}


def participant_code(code):
    if not isinstance(code, str) or not re.fullmatch(r'P[0-9]{2,6}', code):
        raise OperatorError('Código inválido. Usa P seguido de 2 a 6 dígitos; nunca el nombre de una persona.')
    return code


def confirm_deletion(operation, code, purpose, supplied):
    if operation not in {'withdraw', 'purge-study'}:
        raise OperatorError('Operación de borrado inválida.')
    expected = participant_code(code) if operation == 'withdraw' else 'PURGAR'
    if purpose == 'study' and supplied != expected:
        raise OperatorError('Borrado rechazado. Escribe exactamente ' + expected + ' para confirmar.')
    return expected


def publication_export(sessions):
    """Aggregate counts only. All free text is withheld for manual review."""
    counts = {}
    manual = []
    for session in sessions:
        code = participant_code(session.get('participant_id', session.get('assignment', {}).get('participant_id')))
        for attempt in session.get('attempts', []):
            response_class = (attempt.get('decline_class') or attempt.get('response_class')
                              or ('error' if attempt.get('status') == 'error' else 'unknown'))
            key = (attempt['condition'], response_class)
            counts[key] = counts.get(key, 0) + 1
            if attempt.get('analysis_role') == 'free_query':
                manual.append({'participant_code': code, 'field': 'free_query', 'publishable': False})
        for field in ('comparative.C4', 'blinding.reason'):
            manual.append({'participant_code': code, 'field': field, 'publishable': False})
    return {'aggregate': [{'condition': key[0], 'response_class': key[1], 'count': count}
                          for key, count in sorted(counts.items())],
            'manual_review_inventory': manual, 'automatic_publication_allowed': False}
