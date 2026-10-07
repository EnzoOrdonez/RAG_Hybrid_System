"""Link owner effects to an active I5 supervisor without touching sealed runs."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from filelock import FileLock

from scripts.study_operator.cloud_safety import admission
from scripts.study_operator.policy import OperatorError
from src.ui.components.session_storage import atomic_json

CLOUD_ROOT = Path('C:/CloudRAG')


def package(config):
    value = config.get('audit_run')
    if value is None:
        return None  # Future human sessions use their own owner ledger/native STOP.
    path = Path(value).resolve()
    if (path.parent != CLOUD_ROOT.resolve() or not path.name.startswith('iteration5-run-')
            or any(p.is_symlink() or p.is_junction() for p in (Path(value), *Path(value).parents))):
        raise OperatorError('Paquete de supervisión ajeno. Revisa audit_run; no se inicia un efecto pagado.')
    return path


def owner_exposure(owner):
    cost = owner.get('cost', {})
    values = [cost.get('estimated_usd', 0), cost.get('margin_usd', 0),
              owner.get('ip_transfer_gap_margin_usd', 0), *cost.get('reservations', {}).values()]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise OperatorError('Ledger del operador inválido. Concilia los recibos antes de pagar.')
    return sum(values)


def admit(config, owner, amount, *, now=None):
    if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
        raise OperatorError('Reserva de costo inválida. No se inicia el efecto pagado.')
    root = package(config)
    if root is None:
        return
    if json.loads((root/'STATE.json').read_bytes())['status'] not in {'ACTIVE', 'CLOSING'}:
        raise OperatorError('El paquete ya está cerrado. No se modifica ni admite un efecto pagado.')
    with FileLock(str(root/'state.lock'), timeout=10):
        state = json.loads((root/'STATE.json').read_bytes())
        other = sum(r['maximum_usd'] for key, r in state.get('open_exposures', {}).items() if key != 'operator5')
        try:
            admission(state, other+owner_exposure(owner)+amount, now=now)
        except ValueError as error:
            raise OperatorError('El supervisor cerró admisión o el costo llega al corte. Ejecuta stop y conserva los recibos.') from error


def own_row(kind, row, *, marker=None):
    if kind not in {'vm', 'disk', 'snapshot', 'address', 'firewall', 'subnet'}:
        raise OperatorError('Tipo de recurso propio desconocido.')
    value = dict(type=kind, name=row['name'], id=str(row['id']), disposable=True,
                 ownership_marker=marker or row.get('ownership_marker'))
    if (not value['name'].startswith('cloudrag-i5-') or not value['id'].isdigit()
            or not str(value['ownership_marker']).startswith('CloudRAG-I5-')):
        raise OperatorError('Recurso nuevo sin identidad propia I5. No se registra ni se adopta.')
    for key in ('zone', 'region', 'created_utc'):
        if row.get(key):
            value[key] = row[key]
    if kind in {'vm', 'disk'} and not str(value.get('zone', '')).startswith('us-'):
        raise OperatorError('Zona del recurso propio fuera del ámbito de EE. UU.')
    if kind == 'address' and not str(value.get('region', '')).startswith('us-'):
        raise OperatorError('Región de la IP propia fuera del ámbito de EE. UU.')
    return value


def checkpoint(config, owner, *, now=None):
    root = package(config)
    if root is None:
        return
    if json.loads((root/'STATE.json').read_bytes())['status'] not in {'ACTIVE', 'CLOSING'}:
        return  # Do not even create/acquire a lock in the sealed directory.
    now = now or datetime.now(timezone.utc)
    with FileLock(str(root/'state.lock'), timeout=10):
        state = json.loads((root/'STATE.json').read_bytes())
        if state['status'] not in {'ACTIVE', 'CLOSING'}:
            return  # Never rewrite a closed/sealed package, including on stop.
        for row in owner.get('audit_resources', []):
            if not row.get('disposed'):
                continue
            checked = own_row(row['type'], row)
            matches = [r for r in state['resources'] if r['type'] == checked['type'] and r['name'] == checked['name']]
            if (len(matches) != 1 or matches[0]['id'] != checked['id']
                    or matches[0].get('ownership_marker') != checked['ownership_marker']
                    or row.get('absence_verified') is not True or not row.get('absence_verified_utc')):
                raise OperatorError('Retiro sin identidad y ausencia verificadas. Conserva el estado y consulta la API antes de conciliarlo.')
            matches[0].update(disposed=True, absence_verified=True, absence_verified_utc=row['absence_verified_utc'])
            for intent in state.get('resource_intents', []):
                if intent['type'] == checked['type'] and intent['name'] == checked['name']:
                    intent.update(disposed=True, absence_verified=True)
        additions = [own_row(row['type'], row) for row in owner.get('audit_resources', []) if not row.get('disposed')]
        for row in owner.get('alternate_vms', []):
            if row.get('disposed'):
                continue
            additions.append(own_row('vm', row))
            if row.get('disk_id'):
                additions.append(own_row('disk', dict(row, name=row['disk_name'], id=row['disk_id'])))
        for row in owner.get('snapshots', []):
            additions.append(own_row('snapshot', row))
        ip_intent = owner.get('ip_creation_intent', {})
        if owner.get('reserved_address_id'):
            additions.append(own_row('address', dict(name=config['ip_name'], id=owner['reserved_address_id'],
                region=owner.get('reserved_address_region') or ip_intent.get('region', config.get('zone', config['primary_vm']['zone']).rsplit('-', 1)[0]),
                created_utc=owner.get('ip_reserved_utc')), marker=owner.get('ip_ownership_marker') or ip_intent.get('ownership_marker')))
        for row in additions:
            if any(r['type'] == row['type'] and r['id'] == row['id'] and r['name'] != row['name'] for r in state['resources']):
                raise OperatorError('ID de recurso ya asociado a otro nombre. No se adopta.')
            matches = [r for r in state['resources'] if r['type'] == row['type'] and r['name'] == row['name']]
            if matches and (len(matches) != 1 or matches[0]['id'] != row['id'] or matches[0].get('disposed')
                    or matches[0].get('ownership_marker') != row['ownership_marker']):
                raise OperatorError('Identidad de recurso registrada cambió. Conserva el estado; no repitas creación.')
            if not matches:
                state['resources'].append(row)
        intents = []
        for key in ('alternate_creation_intent', 'primary_creation_intent'):
            row = owner.get(key)
            if row:
                for kind, name in [('vm', row['name']), ('disk', row['disk_name'])]:
                    intents.append(dict(type=kind, name=name, zone=row['zone'], disposable=True,
                        ownership_marker=row['ownership_marker']))
        if ip_intent:
            intents.append(dict(type='address', name=ip_intent['name'], region=ip_intent['region'],
                disposable=True, ownership_marker=ip_intent['ownership_marker']))
        snapshot_intent = owner.get('snapshot_creation_intent')
        if snapshot_intent:
            intents.append(dict(type='snapshot', name=snapshot_intent['name'], disposable=True,
                ownership_marker=snapshot_intent['ownership_marker']))
        iap_intent = owner.get('iap_creation_intent')
        if iap_intent:
            intents.append(dict(type='firewall', name=iap_intent['name'], disposable=True,
                ownership_marker=iap_intent['ownership_marker']))
        for row in intents:
            if (not row['name'].startswith('cloudrag-i5-') or not row['ownership_marker'].startswith('CloudRAG-I5-')
                    or row['type'] in {'vm', 'disk'} and not row['zone'].startswith('us-')):
                raise OperatorError('Intento nuevo fuera del ámbito propio I5.')
            matches = [r for r in state.setdefault('resource_intents', []) if r['type'] == row['type'] and r['name'] == row['name']]
            if matches and any(any(r.get(k) != v for k, v in row.items()) for r in matches):
                raise OperatorError('Intento de creación registrado difiere. Concilia antes de repetir.')
            if not matches:
                state['resource_intents'].append(row)
        state.setdefault('open_exposures', {})['operator5'] = dict(maximum_usd=owner_exposure(owner), not_billed_spend=True)
        state['operator5_checkpoint'] = dict(at=now.isoformat(), resources=len(additions), intents=len(intents),
            owner_state_sha256=hashlib.sha256(json.dumps(owner, sort_keys=True).encode()).hexdigest(),
            session_content_not_copied=True)
        state['updated_utc'] = now.isoformat()
        atomic_json(root/'STATE.json', state)
