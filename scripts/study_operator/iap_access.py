"""Own temporary IAP rule; no public SSH and no adoption by name alone."""
import uuid

from scripts.study_operator.cloud_client import no_other_gpu
from scripts.study_operator.policy import OperatorError
from scripts.study_operator.run_checkpoint import admit

SOURCE = '35.235.240.0/20'
TAG = 'cloudrag-i3-managed'


def validate(row, operator):
    if (row.get('name') != operator.config['iap_name'] or not str(row.get('id')).isdigit()
            or row.get('description') != operator.state.get('iap_creation_intent', {}).get('ownership_marker')
            or row.get('network', '').split('/')[-1] != operator.config['network']
            or row.get('direction') != 'INGRESS' or row.get('disabled', False)
            or row.get('sourceRanges') != [SOURCE] or row.get('targetTags') != [TAG]
            or row.get('allowed') != [dict(IPProtocol='tcp', ports=['22'])]
            or row.get('sourceTags') or row.get('sourceServiceAccounts')
            or row.get('targetServiceAccounts') or row.get('denied')):
        raise OperatorError('Regla IAP ajena o más amplia. No se adopta ni cambia; verifica el ID y el recibo.')
    return row


def prepare(operator):
    name = operator.config['iap_name']
    if not name.startswith('cloudrag-i5-') or not name.endswith('-iap'):
        raise OperatorError('Nombre IAP fuera del ámbito propio I5. Revisa la instalación.')
    rows = operator.cloud.command(['compute', 'firewall-rules', 'list', '--filter=name='+name])
    if not rows:
        admit(operator.config, operator.state, 0, now=operator.now())
        if operator.config.get('audit_run') and any(row.get('disposed') and row['type'] == 'firewall'
                and row['name'] == name for row in operator.state.get('audit_resources', [])):
            raise OperatorError('La regla de esta corrida ya fue retirada. No se reutiliza su nombre; conserva el recibo y prepara una corrida nueva.')
        if operator.state.get('iap_rule_id'):
            raise OperatorError('La regla registrada desapareció. Concilia su ausencia antes de crear otra.')
        operator.state.setdefault('iap_creation_intent', dict(name=name,
            ownership_marker='CloudRAG-I5-IAP-'+uuid.uuid4().hex))
        operator.persist()
        operator.cloud.command(['compute', 'firewall-rules', 'create', name,
            '--network='+operator.config['network'], '--direction=INGRESS', '--priority=1000',
            '--action=ALLOW', '--rules=tcp:22', '--source-ranges='+SOURCE, '--target-tags='+TAG,
            '--description='+operator.state['iap_creation_intent']['ownership_marker']])
        rows = operator.cloud.command(['compute', 'firewall-rules', 'list', '--filter=name='+name])
    if len(rows) != 1:
        raise OperatorError('No hay una única regla IAP propia. Conserva el intento y verifica el inventario.')
    row = validate(rows[0], operator)
    if operator.state.get('iap_rule_id') not in (None, str(row['id'])):
        raise OperatorError('El ID de IAP cambió. No se adopta ni se borra una regla recreada.')
    operator.state['iap_rule_id'] = str(row['id'])
    resource = dict(type='firewall', name=name, id=str(row['id']),
                    ownership_marker=row['description'], created_utc=row['creationTimestamp'])
    if resource not in operator.state.setdefault('audit_resources', []):
        operator.state['audit_resources'].append(resource)
    operator.persist()
    return dict(status='OWN_IAP_RULE_VERIFIED', id=str(row['id']), public_ssh=False, idle_usd_day=0)


def release(operator):
    instances = operator.cloud.command(['compute', 'instances', 'list'])
    no_other_gpu(instances, selected_id='none')
    name = operator.config['iap_name']
    rows = operator.cloud.command(['compute', 'firewall-rules', 'list', '--filter=name='+name])
    if rows:
        if len(rows) != 1:
            raise OperatorError('IAP no tiene identidad única. No se elimina ninguna regla.')
        row = validate(rows[0], operator)
        if str(row['id']) != operator.state.get('iap_rule_id'):
            raise OperatorError('IAP no corresponde al ID propio. Conserva los recibos antes de retirar.')
        operator.cloud.command(['compute', 'firewall-rules', 'delete', name])
        if operator.cloud.command(['compute', 'firewall-rules', 'list', '--filter=name='+name]):
            raise OperatorError('IAP sigue presente en la API. No se declara retirada; conserva el ID.')
    for row in operator.state.get('audit_resources', []):
        if row['type'] == 'firewall' and row['name'] == name:
            row.update(disposed=True, absence_verified=True, absence_verified_utc=operator.now().isoformat())
    operator.state.pop('iap_rule_id', None)
    operator.state.pop('iap_creation_intent', None)
    operator.persist()
    return dict(status='OWN_IAP_RULE_ABSENT_VERIFIED', public_ssh=False)
