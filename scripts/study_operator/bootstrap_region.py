"""Move an uncreated primary to the next reviewed US region, never move an IP."""
import copy
import json
import math
from pathlib import Path

from scripts.study_operator.bootstrap import qualified_snapshot
from scripts.study_operator.cloud_client import no_other_gpu
from scripts.study_operator.policy import OperatorError
from scripts.study_operator.pricing import quote_archive
from scripts.study_operator.region_scope import region


IMMUTABLE = ('project', 'network', 'sessions_bucket', 'technical_bucket', 'service_account',
    'image_id', 'commit', 'operator_commit', 'model_digest', 'ollama_image', 'caddy_image',
    'asset_root', 'ollama_models', 'host_code', 'host_infrastructure_commit', 'fingerprint',
    'bootstrap_inputs', 'reviewed_config', 'artifact_manifest', 'preregistration_file',
    'preregistration_sha256', 'final_snapshot', 'audit_run', 'python', 'primary_vm')


def configure(operator, candidate, *, quote):
    config, state, cloud = operator.config, operator.state, operator.cloud
    if (state.get('primary_bootstrap_complete') or config.get('purpose') != 'technical'
            or state.get('purpose') not in (None, 'technical')
            or state.get('alternate_vms') or state.get('ready_verified')):
        raise OperatorError('Ya hubo un primario o actividad de sesión. Usa la contingencia con identidad y smoke propios; no se reconfigura bootstrap.')
    if any(config.get(key) != candidate.get(key) for key in IMMUTABLE):
        raise OperatorError('Cambió la identidad, el ámbito o el preregistro. No se reubica el primario con otra configuración.')
    destination = region(candidate['zone'])
    if not candidate.get('ip_name', '').startswith('cloudrag-i5-static-'+destination+'-'):
        raise OperatorError('Nombre de IP fuera de la región y ámbito propios. Usa la configuración derivada.')
    if (quote['region'] != destination or quote['machine'] != 'g2-standard-4'
            or not math.isfinite(float(quote['usd_per_hour'])) or float(quote['usd_per_hour']) <= 0
            or candidate['official_rates']['compute_usd_h'] != float(quote['usd_per_hour'])):
        raise OperatorError('Tarifa regional no coincide con el catálogo oficial. Consulta la tarifa antes de reubicar.')
    if state.get('reserved_address_id') or state.get('ip_creation_intent') or config.get('static_ip'):
        raise OperatorError('Libera y verifica primero la IP regional anterior; no se transfiere entre regiones.')
    if cloud.command(['compute', 'addresses', 'list', '--filter=name='+config['ip_name']]):
        raise OperatorError('La IP anterior sigue presente en la API. Concilia su recibo antes de otra región.')
    original = operator.observed()
    if original['status'] != 'TERMINATED':
        raise OperatorError('Detén la VM original antes de reconfigurar; su disco se conserva.')
    instances = cloud.command(['compute', 'instances', 'list'])
    no_other_gpu(instances, selected_id='none')
    pending = state.get('primary_creation_intent')
    if pending and (pending.get('capacity_error_code') != 'ZONE_RESOURCE_POOL_EXHAUSTED'
            or any(row['name'] == pending['name'] for row in instances)):
        raise OperatorError('La creación anterior no tiene stockout y ausencia verificados. No se crea una segunda GPU.')
    qualified_snapshot(operator)
    subnet = cloud.command(['compute', 'networks', 'subnets', 'describe', candidate['subnet'], '--region='+destination])
    if (subnet.get('region', '').split('/')[-1] != destination
            or subnet.get('network', '').split('/')[-1] != config['network']
            or subnet.get('privateIpGoogleAccess') is not True or subnet.get('enableFlowLogs', False)):
        raise OperatorError('Subred regional sin aislamiento equivalente. Conserva los recibos y corrige la preparación.')
    if destination == region(config['zone']):
        if candidate['subnet'] != config['subnet'] or candidate['ip_name'] != config['ip_name']:
            raise OperatorError('La configuración de la misma región cambió. No se adopta.')
        return dict(status='BOOTSTRAP_REGION_ALREADY_CONFIGURED', region=destination)
    if pending:
        state.setdefault('primary_capacity_attempts', []).append(dict(pending,
            absence_verified_utc=operator.now().isoformat()))
        state.pop('primary_creation_intent')
    state.setdefault('bootstrap_region_history', []).append(dict(zone=config['zone'], subnet=config['subnet'],
        ip_name=config['ip_name'], at=operator.now().isoformat(), no_running_gpu=True,
        previous_ip_absent=True, image_unchanged=True))
    for key in ('zone', 'subnet', 'ip_name'):
        config[key] = candidate[key]
    config['official_rates']['compute_usd_h'] = float(quote['usd_per_hour'])
    config['configuration_provenance']['official_quote'] = copy.deepcopy(quote)
    operator.persist()
    return dict(status='BOOTSTRAP_REGION_CONFIGURED_NOT_MEASURED', region=destination,
        next_action='ip-reserve; bootstrap --zone '+candidate['zone']+'; preflight; anexo con push antes de medir')


def from_file(operator, path):
    candidate = json.loads(Path(path).read_bytes())
    if not operator.config.get('audit_run'):
        raise OperatorError('La corrida ya cerró. Prepara una instalación regional nueva con catálogo y supervisión propios; no reabras el paquete sellado.')
    archive = Path(operator.config['audit_run'])/'official-compute-skus'
    quote = quote_archive(archive, 'g2-standard-4', region(candidate['zone']))
    return configure(operator, candidate, quote=quote)
