"""Create the first final-image primary without altering the original I2 disk."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import uuid

from scripts.study_operator.cloud_client import checked_vm, no_other_gpu
from scripts.study_operator.deployment import checked_config
from scripts.study_operator.policy import OperatorError
from scripts.study_operator.region_scope import US_L4_ZONES, region
from scripts.study_operator.startup import write_startup


def qualified_snapshot(operator):
    pin = operator.config.get('final_snapshot', {})
    try:
        content = Path(pin['restoration_proof']).read_bytes()
        proof = json.loads(content)
        if (hashlib.sha256(content).hexdigest() != pin['restoration_proof_sha256']
                or proof['status'] != 'CPU_RESTORATION_VERIFIED' or proof['synthetic'] is not False
                or proof['source_snapshot_id'] != pin['id'] or proof['image_id'] != operator.config['image_id']
                or proof['source']['files'] != 19 or proof['source']['all_expected_files_verified'] is not True
                or proof['all_expected_files_verified'] is not True or proof['image_config_verified'] is not True
                or proof['model_manifest_and_blobs_verified'] is not True
                or proof['runtime_user_pair']['status'] != 'PAIRED_RUNTIME_USER_SUPPORTED'):
            raise ValueError('Proof mismatch')
        live = operator.cloud.command(['compute', 'snapshots', 'describe', pin['name']])
        if str(live['id']) != pin['id'] or live['status'] != 'READY':
            raise ValueError('Snapshot mismatch')
    except (KeyError, OSError, ValueError, TypeError):
        raise OperatorError('Falta la restauración CPU calificada de la instantánea final. Conserva los recursos y verifica su recibo; no se crea una VM.') from None
    return live


def bootstrap(operator, zone):
    config, state, cloud = operator.config, operator.state, operator.cloud
    if zone not in US_L4_ZONES or region(zone) != region(config['zone']):
        raise OperatorError('La zona no coincide con la instalación revisada. Prepara subred, IP, tarifa e identidad propias antes de bootstrap.')
    pending = state.get('primary_creation_intent')
    if (not state.get('primary_bootstrap_complete') and pending and pending.get('start_requested_utc')
            and pending.get('capacity_error_code') != 'ZONE_RESOURCE_POOL_EXHAUSTED'):
        # An absent resource does not prove that retrying its unknown failure is
        # eligible. Check before recreating a disk or spending another reserve.
        observed = cloud.command(['compute', 'instances', 'list'])
        matches = [row for row in observed if row['name'] == pending['name']]
        if not matches:
            raise OperatorError('La creación anterior tiene resultado desconocido y la VM está ausente. Conserva el recibo y diagnostica la causa; bootstrap no repite la creación ni recrea el disco.')
        if (len(matches) != 1 or matches[0].get('description') != pending['ownership_marker']
                or zone != pending['zone']):
            raise OperatorError('La creación desconocida no coincide con una VM propia en esa zona. Concilia identidad y recibos antes de otro efecto.')
    if state.get('primary_bootstrap_complete'):
        if zone != operator.selected()['zone']:
            raise OperatorError('Ya existe un primario final. Usa failover con su instantánea preparada; bootstrap no crea otro.')
        vm = operator.observed()
        return dict(status='PRIMARY_ALREADY_CREATED', vm_id=str(vm['id']), next_action='status y preflight')
    checked_config(dict(config, zone=zone))
    snapshot = qualified_snapshot(operator)
    addresses = cloud.command(['compute', 'addresses', 'list', '--filter=name='+config['ip_name']])
    if (len(addresses) != 1 or str(addresses[0]['id']) != state.get('reserved_address_id')
            or addresses[0].get('description') != state.get('ip_ownership_marker')
            or addresses[0].get('address') != config['static_ip']
            or not addresses[0].get('region', '').endswith('/'+zone.rsplit('-', 1)[0])):
        raise OperatorError('IP de bootstrap sin reserva propia en su región. Ejecuta ip-reserve y conserva el recibo antes de crear la VM.')
    subnet = cloud.command(['compute', 'networks', 'subnets', 'describe', config['subnet'],
                            '--region='+zone.rsplit('-', 1)[0]])
    if (subnet.get('network', '').split('/')[-1] != config['network']
            or not subnet.get('region', '').endswith('/'+zone.rsplit('-', 1)[0])
            or subnet.get('privateIpGoogleAccess') is not True or subnet.get('enableFlowLogs', False)):
        raise OperatorError('Subred de bootstrap incompatible. Verifica región, acceso privado y registros antes de crear la VM.')
    original = operator.observed()
    if original['status'] != 'TERMINATED':
        raise OperatorError('Detén y verifica la VM original antes de bootstrap; su disco se conserva.')
    instances = cloud.command(['compute', 'instances', 'list'])
    intent = state.get('primary_creation_intent')
    if intent and intent['zone'] != zone:
        if (intent.get('capacity_error_code') != 'ZONE_RESOURCE_POOL_EXHAUSTED'
                or any(row['name'] == intent['name'] for row in instances)):
            raise OperatorError('La creación anterior no terminó con stockout y ausencia verificados. Concilia ese intento antes de otra zona.')
        no_other_gpu(instances, selected_id='none')
        state.setdefault('primary_capacity_attempts', []).append(dict(intent,
            absence_verified_utc=operator.now().isoformat()))
        state.pop('primary_creation_intent')
        operator.persist()
        intent = None
    config['zone'] = zone
    if not intent:
        no_other_gpu(instances, selected_id='none')
        operator.reserve_cost('bootstrap-primary-'+zone, 3*config['official_rates']['compute_usd_h']+.25)
        operator.reserve_cost('disk-retention-primary-'+zone, 100*config['official_rates']['persistent_disk_gib_usd_h']*72)
        if region(zone) != 'us-central1':
            operator.reserve_cost('snapshot-transfer-primary-'+zone, 100*config['official_rates']['snapshot_transfer_na_usd_gib'])
        name = 'cloudrag-i5-primary-'+zone.replace('us-', '')+'-'+snapshot['id'][-12:]
        intent = dict(name=name, disk_name=name+'-boot', zone=zone, source_snapshot_id=str(snapshot['id']),
                      ownership_marker='CloudRAG-I5-primary-'+uuid.uuid4().hex)
        state['primary_creation_intent'] = intent
        operator.persist()
    if intent['zone'] != zone or intent['source_snapshot_id'] != str(snapshot['id']):
        raise OperatorError('Otro bootstrap está pendiente. Concilia el intento conservado antes de crear recursos.')
    existing = [row for row in instances if row['name'] == intent['name']]
    if existing and (len(existing) != 1 or existing[0].get('description') != intent['ownership_marker']):
        raise OperatorError('Nombre de bootstrap ya ocupado por un recurso ajeno. No se adopta ni se crea otro disco.')
    no_other_gpu(instances, selected_id=str(existing[0]['id']) if len(existing) == 1 else 'none')
    disks = cloud.command(['compute', 'disks', 'list', '--filter=name='+intent['disk_name']])
    if not disks:
        cloud.command(['compute', 'disks', 'create', intent['disk_name'], '--zone='+zone, '--size=100GB',
            '--type=pd-balanced', '--source-snapshot='+snapshot['name'], '--description='+intent['ownership_marker']], timeout=600)
        disks = cloud.command(['compute', 'disks', 'list', '--filter=name='+intent['disk_name']])
    if (len(disks) != 1 or str(disks[0].get('sourceSnapshotId')) != intent['source_snapshot_id']
            or not disks[0].get('zone', '').endswith('/'+zone) or disks[0].get('description') != intent['ownership_marker']):
        raise OperatorError('Disco de bootstrap ajeno o incompatible. No se adopta, recrea ni borra.')
    disk = disks[0]
    row = dict(type='disk', name=disk['name'], id=str(disk['id']), zone=zone,
               ownership_marker=intent['ownership_marker'], created_utc=disk['creationTimestamp'])
    if row not in state.setdefault('audit_resources', []):
        state['audit_resources'].append(row)
    operator.persist()
    if not existing:
        requested = operator.now()
        launch = dict(config, start_requested_utc=requested.isoformat(),
                      native_deadline_utc=(requested+timedelta(hours=3)).isoformat())
        script = write_startup(cloud.root/'primary-startup.sh', launch, discover_instance=True)
        operator.detach_ip_for_creation()
        intent.update(start_requested_utc=requested.isoformat(), startup_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
        operator.persist()
        try:
            cloud.command(['compute', 'instances', 'create', intent['name'], '--zone='+zone, '--machine-type=g2-standard-4',
                '--accelerator=type=nvidia-l4,count=1', '--provisioning-model=STANDARD', '--maintenance-policy=TERMINATE',
                '--disk=name='+disk['name']+',boot=yes,auto-delete=no', '--deletion-protection', '--max-run-duration=3h',
                '--instance-termination-action=STOP', '--service-account='+config['service_account'], '--scopes=storage-rw',
                '--network='+config['network'], '--subnet='+config['subnet'], '--address='+config['static_ip'],
                '--metadata-from-file=startup-script='+str(script), '--metadata=enable-guest-attributes=TRUE',
                '--tags=cloudrag-i3-managed', '--description='+intent['ownership_marker']], timeout=600)
        except OperatorError as error:
            if str(error).startswith('ZONE_RESOURCE_POOL_EXHAUSTED'):
                intent['capacity_error_code'] = 'ZONE_RESOURCE_POOL_EXHAUSTED'
                operator.persist()
            raise
        existing = [cloud.command(['compute', 'instances', 'describe', intent['name'], '--zone='+zone])]
    if len(existing) != 1:
        raise OperatorError('Bootstrap sin una única VM propia. Conserva el intento y verifica el inventario.')
    vm = existing[0]
    selected = dict(name=intent['name'], id=str(vm['id']), zone=zone)
    checked_vm(vm, name=selected['name'], instance_id=selected['id'], zone=zone)
    startup = next((r['value'] for r in vm.get('metadata', {}).get('items', []) if r['key'] == 'startup-script'), None)
    if (vm.get('description') != intent['ownership_marker'] or vm['disks'][0]['source'] != disk['selfLink']
            or not startup or hashlib.sha256(startup.encode()).hexdigest() != intent.get('startup_sha256')):
        raise OperatorError('VM creada sin identidad o supervisor coincidente. No se adopta; verifica status y conserva sus recibos.')
    row = dict(type='vm', **selected, ownership_marker=intent['ownership_marker'], created_utc=vm['creationTimestamp'])
    if row not in state['audit_resources']:
        state['audit_resources'].append(row)
    state.update(selected_vm=selected, purpose=config['purpose'], boot_started_utc=intent['start_requested_utc'],
                 ready_verified=False, primary_bootstrap_complete=True)
    config.setdefault('original_preserved_vm', config['primary_vm'])
    config['primary_vm'] = selected
    config['instance_id'] = selected['id']
    config['primary_disk'] = dict(name=disk['name'], id=str(disk['id']), zone=zone,
                                  source_snapshot_id=str(snapshot['id']))
    operator.persist()
    return dict(status='FINAL_PRIMARY_CREATED_NOT_READY', vm_id=selected['id'], zone=zone,
                original_vm_and_disk_preserved=True, next_action='preflight; el arranque adquirido conserva su STOP nativo')
