"""Idempotent owner operations for the prepared installation, with receipts."""
import base64
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import shlex
import socket
import ssl
import time
import uuid

from scripts.study_operator.cloud_client import checked_vm, no_other_gpu, readiness
from scripts.study_operator.deployment import checked_config
from scripts.study_operator.policy import OperatorError, ReadyPending, participant_code, purpose_allowed, session_margin
from scripts.study_operator.service_gateway import save_state
from scripts.study_operator.region_scope import region


class Operator:
    def __init__(self, root, cloud, *, now=None, sleep=time.sleep):
        self.root, self.cloud, self.sleep = Path(root), cloud, sleep
        self.now = now or (lambda: datetime.now(timezone.utc))
        try:
            self.config = json.loads((self.root/'installation.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):
            raise OperatorError('Falta installation.json válido. Usa la instalación sellada de iteración 5; no copies el operador anterior.') from None
        self.state_path = self.root/'active.json'
        self.state = json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {}

    def persist(self):
        from scripts.study_operator.run_checkpoint import checkpoint

        checkpoint(self.config,self.state,now=self.now())
        save_state(self.root/'installation.json',self.config)
        save_state(self.state_path,self.state)

    def selected(self):
        return self.state.get('selected_vm',self.config['primary_vm'])

    def observed(self):
        selected = self.selected()
        observed = self.cloud.command(['compute','instances','describe',selected['name'],'--zone='+selected['zone']])
        return checked_vm(observed,name=selected['name'],instance_id=selected['id'],zone=selected['zone'])

    def bridge(self, request, *, private=False, timeout=180):
        selected = self.selected()
        fixed = self.config['host_code']+'/scripts/study_operator/guest_bridge.py'
        content = self.cloud.command(['compute','ssh',selected['name'],'--zone='+selected['zone'],
            '--tunnel-through-iap','--ssh-key-expire-after=10m','--command=sudo -n python3 -B '+shlex.quote(fixed)],
            input_data=json.dumps(request).encode(),json_output=False,private_output=private,timeout=timeout)
        try:
            result = json.loads(content)
            if result.get('status') == 'WAITING' and request.get('operation') == 'preflight':
                raise ReadyPending('El invitado sigue preparando READY. Espera dentro del límite de 15 minutos; no repitas start.')
            if result.get('status') == 'ERROR':
                reason = result.get('reason','GUEST_OPERATION_REJECTED')
                if not re.fullmatch('[A-Z][A-Z0-9_]{0,80}',str(reason)):
                    reason = 'GUEST_OPERATION_REJECTED'
                raise OperatorError('El invitado rechazó la operación ('+reason+'). Ejecuta diagnostics y status; conserva los recibos antes de repetir.')
            return result
        except (TypeError,ValueError):
            raise OperatorError('El invitado rechazó la operación o aún no responde. Conserva el recibo, ejecuta status y revisa preflight.') from None

    def status(self):
        observed = self.observed()
        return dict(status=observed['status'],vm_id=str(observed['id']),zone=observed['zone'].split('/')[-1],
                    retained_disk=True,deletion_protection=True,native_stop_s=10800)

    def diagnostics(self):
        observed = self.observed()
        provenance = {}
        if observed['status'] != 'RUNNING':
            from scripts.study_operator.bootstrap_failure import validate_summary
            from scripts.study_operator.gcs import Storage

            storage = Storage(self.config['technical_bucket'],self.cloud.owner_token)
            prefix = 'iteration4/failed-boots/'+str(observed['id'])+'/'
            rows = storage.objects(prefix)
            if not rows:
                raise OperatorError('VM detenida sin recibo de fallo publicado. Conserva runs y el disco; no repitas start para adivinar la causa. Solicita recuperación técnica del disco.')
            row = max(rows,key=lambda item:item['timeCreated'])
            try:
                content = storage.read(row['name'],row['generation'])
                if (len(content) != int(row['size'])
                        or base64.b64encode(hashlib.md5(content).digest()).decode() != row['md5Hash']):
                    raise ValueError('FAILED_BOOT_OBJECT_CHECKSUM_CHANGED')
                result = validate_summary(json.loads(content),instance_id=observed['id'],
                    image_id=self.config['image_id'],commit=self.config['commit'])
                if row['name'] != prefix+result['boot_id']+'.json':
                    raise ValueError('FAILED_BOOT_OBJECT_NAME_CHANGED')
            except (ValueError,KeyError,TypeError):
                raise OperatorError('Recibo de fallo incompatible o alterado. Conserva el disco y los metadatos; no lo trates como READY ni repitas start.') from None
            result = dict(result,files={'failure.json':result['failure']},source='IMMUTABLE_GCS_FAILURE_RECEIPT')
            provenance = dict(object=row['name'],generation=str(row['generation']),
                server_created=row['timeCreated'],object_sha256=hashlib.sha256(content).hexdigest())
        else:
            result = self.bridge(dict(operation='technical-evidence'))
        path = self.cloud.root/'technical-diagnostics.json'
        save_state(path,result)
        return dict(status='TECHNICAL_DIAGNOSTICS_SAVED',path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    boot_id=result['boot_id'],receipt_count=len(result['files']),
                      failure=result['files'].get('failure.json'),session_content_excluded=True,
                      provenance=provenance,
                    next_action='Revisa el primer comando con exit_code distinto de cero; corrige su causa y ejecuta stop antes de otro start.')

    def reserve_cost(self, operation, amount):
        from scripts.study_operator.run_checkpoint import admit

        admit(self.config,self.state,amount,now=self.now())
        cost = self.state.setdefault('cost',dict(estimated_usd=0,margin_usd=0,reservations={}))
        inherited = self.config.get('cost',{})
        retention = 0
        if inherited.get('as_of_utc'):
            seconds = max(0,(self.now()-datetime.fromisoformat(inherited['as_of_utc'])).total_seconds())
            retention = seconds/86400*inherited.get('retention_usd_day',0)
        for resource in self.state.get('alternate_vms',[]):
            if resource.get('created_utc') and not resource.get('disposed'):
                retention += max(0,(self.now()-datetime.fromisoformat(resource['created_utc'])).total_seconds())/86400*.3287664
        for resource in self.state.get('snapshots',[]):
            retention += max(0,(self.now()-datetime.fromisoformat(resource['created_utc'])).total_seconds())/86400*resource['idle_usd_day']
        # Initial reservation covers72h. Transfer gaps or external cleanup may
        # invalidate the first association timestamp; use the unused-rate upper.
        active_ip_extra = 0
        if self.state.get('ip_reserved_utc'):
            elapsed = self.ip_charge_upper()['estimated_usd']
            active_ip_extra = max(0,elapsed-sum(value for key,value in cost['reservations'].items() if key.startswith('ip-')))
        total = (inherited.get('estimated_usd',0)+inherited.get('margin_usd',0)+cost['estimated_usd']+
                 cost['margin_usd']+sum(cost['reservations'].values())+amount+retention+active_ip_extra)
        total += self.state.get('ip_transfer_gap_margin_usd',0)
        if not 0 <= total < 90:
            raise OperatorError('El costo reservado alcanza el corte de USD90. Mantén apagada la VM y concilia el ledger.')
        cost['reservations'][operation] = amount
        self.persist()
        save_state(self.cloud.root/('cost-'+operation+'.json'),dict(status='RESERVED_BEFORE_EFFECT',
            estimate_usd=amount,conservative_total_usd=total,official_rates=self.config.get('official_rates'),
            elapsed_retention_estimate_usd=retention,elapsed_ip_above_reservation_usd=active_ip_extra,
            observed_utc=self.now().isoformat()))

    def start(self, purpose):
        requested = self.now().isoformat()
        purpose_allowed(purpose,self.root)
        observed = self.observed()
        if observed['status'] == 'RUNNING':
            if self.state.get('purpose') != purpose:
                raise OperatorError('La VM ya corre con otro propósito. Ejecuta stop antes de cambiarlo.')
            return dict(status='ALREADY_RUNNING',next_action='preflight')
        if observed['status'] != 'TERMINATED':
            raise OperatorError('La VM está cambiando de estado. Ejecuta status; no repitas start hasta TERMINATED.')
        instances = self.cloud.command(['compute','instances','list'])
        no_other_gpu(instances,selected_id=observed['id'])
        periods = self.config.setdefault('period_ids',{})
        new_period = purpose not in periods and self.config.get('purpose') != purpose
        periods.setdefault(purpose,self.config['period_id'] if self.config.get('purpose') == purpose else uuid.uuid4().hex)
        self.config.update(purpose=purpose,period_id=periods[purpose])
        if new_period:
            # New random period, never invited or written on a VM. Record its empty origin.
            self.state['failover_data_reconciled'] = True
            self.state['period_origin'] = dict(period_id=periods[purpose],status='NEW_UNINVITED_PERIOD')
        config = dict(self.config,zone=self.selected()['zone'],instance_id=self.selected()['id'],
                      native_deadline_utc=(self.now()+timedelta(hours=3)).isoformat(),start_requested_utc=requested)
        checked_config(config)
        self.reserve_cost('boot-'+self.now().strftime('%Y%m%dT%H%M%S%fZ'),
            3*self.config['official_rates']['compute_usd_h']+.25)
        self.state.update(purpose=purpose,boot_started_utc=self.now().isoformat(),ready_verified=False)
        self.state.pop('snapshot_empty_verified',None)
        self.persist()
        script = self.cloud.root/'startup.sh'
        from scripts.study_operator.startup import write_startup

        write_startup(script,config)
        selected = self.selected()
        self.cloud.command(['compute','instances','add-metadata',selected['name'],'--zone='+selected['zone'],
            '--metadata-from-file=startup-script='+str(script),'--metadata=enable-guest-attributes=TRUE'])
        self.cloud.command(['compute','instances','start',selected['name'],'--zone='+selected['zone']],timeout=300)
        return dict(status='STARTED_SUPERVISED',next_action='preflight',purpose=purpose,maximum_ready_s=900)

    def preflight(self):
        observed = self.observed()
        if observed['status'] != 'RUNNING':
            raise OperatorError('La VM no está RUNNING. Ejecuta start y después preflight.')
        value = self.bridge(dict(operation='preflight'))
        ready = readiness(value.get('ready'),image_id=self.config['image_id'],url='https://'+self.config['hostname'],
                          boot_id=value.get('boot_id'))
        if ready.get('start_to_ready_s',901) > 900:
            raise OperatorError('READY excedió 15 minutos. No admitas una sesión; conserva el recibo y reprograma.')
        session_margin(value['guest_deadline_utc'],value['native_deadline_utc'],now=self.now())
        # Independent owner readback of actual IAM, bucket configuration and firewall.
        from scripts.study_operator.policy import private_session_bucket, validate_minimal_iam
        project_policy = self.cloud.command(['projects','get-iam-policy',self.cloud.project])
        session_policy = self.cloud.command(['storage','buckets','get-iam-policy','gs://'+self.config['sessions_bucket']])
        technical_policy = self.cloud.command(['storage','buckets','get-iam-policy','gs://'+self.config['technical_bucket']])
        bucket = self.cloud.command(['storage','buckets','describe','gs://'+self.config['sessions_bucket']])
        private_session_bucket(bucket)
        validate_minimal_iam(observed,project_policy,session_policy,technical_policy,
            sa=self.config['service_account'],bucket=self.config['technical_bucket'],
            metadata_reachable=not value.get('metadata_unreachable',False))
        rules = self.cloud.command(['compute','firewall-rules','list'])
        public = [rule for rule in rules if not rule.get('disabled') and rule.get('network','').endswith('/'+self.config['network'])
            and rule.get('direction') == 'INGRESS' and '0.0.0.0/0' in rule.get('sourceRanges',[])
            and (not rule.get('targetTags') or 'cloudrag-i3-managed' in rule['targetTags'])]
        if any(rule.get('allowed') != [{'IPProtocol':'tcp','ports':['443']}] for rule in public) or not public:
            raise OperatorError('Firewall público distinto de 443. Mantén cerrada la admisión y revisa las reglas.')
        with socket.create_connection((self.config['hostname'],443),timeout=10) as connection:
            with ssl.create_default_context().wrap_socket(connection,server_hostname=self.config['hostname']) as tls:
                if hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest() != ready['tls']['certificate_sha256']:
                    raise OperatorError('El certificado externo difiere del recibo. Revisa la IP y repite preflight.')
        self.state.update(ready_verified=True,boot_id=value['boot_id'],ready=ready,
            guest_deadline_utc=value['guest_deadline_utc'],native_deadline_utc=value['native_deadline_utc'])
        self.state['failover_data_reconciled'] = (value.get('session_count') == 0 and value.get('invitation_count') == 0)
        self.persist()
        return dict(status='READY_VERIFIED',url=ready['url'],purpose=self.state['purpose'],
                    start_to_ready_s=ready['start_to_ready_s'])

    def stop(self):
        observed = self.observed()
        selected = self.selected()
        if observed['status'] != 'TERMINATED':
            # Host performs ordered shutdown; owner's STOP is a bounded fallback.
            try:
                shutdown = self.bridge(dict(operation='stop'),timeout=30)
                self.state['failover_data_reconciled'] = shutdown.get('failover_data_reconciled',False)
            except OperatorError:
                pass
            self.cloud.command(['compute','instances','stop',selected['name'],'--zone='+selected['zone']],timeout=600)
        if self.observed()['status'] != 'TERMINATED':
            raise OperatorError('STOP no confirmado. Verifica status; el límite nativo sigue activo.')
        self.cloud.command(['compute','instances','remove-metadata',selected['name'],'--zone='+selected['zone'],
                            '--keys=startup-script'])
        self.state['ready_verified'] = False
        if self.state.get('boot_started_utc'):
            began = datetime.fromisoformat(self.state.pop('boot_started_utc'))
            seconds = max(0,(self.now()-began).total_seconds())
            self.state['cost']['estimated_usd'] += seconds/3600*self.config['official_rates']['compute_usd_h']
            self.state['cost']['margin_usd'] += .25
            self.state['cost']['reservations'] = {key:value for key,value in self.state['cost']['reservations'].items()
                                                 if key.startswith(('ip-','disk-retention-','snapshot-transfer-'))}
            self.state['last_estimated_vm_interval_s'] = seconds
        self.persist()
        return dict(status='TERMINATED_VERIFIED',retained_disk=True,metadata_disarmed=True)

    def invite(self, code, *, cell=None, profile=None):
        participant_code(code)
        purpose = self.state.get('purpose')
        purpose_allowed(purpose,self.root)
        if purpose == 'study' and (cell is not None or profile is not None):
            raise OperatorError('study usa la asignación congelada. Ejecuta invite con el código, sin --cell ni --profile.')
        if purpose != 'study' and (cell not in {1,2,3,4} or profile not in {'with_experience','without_experience'}):
            raise OperatorError('La invitación sintética exige --cell 1|2|3|4 y --profile with_experience|without_experience.')
        self.preflight()
        from scripts.study_operator.gcs import Storage

        prefix = 'periods/'+self.config['period_id']+'/'+code+'/'
        if Storage(self.config['sessions_bucket'],self.cloud.owner_token).objects(prefix):
            raise OperatorError('El código ya tiene una sesión respaldada en este periodo. No repitas la observación; revisa su recibo.')
        token = secrets.token_urlsafe(32)
        request = dict(operation='invite',participant_id=code,token_sha256=hashlib.sha256(token.encode()).hexdigest(),
                       cell=cell,profile=profile)
        self.bridge(request,private=True)
        self.state['failover_data_reconciled'] = False
        self.persist()
        # Caller prints only once to the interactive console. Never include in receipt objects.
        return token

    def ip_reserve(self):
        name = self.config['ip_name']
        ip_region = region(self.config['zone'])
        if self.state.get('reserved_address_region') not in (None, ip_region):
            raise OperatorError('La región cambió mientras hay otra IP reservada. Libera la IP anterior y conserva su recibo antes de reubicar.')
        addresses = self.cloud.command(['compute','addresses','list','--filter=name='+name])
        if not addresses:
            intent = self.state.get('ip_creation_intent')
            if intent and (intent.get('name') != name or intent.get('region') != ip_region):
                raise OperatorError('Hay otra reserva de IP pendiente. Concilia su recibo antes de crear una nueva.')
            if self.state.get('reserved_address_id') or self.state.get('ip_reserved_utc'):
                self.finish_ip_release('ABSENT_BEFORE_NEW_RESERVATION')
                intent = self.state.get('ip_creation_intent')
            if not intent:
                self.reserve_cost('ip-'+self.now().strftime('%Y%m%dT%H%M%S%fZ'),.01*24*3)
                intent = dict(name=name,region=ip_region,requested_utc=self.now().isoformat(),
                              ownership_marker='CloudRAG-I5-owned-'+uuid.uuid4().hex)
            self.state['ip_creation_intent'] = intent
            self.persist()
            self.cloud.command(['compute','addresses','create',name,'--region='+ip_region,
                                '--description='+intent['ownership_marker']])
        observed = self.cloud.command(['compute','addresses','describe',name,'--region='+ip_region])
        if not observed.get('region','').endswith('/'+ip_region) or observed.get('addressType') != 'EXTERNAL':
            raise OperatorError('La IP no es externa en la región de esta instalación. Conserva el recurso y revisa su recibo.')
        owned_id = self.state.get('reserved_address_id')
        intent = self.state.get('ip_creation_intent',{})
        if ((owned_id and str(observed['id']) != owned_id)
                or (not owned_id and (intent.get('name') != name or intent.get('region') != ip_region
                                      or observed.get('description') != intent.get('ownership_marker')))):
            raise OperatorError('La IP no acredita el intento de creación propio. No se asocia ni se libera; revisa los recibos.')
        self.config.update(static_ip=observed['address'],hostname=observed['address']+'.sslip.io')
        self.state['reserved_address_id'] = str(observed['id'])
        self.state['reserved_address_region'] = ip_region
        self.state['ip_ownership_marker'] = observed.get('description')
        self.state.setdefault('ip_reserved_utc',observed.get('creationTimestamp',self.now().isoformat()))
        self.persist()  # Recoverable even if association fails or its response is lost.
        selected = self.selected()
        vm = self.observed()
        if vm['status'] != 'TERMINATED':
            raise OperatorError('La IP está reservada, pero la VM corre. Ejecuta stop antes de asociarla.')
        association_pending = region(selected['zone']) != ip_region
        if not association_pending:
            self.attach_ip(selected,vm)
            self.state.setdefault('ip_associated_utc',self.now().isoformat())
        self.state.pop('ip_creation_intent',None)
        self.persist()
        return dict(status='STATIC_IP_RESERVED',url='https://'+self.config['hostname'],idle_usd_day=.24 if association_pending else .12,
                    unused_usd_day=.24,address_id=str(observed['id']),association_pending_bootstrap=association_pending)

    def attach_ip(self, selected, observed):
        interface = observed['networkInterfaces'][0]
        configs = interface.get('accessConfigs',[])
        if len(configs) == 1 and configs[0].get('natIP') == self.config['static_ip']:
            return
        for access in configs:
            self.cloud.command(['compute','instances','delete-access-config',selected['name'],'--zone='+selected['zone'],
                '--network-interface='+interface['name'],'--access-config-name='+access['name']])
        self.cloud.command(['compute','instances','add-access-config',selected['name'],'--zone='+selected['zone'],
            '--network-interface='+interface['name'],'--address='+self.config['static_ip']])

    def ip_release(self):
        name = self.config['ip_name']
        ip_region = self.state.get('reserved_address_region',region(self.config['zone']))
        addresses = self.cloud.command(['compute','addresses','list','--filter=name='+name])
        if not addresses:
            if self.state.get('reserved_address_id') or self.state.get('ip_reserved_utc'):
                self.finish_ip_release('ABSENT_AFTER_EXTERNAL_CLEANUP_OR_LOST_RESPONSE')
            return dict(status='ALREADY_RELEASED')
        if len(addresses) != 1 or str(addresses[0]['id']) != self.state.get('reserved_address_id'):
            raise OperatorError('La IP no coincide con el ID reservado por este operador. No se libera; revisa el recibo.')
        if not addresses[0].get('region','').endswith('/'+ip_region):
            raise OperatorError('La IP pertenece a otra región. No se libera ni se transfiere; concilia la reubicación.')
        # Recover only from the live, identity-checked resource's server date.
        if not self.state.get('ip_reserved_utc') and addresses[0].get('creationTimestamp'):
            self.state['ip_reserved_utc'] = addresses[0]['creationTimestamp']
            self.persist()
        self.ip_charge_upper()  # Reject incomplete accounting before deletion.
        for item in [self.config['primary_vm'],*[row for row in self.state.get('alternate_vms',[]) if not row.get('disposed')]]:
            observed = self.cloud.command(['compute','instances','describe',item['name'],'--zone='+item['zone']])
            checked_vm(observed,name=item['name'],instance_id=item['id'],zone=item['zone'])
            if observed['status'] != 'TERMINATED':
                raise OperatorError('Hay una VM activa. Ejecuta stop antes de liberar la IP.')
            for interface in observed['networkInterfaces']:
                for access in interface.get('accessConfigs',[]):
                    if access.get('natIP') == addresses[0]['address']:
                        self.cloud.command(['compute','instances','delete-access-config',item['name'],'--zone='+item['zone'],
                            '--network-interface='+interface['name'],'--access-config-name='+access['name']])
        self.cloud.command(['compute','addresses','delete',name,'--region='+ip_region])
        if self.cloud.command(['compute','addresses','list','--filter=name='+name]):
            raise OperatorError('La IP sigue en la API tras delete. Conserva su ID y el recibo; no se declara liberada.')
        self.finish_ip_release('OWNER_DELETE_ABSENCE_VERIFIED')
        return dict(status='STATIC_IP_RELEASED_VERIFIED',https_idle_usd_day=0)

    def ip_charge_upper(self):
        try:
            created = datetime.fromisoformat(self.state['ip_reserved_utc'])
            now = self.now()
            if created.tzinfo is None or now.tzinfo is None or created > now:
                raise ValueError('Invalid reservation time')
        except (KeyError,TypeError,ValueError):
            raise OperatorError('Falta una fecha válida de reserva de IP. Conserva el estado y concilia su recibo antes de otra operación pagada.') from None
        seconds = (now-created).total_seconds()
        rate = self.config.get('official_rates',{}).get('unused_ip_usd_h',.01)
        return dict(elapsed_s=seconds,estimated_usd=seconds/3600*rate,rate_upper_usd_h=rate,
                    estimation_method='UNUSED_RATE_UPPER_UNTIL_OBSERVED_RELEASE',observed_utc=now.isoformat(),
                    not_invoice=True,association_intervals_not_inferred=True)

    def finish_ip_release(self, method):
        charge = self.ip_charge_upper()
        cost = self.state.setdefault('cost',dict(estimated_usd=0,margin_usd=0,reservations={}))
        cost['estimated_usd'] += charge['estimated_usd']
        cost['margin_usd'] += self.state.pop('ip_transfer_gap_margin_usd',0)
        cost['reservations'] = {key:value for key,value in cost['reservations'].items() if not key.startswith('ip-')}
        self.state['last_ip_interval'] = dict(charge,release_method=method,
            address_id=self.state.get('reserved_address_id'))
        self.state.pop('reserved_address_id',None)
        self.state.pop('reserved_address_region',None)
        self.state.pop('ip_reserved_utc',None)
        self.state.pop('ip_associated_utc',None)
        self.state.pop('ip_creation_intent',None)
        self.state.pop('ready',None)
        self.state['ready_verified'] = False
        self.config.pop('static_ip',None)
        self.config.pop('hostname',None)
        self.config.pop('prepared_snapshot',None)
        self.persist()
        save_state(self.cloud.root/'ip-release-settlement.json',dict(
            status='IP_RELEASE_RECONCILED',**self.state['last_ip_interval'],
            live_tls_state_invalidated=True,snapshot_resources_preserved=True))

    def tls_prepare(self, first_session):
        try:
            session = date.fromisoformat(first_session)
        except ValueError:
            raise OperatorError('Fecha inválida. Indica --first-session YYYY-MM-DD con al menos 3 días de anticipación.') from None
        if session < self.now().date()+timedelta(days=3):
            raise OperatorError('tls-prepare exige al menos 3 días antes de la sesión. Reprograma la preparación y conserva el certificado.')
        self.start('technical')
        began = time.monotonic()
        result = None
        try:
            while time.monotonic()-began < 900:
                try:
                    result = self.preflight()
                    self.maintenance()
                    proof = self.bridge(dict(operation='snapshot-safety'))
                    if proof.get('status') != 'ALL_I4_PERIODS_EMPTY':
                        raise OperatorError('Persisten copias del estudio. Purga todos los periodos antes de preparar contingencia.')
                    self.state['snapshot_empty_verified'] = True
                    self.persist()
                    break
                except ReadyPending:
                    self.sleep(15)
            if result is None:
                raise OperatorError('TLS no llegó a READY en 15 minutos. Conserva los recibos; revisa certificado y capacidad antes de repetir.')
        finally:
            self.stop()
        from scripts.study_operator.prepared_snapshot import prepare

        snapshot = prepare(self,self.state['ready']['tls']['certificate_sha256'])
        return dict(result,first_session=session.isoformat(),
                    certificate_prepared_days_ahead=(session-self.now().date()).days,prepared_snapshot=snapshot)

    def failover(self, zone):
        if region(zone) != region(self.selected()['zone']):
            raise OperatorError('Una IP regional no se puede transferir a otra región. La reubicación exige subred, IP, certificado y anexo propios antes de medir; usa el runbook de reubicación.')
        if self.selected()['zone'] == zone:
            return dict(status='ALREADY_SELECTED',next_action='start y preflight')
        if not self.config.get('prepared_snapshot'):
            raise OperatorError('Falta la instantánea preparada y verificada. No se crea una VM a partir de un disco sin sellar.')
        prepared = self.config['prepared_snapshot']
        purpose = self.state.get('purpose',self.config['purpose'])
        purpose_allowed(purpose,self.root)
        if (prepared.get('image_id') != self.config['image_id'] or prepared.get('hostname') != self.config['hostname']
                or prepared.get('certificate_sha256') != self.state.get('ready',{}).get('tls',{}).get('certificate_sha256')
                or prepared.get('session_data') != 'ALL_I4_PERIODS_EMPTY_VERIFIED'):
            raise OperatorError('Contingencia desactualizada para la IP, imagen o certificado. Ejecuta tls-prepare antes de la sesión.')
        self.stop()
        # Closed sessions and pending checkpoints must be reconciled before switching disks.
        if not self.state.get('failover_data_reconciled',False):
            raise OperatorError('Conmutación bloqueada por inventario de datos no conciliado. Ejecuta el inventario y recuperación indicados en el runbook.')
        instances = self.cloud.command(['compute','instances','list'])
        # A refreshed IP/certificate/image uses a new snapshot and a new standby,
        # never silently reuses an old disk containing the previous environment.
        name = 'cloudrag-i5-alternate-'+zone[-1]+'-'+prepared['id'][-12:]
        existing = [row for row in instances if row['name'] == name]
        if existing:
            alternate = dict(name=name,id=str(existing[0]['id']),zone=zone)
            checked_vm(existing[0],name=name,instance_id=alternate['id'],zone=zone)
            owned = next((row for row in self.state.get('alternate_vms',[]) if row['id'] == alternate['id']),None)
            intent = self.state.get('alternate_creation_intent',{})
            marker = owned.get('ownership_marker') if owned else intent.get('ownership_marker')
            if (not marker or existing[0].get('description') != marker
                    or (not owned and (intent.get('name') != name or intent.get('source_snapshot_id') != prepared['id']))):
                raise OperatorError('VM alterna sin prueba de creación propia. No se adopta ni se asocia la IP; revisa el intento conservado.')
            source = existing[0]['disks'][0]['source']
            disk = self.cloud.command(['compute','disks','describe',source.rsplit('/',1)[-1],'--zone='+zone])
            if disk.get('selfLink') != source or str(disk.get('sourceSnapshotId')) != prepared['id'] or disk.get('description') != marker:
                raise OperatorError('Disco alterno distinto de la instantánea propia. No se arranca ni se transfiere la IP.')
            no_other_gpu(instances,selected_id=alternate['id'])
            if not owned:
                self.state.setdefault('alternate_vms',[]).append(dict(alternate,disposable=True,
                    disk_name=disk['name'],disk_id=str(disk['id']),ownership_marker=marker,
                    created_utc=existing[0]['creationTimestamp']))
                self.persist()
            if existing[0]['status'] == 'RUNNING':
                startup = next((row['value'] for row in existing[0].get('metadata',{}).get('items',[])
                                if row['key'] == 'startup-script'),None)
                if (not intent.get('start_requested_utc') or intent.get('name') != name
                        or not startup or hashlib.sha256(startup.encode()).hexdigest() != intent.get('startup_sha256')):
                    raise OperatorError('La alterna corre sin un arranque recuperable. Verifica status y stop; no se adopta una sesión en vivo.')
                self.state.update(selected_vm=alternate,purpose=intent['purpose'],ready_verified=False,
                    boot_started_utc=intent['start_requested_utc'])
                self.state.setdefault('alternate_creation_cost_settled_ids',[]).append(alternate['id'])
                self.state.pop('alternate_creation_intent',None)
                self.persist()
                return dict(status='ALTERNATE_STARTED_SUPERVISED',zone=zone,url='https://'+self.config['hostname'],
                            next_action='preflight; capacidad conservada tras recuperar la creación')
        else:
            no_other_gpu(instances,selected_id='none')
            snapshot = self.cloud.command(['compute','snapshots','describe',self.config['prepared_snapshot']['name']])
            if (str(snapshot.get('id')) != self.config['prepared_snapshot']['id'] or snapshot.get('status') != 'READY'
                    or not snapshot.get('storageLocations') == ['us-central1']):
                raise OperatorError('Instantánea distinta o no READY. Conserva los recursos y verifica el recibo preparado.')
            self.reserve_cost('alternate-'+zone[-1],3*self.config['official_rates']['compute_usd_h']+.25+.3287664)
            # This explicit intent is recoverable even if create succeeds but its response is lost.
            intent = self.state.get('alternate_creation_intent')
            if intent and (intent.get('name') != name or intent.get('source_snapshot_id') != prepared['id']):
                raise OperatorError('Hay otro intento de contingencia pendiente. Concilia sus recursos antes de crear otra VM.')
            if not intent:
                intent = dict(name=name,zone=zone,disk_name=name+'-boot',source_snapshot_id=prepared['id'],
                    ownership_marker='CloudRAG-I5-alternate-'+uuid.uuid4().hex,
                    disposable=True,preserve_snapshot=True,observed_utc=self.now().isoformat())
            self.state['alternate_creation_intent'] = intent
            self.persist()
            disk_name = name+'-boot'
            disks = self.cloud.command(['compute','disks','list','--filter=name='+disk_name])
            if not disks:
                self.cloud.command(['compute','disks','create',disk_name,'--zone='+zone,'--size=100GB',
                    '--type=pd-balanced','--source-snapshot='+snapshot['name'],
                    '--description='+intent['ownership_marker']],timeout=600)
                disks = self.cloud.command(['compute','disks','list','--filter=name='+disk_name])
            if (len(disks) != 1 or not disks[0]['zone'].endswith('/'+zone)
                    or str(disks[0].get('sourceSnapshotId')) != str(snapshot['id'])
                    or disks[0].get('description') != intent['ownership_marker']):
                raise OperatorError('Disco alterno no coincide con la instantánea. No se recrea ni se borra.')
            disk = disks[0]
            audit_disk = dict(type='disk',name=disk_name,id=str(disk['id']),zone=zone,
                ownership_marker=intent['ownership_marker'],created_utc=disk['creationTimestamp'])
            if audit_disk not in self.state.setdefault('audit_resources',[]):
                self.state['audit_resources'].append(audit_disk)
            self.persist()  # A failed VM create must not orphan the already-paid disk.
            requested = self.now().isoformat()
            config = dict(self.config,zone=zone,purpose=purpose,start_requested_utc=requested,
                          native_deadline_utc=(self.now()+timedelta(hours=3)).isoformat())
            checked_config(config)
            from scripts.study_operator.startup import write_startup

            script = write_startup(self.cloud.root/'alternate-startup.sh',config,discover_instance=True)
            # Detach only after all source VMs are stopped. Create with the same
            # IP and startup already armed; STOP would discard acquired capacity.
            self.detach_ip_for_creation()
            intent.update(start_requested_utc=requested,purpose=purpose,
                          startup_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
            self.state['alternate_creation_intent'] = intent
            self.persist()
            self.cloud.command(['compute','instances','create',name,'--zone='+zone,'--machine-type=g2-standard-4',
                '--accelerator=type=nvidia-l4,count=1','--provisioning-model=STANDARD','--maintenance-policy=TERMINATE',
                '--disk=name='+disk_name+',boot=yes,auto-delete=no','--deletion-protection',
                '--max-run-duration=3h','--instance-termination-action=STOP',
                '--service-account='+self.config['service_account'],'--scopes=storage-rw',
                '--network='+self.config['network'],'--subnet='+self.config['subnet'],
                '--address='+self.config['static_ip'],'--metadata-from-file=startup-script='+str(script),
                '--metadata=enable-guest-attributes=TRUE',
                '--tags=cloudrag-i3-managed','--description='+intent['ownership_marker']],timeout=600)
            observed = self.cloud.command(['compute','instances','describe',name,'--zone='+zone])
            alternate = dict(name=name,id=str(observed['id']),zone=zone)
            checked_vm(observed,name=name,instance_id=alternate['id'],zone=zone)
            self.state.setdefault('alternate_vms',[]).append(dict(alternate,disposable=True,disk_name=disk_name,
                disk_id=str(disk['id']),ownership_marker=intent['ownership_marker'],created_utc=observed['creationTimestamp']))
            self.persist()
            self.state.update(selected_vm=alternate,purpose=purpose,ready_verified=False,boot_started_utc=requested)
            # The automatic create interval belongs to this uninterrupted boot;
            # stop accounts for it once, never again as a separate creation boot.
            self.state.setdefault('alternate_creation_cost_settled_ids',[]).append(alternate['id'])
            self.state.pop('alternate_creation_intent',None)
            self.persist()
            return dict(status='ALTERNATE_STARTED_SUPERVISED',zone=zone,url='https://'+self.config['hostname'],
                        next_action='preflight; no se libera la capacidad antes de READY')
        self.settle_alternate_creation(alternate)
        self.transfer_ip(alternate)
        self.state['selected_vm'] = alternate
        self.state.pop('alternate_creation_intent',None)
        self.persist()
        return dict(status='ALTERNATE_PREPARED',zone=zone,url='https://'+self.config['hostname'],
                    next_action='start con el mismo propósito; preflight verifica la identidad nueva')

    def detach_ip_for_creation(self):
        for item in [self.config['primary_vm'],*[row for row in self.state.get('alternate_vms',[]) if not row.get('disposed')]]:
            observed = self.cloud.command(['compute','instances','describe',item['name'],'--zone='+item['zone']])
            checked_vm(observed,name=item['name'],instance_id=item['id'],zone=item['zone'])
            if observed['status'] != 'TERMINATED':
                raise OperatorError('Hay una VM activa. Detén todas antes de crear la alterna con la IP.')
            for interface in observed['networkInterfaces']:
                for access in interface.get('accessConfigs',[]):
                    if access.get('natIP') == self.config['static_ip']:
                        self.cloud.command(['compute','instances','delete-access-config',item['name'],'--zone='+item['zone'],
                            '--network-interface='+interface['name'],'--access-config-name='+access['name']])
        self.state['ip_transfer_gap_margin_usd'] = self.state.get('ip_transfer_gap_margin_usd',0)+.005
        self.persist()

    def settle_alternate_creation(self, alternate):
        settled = self.state.setdefault('alternate_creation_cost_settled_ids',[])
        if alternate['id'] in settled:
            return
        observed = self.cloud.command(['compute','instances','describe',alternate['name'],'--zone='+alternate['zone']])
        checked_vm(observed,name=alternate['name'],instance_id=alternate['id'],zone=alternate['zone'])
        if observed['status'] != 'TERMINATED':
            raise OperatorError('STOP de la VM alterna no confirmado. No se transfiere la IP; verifica status.')
        try:
            seconds = (datetime.fromisoformat(observed['lastStopTimestamp'])-
                       datetime.fromisoformat(observed['lastStartTimestamp'])).total_seconds()
        except (KeyError,ValueError):
            raise OperatorError('Faltan timestamps de creación y STOP. Conserva la reserva y concilia el costo antes de usar la alterna.') from None
        if seconds < 0:
            raise OperatorError('Timestamps de cómputo inconsistentes. Conserva la reserva; no se transfiere la IP.')
        cost = self.state.setdefault('cost',dict(estimated_usd=0,margin_usd=0,reservations={}))
        estimate = seconds/3600*self.config['official_rates']['compute_usd_h']
        cost['estimated_usd'] += estimate
        cost['margin_usd'] += .25
        cost['reservations'].pop('alternate-'+alternate['zone'][-1],None)
        settled.append(alternate['id'])
        self.persist()
        save_state(self.cloud.root/('alternate-creation-cost-'+alternate['id']+'.json'),dict(
            vm_id=alternate['id'],seconds=seconds,compute_estimate_usd=estimate,invoiced=False,
            source_timestamps=dict(start=observed['lastStartTimestamp'],stop=observed['lastStopTimestamp']),
            operations_margin_usd=.25,external_ip_assigned=False))

    def transfer_ip(self, target):
        # Reserve a separate upper bound for up to an hour detached at the differential rate.
        self.state['ip_transfer_gap_margin_usd'] = self.state.get('ip_transfer_gap_margin_usd',0)+.005
        self.persist()
        for item in [self.config['primary_vm'],*[row for row in self.state.get('alternate_vms',[]) if not row.get('disposed')]]:
            observed = self.cloud.command(['compute','instances','describe',item['name'],'--zone='+item['zone']])
            checked_vm(observed,name=item['name'],instance_id=item['id'],zone=item['zone'])
            if observed['status'] != 'TERMINATED':
                raise OperatorError('Hay una VM activa. Detén todas antes de transferir la IP regional.')
            if item['id'] == target['id']:
                continue
            for interface in observed['networkInterfaces']:
                for access in interface.get('accessConfigs',[]):
                    if access.get('natIP') == self.config['static_ip']:
                        self.cloud.command(['compute','instances','delete-access-config',item['name'],'--zone='+item['zone'],
                            '--network-interface='+interface['name'],'--access-config-name='+access['name']])
        observed = self.cloud.command(['compute','instances','describe',target['name'],'--zone='+target['zone']])
        self.attach_ip(target,observed)

    def failback(self):
        if self.selected()['id'] == self.config['primary_vm']['id']:
            return dict(status='ALREADY_PRIMARY')
        self.stop()
        if not self.state.get('failover_data_reconciled',False):
            raise OperatorError('Vuelta bloqueada hasta conciliar datos. Recupera los respaldos según el runbook.')
        self.transfer_ip(self.config['primary_vm'])
        self.state['selected_vm'] = self.config['primary_vm']
        self.persist()
        return dict(status='PRIMARY_PREPARED',url='https://'+self.config['hostname'],next_action='start y preflight')

    def maintenance(self):
        if self.observed()['status'] != 'RUNNING':
            raise OperatorError('El mantenimiento necesita una VM RUNNING. Enciende con el mismo propósito, sin invitar, y ejecuta el comando.')
        purpose_allowed(self.state.get('purpose'),self.root)
        result = self.bridge(dict(operation='maintenance'))
        if not result.get('admission_closed'):
            raise OperatorError('No se verificó el cierre de admisión. No se descarga ni se borra ningún dato.')
        self.state['ready_verified'] = False
        self.persist()

    def delete_sessions(self, operation, *, code=None, dry_run=True, confirmation=None):
        from scripts.study_operator.bucket_history import collect
        from scripts.study_operator.deletion import execute, private_directory
        from scripts.study_operator.gcs import Storage
        from scripts.study_operator.policy import confirm_deletion

        purpose = self.state.get('purpose')
        purpose_allowed(purpose,self.root)
        if operation not in {'withdraw','purge-study','archive-local'}:
            raise OperatorError('Operación de mantenimiento inválida. Usa withdraw, purge-study o archive-local.')
        if operation == 'withdraw':
            participant_code(code)
        if not dry_run:
            if operation == 'archive-local':
                if purpose == 'study' and confirmation != 'ARCHIVAR':
                    raise OperatorError('Escribe ARCHIVAR en consola para retirar las copias de disco y conservar GCS.')
            else:
                confirm_deletion(operation,code,purpose,confirmation)
        self.maintenance()
        prefix = 'periods/'+self.config['period_id']+'/' + (code+'/' if code else '')
        storage = Storage(self.config['sessions_bucket'],self.cloud.owner_token,
                          creation_anchor=self.config.get('sessions_bucket_creation'),
                          policy_history=lambda metadata: collect(self.cloud.owner_token,self.config['project'],
                              self.config['sessions_bucket'],metadata))
        scope = hashlib.sha256((operation+'\0'+self.config['sessions_bucket']+'\0'+prefix).encode()).hexdigest()
        transactions = self.state.setdefault('deletion_transactions',{})
        transaction = transactions.get(scope)
        if not transaction:
            transaction = uuid.uuid4().hex
            if not dry_run:
                transactions[scope] = transaction
                self.persist()
        if not re.fullmatch('[a-f0-9]{32}',transaction):
            raise OperatorError('Recibo de borrado no válido. Conserva las descargas y revisa active.json.')
        local = private_directory(self.root/'private'/'deletion'/scope/transaction)
        saved = local/'disk-plan.json'
        if saved.exists():
            plan = json.loads(saved.read_text(encoding='utf-8'))
        else:
            plan = self.bridge(dict(operation='inventory',code=code,synthetic_only=purpose!='study'),private=True)['plan']
            if not dry_run:
                save_state(saved,plan)
        if dry_run:
            remote = execute(storage,prefix,local,dry_run=True)
            return dict(status='DRY_RUN',disk_files=len(plan['files']),session_count=plan['session_count'],
                        remote_objects=len(remote['objects']),deletion_count=0,admission_closed=True)
        receipts_path = local/'disk-downloads.json'
        if receipts_path.exists():
            verified = json.loads(receipts_path.read_text(encoding='utf-8'))
            if any(not Path(row['local_path']).is_file() or hashlib.sha256(Path(row['local_path']).read_bytes()).hexdigest() != row['sha256'] for row in verified):
                raise OperatorError('Descarga local del disco alterada. No se continúa el borrado; recupera la copia verificada.')
        else:
            files = self.bridge(dict(operation='download-disk',plan=plan),private=True)['files']
            verified = []
            for row in files:
                data = base64.b64decode(row.pop('content_base64'),validate=True)
                if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                    raise OperatorError('SHA-256 descargado del disco distinto. Conserva todas las copias; no se borra.')
                path = local/'disk'/row['store_id']/row['kind']/row['relative']
                if not path.resolve().is_relative_to(local):
                    raise OperatorError('Descarga fuera del directorio privado. No se borra.')
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                    raise OperatorError('Copia local previa distinta. Conserva ambas; no se borra.')
                if not path.exists():
                    with path.open('xb') as stream:
                        stream.write(data)
                        stream.flush()
                        import os

                        os.fsync(stream.fileno())
                verified.append(dict(row,local_path=str(path)))
            if any((row['store_id'],row['path'],row['sha256']) not in {
                    (item['store_id'],item['path'],item['sha256']) for item in verified} for row in plan['files']):
                raise OperatorError('Faltan copias locales del inventario. No se borra ninguna copia remota.')
            save_state(receipts_path,verified)

        def disk_cleanup(downloads):
            if operation == 'archive-local':
                from scripts.study_operator.archive_local import verified_closed_copies

                verified_closed_copies(verified,downloads)
            result = self.bridge(dict(operation='clean-disk',plan=plan,verified_downloads=verified),private=True)['result']
            return result

        receipt = execute(storage,prefix,local,dry_run=False,disk_cleanup=disk_cleanup,
                          retain_remote=operation=='archive-local')
        if not code:
            self.state['failover_data_reconciled'] = True
            self.persist()
        summary = dict(status=receipt['status'],remote_versions_empty=receipt.get('remote_versions_empty',False),
            zero_retention_policy_verified=receipt.get('zero_retention_policy_verified',False),
            normal_objects_empty=receipt.get('normal_objects_empty',False),disk_empty=receipt['disk']['empty'],
            downloaded_disk_files=len(verified),deleted_generations=receipt['deleted'],
            private_receipt=str(local/'receipt.json'),policy_verification=storage.policy_verification)
        if operation == 'archive-local':
            summary['remote_objects_retained'] = receipt['remote_objects_retained']
        receipt['policy_verification'] = storage.policy_verification
        save_state(local/'receipt.json',receipt)
        self.state['deletion_transactions'].pop(scope,None)
        self.persist()
        return summary

    def archive_local(self, *, dry_run=True, confirmation=None):
        return self.delete_sessions('archive-local',dry_run=dry_run,confirmation=confirmation)

    def export_anonymized(self):
        self.maintenance()
        from scripts.study_operator.deletion import private_directory

        result = self.bridge(dict(operation='export'),private=True)
        root = private_directory(self.root/'private'/'exports'/self.now().strftime('%Y%m%dT%H%M%S%fZ'))
        path = root/'coded-review-required.json'
        save_state(path,result['export'])
        return dict(status='PSEUDONYMOUS_EXPORT_PRIVATE',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    manual_free_text_review_required=True,automatic_publication_allowed=False)

    def restore(self, code, session_id, *, full_generation, manifest_generation):
        from scripts.study_operator.gcs import Storage

        participant_code(code)
        if not re.fullmatch('[a-f0-9]{32}',session_id) or not str(full_generation).isdigit() or not str(manifest_generation).isdigit():
            raise OperatorError('ID o generaciones inválidos. Usa el inventario del respaldo verificado.')
        self.maintenance()
        storage = Storage(self.config['sessions_bucket'],self.cloud.owner_token)
        prefix = 'periods/'+self.config['period_id']+'/'+code+'/'+session_id+'/'
        full = storage.read(prefix+'full_session.json',full_generation)
        manifest = storage.read(prefix+'export_manifest.json',manifest_generation)
        objects = {name:dict(object=prefix+name,generation=str(generation),sha256=hashlib.sha256(data).hexdigest())
                   for name,data,generation in [('full_session.json',full,full_generation),
                       ('export_manifest.json',manifest,manifest_generation)]}
        if json.loads(full)['assignment']['participant_id'] != code:
            raise OperatorError('El respaldo corresponde a otro código. No se restaura.')
        return self.bridge(dict(operation='restore',full_session_base64=base64.b64encode(full).decode(),
            manifest_base64=base64.b64encode(manifest).decode(),objects=objects),private=True)
