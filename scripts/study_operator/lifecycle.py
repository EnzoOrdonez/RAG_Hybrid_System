"""Idempotent owner operations for the prepared installation, with receipts."""
import base64
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import secrets
import shlex
import socket
import ssl
import time
import uuid

from scripts.study_operator.cloud_client import checked_vm, no_other_gpu, readiness
from scripts.study_operator.deployment import checked_config
from scripts.study_operator.policy import OperatorError, participant_code, purpose_allowed, session_margin
from scripts.study_operator.service_gateway import save_state


class Operator:
    def __init__(self, root, cloud, *, now=None, sleep=time.sleep):
        self.root, self.cloud, self.sleep = Path(root), cloud, sleep
        self.now = now or (lambda: datetime.now(timezone.utc))
        try:
            self.config = json.loads((self.root/'installation.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):
            raise OperatorError('Falta installation.json válido. Usa la instalación sellada de iteración 4; no copies el operador anterior.') from None
        self.state_path = self.root/'active.json'
        self.state = json.loads(self.state_path.read_text(encoding='utf-8')) if self.state_path.exists() else {}

    def persist(self):
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
            if result.get('status') == 'ERROR':
                raise ValueError('remote refusal')
            return result
        except (TypeError,ValueError):
            raise OperatorError('El invitado rechazó la operación o aún no responde. Conserva el recibo, ejecuta status y revisa preflight.') from None

    def status(self):
        observed = self.observed()
        return dict(status=observed['status'],vm_id=str(observed['id']),zone=observed['zone'].split('/')[-1],
                    retained_disk=True,deletion_protection=True,native_stop_s=10800)

    def reserve_cost(self, operation, amount):
        cost = self.state.setdefault('cost',dict(estimated_usd=0,margin_usd=0,reservations={}))
        inherited = self.config.get('cost',{})
        total = (inherited.get('estimated_usd',0)+inherited.get('margin_usd',0)+cost['estimated_usd']+
                 cost['margin_usd']+sum(cost['reservations'].values())+amount)
        if not 0 <= total < 90:
            raise OperatorError('El costo reservado alcanza el corte de USD90. Mantén apagada la VM y concilia el ledger.')
        cost['reservations'][operation] = amount
        self.persist()
        save_state(self.cloud.root/('cost-'+operation+'.json'),dict(status='RESERVED_BEFORE_EFFECT',
            estimate_usd=amount,conservative_total_usd=total,official_rates=self.config.get('official_rates'),
            observed_utc=self.now().isoformat()))

    def start(self, purpose):
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
                      native_deadline_utc=(self.now()+timedelta(hours=3)).isoformat())
        checked_config(config)
        self.reserve_cost('boot-'+self.now().strftime('%Y%m%dT%H%M%S%fZ'),
            3*self.config['official_rates']['compute_usd_h']+.25)
        self.state.update(purpose=purpose,boot_started_utc=self.now().isoformat(),ready_verified=False)
        self.persist()
        script = self.cloud.root/'startup.sh'
        encoded = base64.b64encode(json.dumps(config).encode()).decode()
        script.write_text('#!/bin/bash\nset -euo pipefail\npython3 - <<\'PY\'\n'
            'import base64,json,subprocess,os\nfrom pathlib import Path\n'
            'root=Path("/srv/cloudrag/iteration4");root.mkdir(exist_ok=True)\n'
            'p=root/"launch-config.json"\np.write_bytes(base64.b64decode('+repr(encoded)+'))\nos.chmod(p,0o600)\n'
            'c=json.loads(p.read_text())\nsubprocess.Popen(["python3","-B","-m","scripts.study_operator.host_runtime",'
            '"--settings",str(p)],cwd=c["host_code"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\n'
            'PY\n',encoding='utf-8',newline='\n')
        selected = self.selected()
        self.cloud.command(['compute','instances','add-metadata',selected['name'],'--zone='+selected['zone'],
            '--metadata-from-file=startup-script='+str(script)])
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
            self.state['cost']['reservations'] = {key:value for key,value in self.state['cost']['reservations'].items()
                                                 if key.startswith('ip-')}
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
        addresses = self.cloud.command(['compute','addresses','list','--filter=name='+name])
        if not addresses:
            self.reserve_cost('ip-'+self.now().strftime('%Y%m%dT%H%M%S%fZ'),.01*24*3)
            self.cloud.command(['compute','addresses','create',name,'--region=us-central1','--ip-version=IPV4'])
        observed = self.cloud.command(['compute','addresses','describe',name,'--region=us-central1'])
        if not observed.get('region','').endswith('/us-central1') or observed.get('addressType') != 'EXTERNAL':
            raise OperatorError('La IP no es externa regional en us-central1. Conserva el recurso y revisa su recibo.')
        self.config.update(static_ip=observed['address'],hostname=observed['address']+'.sslip.io')
        selected = self.selected()
        vm = self.observed()
        if vm['status'] != 'TERMINATED':
            raise OperatorError('La IP está reservada, pero la VM corre. Ejecuta stop antes de asociarla.')
        self.attach_ip(selected,vm)
        self.state['reserved_address_id'] = str(observed['id'])
        self.state.setdefault('ip_reserved_utc',observed.get('creationTimestamp',self.now().isoformat()))
        self.state.setdefault('ip_associated_utc',self.now().isoformat())
        self.persist()
        return dict(status='STATIC_IP_RESERVED',url='https://'+self.config['hostname'],idle_usd_day=.12,
                    unused_usd_day=.24,address_id=str(observed['id']))

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
        addresses = self.cloud.command(['compute','addresses','list','--filter=name='+name])
        if not addresses:
            return dict(status='ALREADY_RELEASED')
        if len(addresses) != 1 or str(addresses[0]['id']) != self.state.get('reserved_address_id'):
            raise OperatorError('La IP no coincide con el ID reservado por este operador. No se libera; revisa el recibo.')
        for item in [self.config['primary_vm'],*self.state.get('alternate_vms',[])]:
            observed = self.cloud.command(['compute','instances','describe',item['name'],'--zone='+item['zone']])
            checked_vm(observed,name=item['name'],instance_id=item['id'],zone=item['zone'])
            if observed['status'] != 'TERMINATED':
                raise OperatorError('Hay una VM activa. Ejecuta stop antes de liberar la IP.')
            for interface in observed['networkInterfaces']:
                for access in interface.get('accessConfigs',[]):
                    if access.get('natIP') == addresses[0]['address']:
                        self.cloud.command(['compute','instances','delete-access-config',item['name'],'--zone='+item['zone'],
                            '--network-interface='+interface['name'],'--access-config-name='+access['name']])
        self.cloud.command(['compute','addresses','delete',name,'--region=us-central1'])
        self.state.pop('reserved_address_id',None)
        if self.state.get('ip_reserved_utc'):
            created = datetime.fromisoformat(self.state.pop('ip_reserved_utc'))
            associated = datetime.fromisoformat(self.state.pop('ip_associated_utc',created.isoformat()))
            unused_s = max(0,(associated-created).total_seconds())
            associated_s = max(0,(self.now()-associated).total_seconds())
            cost = self.state.setdefault('cost',dict(estimated_usd=0,margin_usd=0,reservations={}))
            cost['estimated_usd'] += (unused_s*.01+associated_s*.005)/3600
            # Rate uncertainty during short transfer gaps is a margin, not compute spend.
            cost['margin_usd'] += self.state.get('ip_transfer_gap_margin_usd',0)
            cost['reservations'] = {key:value for key,value in cost['reservations'].items() if not key.startswith('ip-')}
            self.state['last_ip_interval'] = dict(unused_s=unused_s,associated_s=associated_s,
                estimated_usd=(unused_s*.01+associated_s*.005)/3600)
        self.config.pop('static_ip',None)
        self.config.pop('hostname',None)
        self.persist()
        return dict(status='STATIC_IP_RELEASED_VERIFIED',https_idle_usd_day=0)

    def tls_prepare(self, first_session):
        try:
            session = date.fromisoformat(first_session)
        except ValueError:
            raise OperatorError('Fecha inválida. Indica --first-session YYYY-MM-DD con al menos 3 días de anticipación.') from None
        if session < self.now().date()+timedelta(days=3):
            raise OperatorError('tls-prepare exige al menos 3 días antes de la sesión. Reprograma la preparación y conserva el certificado.')
        self.start('technical')
        began = time.monotonic()
        try:
            while time.monotonic()-began < 900:
                try:
                    result = self.preflight()
                    return dict(result,first_session=session.isoformat(),certificate_prepared_days_ahead=(session-self.now().date()).days)
                except OperatorError:
                    self.sleep(15)
            raise OperatorError('TLS no llegó a READY en 15 minutos. Conserva los recibos; revisa certificado y capacidad antes de repetir.')
        finally:
            self.stop()

    def failover(self, zone):
        if zone not in {'us-central1-b','us-central1-c'}:
            raise OperatorError('Zona alterna inválida. Usa us-central1-b o us-central1-c.')
        if self.selected()['zone'] == zone:
            return dict(status='ALREADY_SELECTED',next_action='start y preflight')
        if not self.config.get('prepared_snapshot'):
            raise OperatorError('Falta la instantánea preparada y verificada. No se crea una VM a partir de un disco sin sellar.')
        self.stop()
        # Closed sessions and pending checkpoints must be reconciled before switching disks.
        if not self.state.get('failover_data_reconciled',False):
            raise OperatorError('Conmutación bloqueada por inventario de datos no conciliado. Ejecuta el inventario y recuperación indicados en el runbook.')
        instances = self.cloud.command(['compute','instances','list'])
        no_other_gpu(instances,selected_id='none')
        name = 'cloudrag-i4-alternate-'+zone[-1]+'-'+self.config['period_id'][:8]
        existing = [row for row in instances if row['name'] == name]
        if existing:
            alternate = dict(name=name,id=str(existing[0]['id']),zone=zone)
            checked_vm(existing[0],name=name,instance_id=alternate['id'],zone=zone)
        else:
            snapshot = self.cloud.command(['compute','snapshots','describe',self.config['prepared_snapshot']['name']])
            if (str(snapshot.get('id')) != self.config['prepared_snapshot']['id'] or snapshot.get('status') != 'READY'
                    or not snapshot.get('storageLocations') == ['us-central1']):
                raise OperatorError('Instantánea distinta o no READY. Conserva los recursos y verifica el recibo preparado.')
            self.reserve_cost('alternate-'+zone[-1],3*self.config['official_rates']['compute_usd_h']+.25+.3287664)
            # This explicit intent is recoverable even if create succeeds but its response is lost.
            self.state['alternate_creation_intent'] = dict(name=name,zone=zone,disk_name=name+'-boot',
                disposable=True,preserve_snapshot=True,observed_utc=self.now().isoformat())
            self.persist()
            disk_name = name+'-boot'
            disks = self.cloud.command(['compute','disks','list','--filter=name='+disk_name])
            if not disks:
                self.cloud.command(['compute','disks','create',disk_name,'--zone='+zone,'--size=100GB',
                    '--type=pd-balanced','--source-snapshot='+snapshot['name']],timeout=600)
            else:
                if len(disks) != 1 or not disks[0]['zone'].endswith('/'+zone) or str(disks[0].get('sourceSnapshotId')) != str(snapshot['id']):
                    raise OperatorError('Disco alterno no coincide con la instantánea. No se recrea ni se borra.')
            self.cloud.command(['compute','instances','create',name,'--zone='+zone,'--machine-type=g2-standard-4',
                '--accelerator=type=nvidia-l4,count=1','--provisioning-model=STANDARD','--maintenance-policy=TERMINATE',
                '--disk=name='+disk_name+',boot=yes,auto-delete=no','--deletion-protection',
                '--max-run-duration=3h','--instance-termination-action=STOP',
                '--service-account='+self.config['service_account'],'--scopes=storage-rw',
                '--network='+self.config['network'],'--subnet='+self.config['subnet'],'--no-address',
                '--tags=cloudrag-i3-managed'],timeout=600)
            observed = self.cloud.command(['compute','instances','describe',name,'--zone='+zone])
            alternate = dict(name=name,id=str(observed['id']),zone=zone)
            checked_vm(observed,name=name,instance_id=alternate['id'],zone=zone)
            self.state.setdefault('alternate_vms',[]).append(dict(alternate,disposable=True,disk_name=disk_name))
            self.persist()
            self.cloud.command(['compute','instances','stop',name,'--zone='+zone],timeout=600)
        self.transfer_ip(alternate)
        self.state['selected_vm'] = alternate
        self.state.pop('alternate_creation_intent',None)
        self.persist()
        return dict(status='ALTERNATE_PREPARED',zone=zone,url='https://'+self.config['hostname'],
                    next_action='start con el mismo propósito; preflight verifica la identidad nueva')

    def transfer_ip(self, target):
        # Reserve a separate upper bound for up to an hour detached at the differential rate.
        self.state['ip_transfer_gap_margin_usd'] = self.state.get('ip_transfer_gap_margin_usd',0)+.005
        self.persist()
        for item in [self.config['primary_vm'],*self.state.get('alternate_vms',[])]:
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
        from scripts.study_operator.deletion import execute, private_directory
        from scripts.study_operator.gcs import Storage
        from scripts.study_operator.policy import confirm_deletion

        purpose = self.state.get('purpose')
        purpose_allowed(purpose,self.root)
        if operation not in {'withdraw','purge-study'}:
            raise OperatorError('Operación de borrado inválida. Usa withdraw o purge-study.')
        if operation == 'withdraw':
            participant_code(code)
        if not dry_run:
            confirm_deletion(operation,code,purpose,confirmation)
        self.maintenance()
        prefix = 'periods/'+self.config['period_id']+'/' + (code+'/' if code else '')
        storage = Storage(self.config['sessions_bucket'],self.cloud.owner_token)
        scope = hashlib.sha256((self.config['sessions_bucket']+'\0'+prefix).encode()).hexdigest()
        local = private_directory(self.root/'private'/'deletion'/scope)
        saved = local/'disk-plan.json'
        if saved.exists():
            plan = json.loads(saved.read_text(encoding='utf-8'))
        else:
            plan = self.bridge(dict(operation='inventory',code=code,synthetic_only=purpose!='study'),private=True)['plan']
            if not dry_run:
                save_state(saved,plan)
        if dry_run:
            remote = execute(storage,prefix,local,dry_run=True)
            return dict(status='DRY_RUN',disk_files=len(plan['files']),session_count=len(plan['session_codes']),
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
                path = local/'disk'/row['kind']/row['relative']
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
            if any((row['path'],row['sha256']) not in {(item['path'],item['sha256']) for item in verified} for row in plan['files']):
                raise OperatorError('Faltan copias locales del inventario. No se borra ninguna copia remota.')
            save_state(receipts_path,verified)

        def disk_cleanup(downloads):
            result = self.bridge(dict(operation='clean-disk',plan=plan,verified_downloads=verified),private=True)['result']
            return result

        receipt = execute(storage,prefix,local,dry_run=False,disk_cleanup=disk_cleanup)
        if not code:
            self.state['failover_data_reconciled'] = True
            self.persist()
        summary = dict(status=receipt['status'],remote_versions_empty=receipt['remote_versions_empty'],
            remote_soft_deleted_empty=receipt['remote_soft_deleted_empty'],disk_empty=receipt['disk']['empty'],
            downloaded_disk_files=len(verified),deleted_generations=receipt['deleted'],
            private_receipt=str(local/'receipt.json'))
        save_state(local/'receipt.json',receipt)
        return summary

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
        import re

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
