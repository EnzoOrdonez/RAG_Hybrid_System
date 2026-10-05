"""Bounded cloud host service. Only fixed technical receipts are logged."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import time
import urllib.request

from scripts.study_operator.deployment import app_command, assert_isolation, bind, caddyfile, checked_config
from scripts.study_operator.service_gateway import save_state


ROOT = Path('/srv/cloudrag/iteration4')


def metadata(field):
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/' + field,
        headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.read().decode()


def certificate(hostname):
    with socket.create_connection(('127.0.0.1', 443), timeout=5) as connection:
        with ssl.create_default_context().wrap_socket(connection, server_hostname=hostname) as secure:
            return dict(certificate_sha256=hashlib.sha256(secure.getpeercert(binary_form=True)).hexdigest(),
                        tls_version=secure.version(), chain_verified=True)


def metadata_unreachable(container, *, invoke=subprocess.run):
    # Both the literal link-local route and DNS hostname are checked inside the actual app.
    code = '''import json,urllib.request,urllib.error
result={}
for target in ('169.254.169.254','metadata.google.internal'):
 try:
  with urllib.request.urlopen(urllib.request.Request('http://'+target+'/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'}),timeout=2) as response: response.read(64)
  result[target]=True
 except urllib.error.HTTPError: result[target]=True
 except OSError: result[target]=False
print(json.dumps(result))
'''
    result = invoke(['docker','exec',container,'python','-c',code], capture_output=True, timeout=15)
    if result.returncode or json.loads(result.stdout) != {'169.254.169.254':False,'metadata.google.internal':False}:
        raise ValueError('METADATA_ISOLATION_NOT_VERIFIED')
    return {'metadata_unreachable':True, 'targets_checked':2}


class Host:
    def __init__(self, config, *, invoke=subprocess.run):
        self.config = checked_config(config)
        self.invoke = invoke
        self.boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.root = ROOT / 'boots' / self.boot
        self.root.mkdir(parents=True, exist_ok=False)
        self.began = time.monotonic()
        self.started = datetime.now(timezone.utc)
        self.children, self.containers = [], []
        self.sequence = 0
        self.session_root = ROOT / 'periods' / config['period_id'] / 'sessions'

    def step(self, argv, *, timeout=60, capture=False):
        self.sequence += 1
        began = time.monotonic()
        at = datetime.now(timezone.utc).isoformat()
        result = self.invoke(argv, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, timeout=timeout)
        save_state(self.root / f'command-{self.sequence:04d}.json',
            dict(command=argv, started_utc=at, duration_s=time.monotonic()-began, exit_code=result.returncode,
                 output_policy='DISCARDED_OR_MEMORY_ONLY'))
        if result.returncode:
            raise ValueError('HOST_COMMAND_FAILED')
        return result.stdout if capture else None

    def child(self, module, settings):
        child = subprocess.Popen([sys.executable,'-B','-m',module,'--settings',str(settings)],
            cwd=self.config['host_code'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1'))
        self.children.append(child)
        return child

    def container(self, argv, name, *, detached=True):
        self.containers.append(name)  # Register before launch so failures are still cleaned up.
        command = [*argv[:2], *(['-d'] if detached else []), *argv[2:]]
        return self.step(command, timeout=300, capture=detached)

    def prepare(self):
        for directory in ('meta','config','sockets','web','caddy-config'):
            path = self.root / directory
            path.mkdir(mode=0o750)
            os.chown(path, 10001 if directory in {'meta','web'} else 0, 10001)
        self.session_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        marker = self.session_root / '_i4_root.json'
        expected = dict(schema_version=1,purpose=self.config['purpose'])
        if marker.exists() and json.loads(marker.read_text()) != expected:
            raise ValueError('SESSION_PURPOSE_CHANGED')
        if not marker.exists():
            save_state(marker, expected)
        os.chown(self.session_root,10001,10001)
        for path in Path(self.config['reviewed_config']).iterdir():
            if path.is_file():
                shutil.copyfile(path,self.root/'config'/path.name)
        for path in (self.root/'config').iterdir():
            os.chmod(path,0o640)
            os.chown(path,0,10001)
        self.step(['systemd-run','--unit=cloudrag-i4-stop-'+self.boot,'--on-active=165m',
                   '/sbin/shutdown','-h','now'], timeout=30)
        # Retained containers from previous iterations are stopped, never removed.
        names = self.step(['docker','ps','--format','{{.Names}}'],capture=True).decode().splitlines()
        for name in names:
            if name.startswith(('cloudrag-i3-','cloudrag-i4-')):
                self.step(['docker','stop','--time','15',name],timeout=30)
        self.ollama = 'cloudrag-i4-ollama-'+self.boot
        self.container(['docker','run','--rm=false','--name',self.ollama,'--network','host',
            '--log-driver','none','--gpus','all','-e','OLLAMA_HOST=127.0.0.1:11434',
            *bind(self.config['ollama_models'],'/root/.ollama/models'), self.config['ollama_image']],self.ollama)
        for _ in range(60):
            try:
                with urllib.request.urlopen('http://127.0.0.1:11434/api/version',timeout=2) as response:
                    if json.load(response)['version'] != self.config['ollama_version']:
                        raise ValueError('OLLAMA_VERSION_CHANGED')
                break
            except OSError:
                time.sleep(1)
        else:
            raise ValueError('OLLAMA_START_TIMEOUT')
        runtime = dict(schema_version=1,boot_id=self.boot,observed_utc=datetime.now(timezone.utc).isoformat(),
            instance={field:metadata(field) for field in ('id','zone','machine-type')})
        if (runtime['instance']['id'] != str(self.config['instance_id'])
                or runtime['instance']['zone'].split('/')[-1] != self.config['zone']
                or runtime['instance']['machine-type'].split('/')[-1] != 'g2-standard-4'
                or metadata('network-interfaces/0/access-configs/0/external-ip') != self.config['static_ip']):
            raise ValueError('CLOUD_IDENTITY_CHANGED')
        save_state(self.root/'meta'/'host-runtime.json',runtime)
        image = json.loads(self.step(['docker','image','inspect',self.config['image_id']],capture=True))[0]
        if image['Id'] != self.config['image_id']:
            raise ValueError('IMAGE_CHANGED')
        save_state(self.root/'meta'/'image-receipt.json',dict(image_id=image['Id'],container_image_id=image['Id']))
        policy = dict(schema_version=1,mode='fresh_runner',generation_options=dict(
            temperature=0,num_predict=1024,seed=42,num_ctx=4096),reset_deadline_s=10,
            load_deadline_s=30,call_deadline_s=600,scope='every content query, both conditions')
        save_state(self.root/'meta'/'service-policy.json',policy)
        policy_sha = hashlib.sha256((self.root/'meta'/'service-policy.json').read_bytes()).hexdigest()
        shutil.copyfile(self.config['artifact_manifest'],self.root/'meta'/'deployment-artifacts.json')
        seed = dict(cloud=True,build_id=self.config['commit'],model_digest=self.config['model_digest'],
            ollama_version=self.config['ollama_version'],artifact_manifest='/deployment/deployment-artifacts.json',
            artifact_manifest_sha256=hashlib.sha256((self.root/'meta'/'deployment-artifacts.json').read_bytes()).hexdigest(),
            config_dir='/reviewed',session_root='/sessions',purpose=self.config['purpose'],device_gpu='1',
            bucket=self.config['sessions_bucket'],backup_prefix='periods/'+self.config['period_id'],
            backup_socket='/service/backup.sock',fingerprint=self.config['fingerprint'],
            runtime_image_receipt='/deployment/image-receipt.json',vendor_manifest='/opt/cloudrag/vendor-manifest.json',
            host_runtime_receipt='/deployment/host-runtime.json',zone=self.config['zone'],machine_type='g2-standard-4',
            instance_id=str(self.config['instance_id']),service_mode='fresh_runner',service_boot_id=self.boot,
            service_policy_receipt='/deployment/service-policy.json',service_policy_sha256=policy_sha,
            service_state_path='/service/service-state.json',hostname=self.config['hostname'])
        save_state(self.root/'meta'/'deployment-seed.json',seed)
        for path in (self.root/'meta').iterdir():
            os.chmod(path,0o640)
            os.chown(path,10001,10001)
        generation = dict(socket=str(self.root/'sockets'/'generation.sock'),ollama_container=self.ollama,
            model='granite4.1:8b',digest=self.config['model_digest'],state_path=str(self.root/'sockets'/'service-state.json'))
        save_state(self.root/'generation-settings.json',generation)
        self.child('scripts.study_operator.service_gateway',self.root/'generation-settings.json')
        inventory = self.session_root.parent/'private-inventory'
        inventory.mkdir(exist_ok=True,mode=0o700)
        os.chown(inventory,10001,10001)
        seed['private_inventory'] = '/private-inventory'
        save_state(self.root/'meta'/'deployment-seed.json',seed)
        os.chmod(self.root/'meta'/'deployment-seed.json',0o640)
        os.chown(self.root/'meta'/'deployment-seed.json',10001,10001)
        settings = dict(socket=str(self.root/'sockets'/'backup.sock'), sessions_bucket=self.config['sessions_bucket'],
            prefix='periods/'+self.config['period_id'],sessions_root=str(self.session_root),purpose=self.config['purpose'],
            private_inventory_root=str(inventory))
        save_state(self.root/'backup-settings.json',settings)
        self.child('scripts.study_operator.backup_agent',self.root/'backup-settings.json')
        for _ in range(50):
            if all((self.root/'sockets'/name).exists() for name in ('generation.sock','backup.sock','service-state.json')):
                break
            if any(child.poll() is not None for child in self.children):
                raise ValueError('PRIVATE_AGENT_FAILED')
            time.sleep(.1)
        else:
            raise ValueError('PRIVATE_AGENT_TIMEOUT')
        for path in (self.root/'sockets').iterdir():
            os.chown(path,0,10001)
            os.chmod(path,0o660 if path.suffix == '.sock' else 0o640)
        freeze_name = 'cloudrag-i4-freeze-'+self.boot
        self.container(app_command(self.config,self.root,self.session_root,freeze_name,
            operation='freeze',output='/deployment/deployment.json'),freeze_name,detached=False)
        # The freezing container actually ran the image; verify its image at the host boundary.
        if json.loads(self.step(['docker','inspect',freeze_name],capture=True))[0]['Image'] != image['Id']:
            raise ValueError('FREEZE_IMAGE_CHANGED')
        self.app = 'cloudrag-i4-app-'+self.boot
        self.container(app_command(self.config,self.root,self.session_root,self.app),self.app)
        assert_isolation(json.loads(self.step(['docker','inspect',self.app],capture=True))[0],image['Id'])
        metadata_unreachable(self.app)
        save_state(ROOT/'active.json',dict(schema_version=1,boot_id=self.boot,boot_root=str(self.root),
            session_root=str(self.session_root),app_container=self.app,ollama_container=self.ollama,
            config=self.config,guest_deadline_utc=(self.started+timedelta(minutes=165)).isoformat(),
            native_deadline_utc=self.config['native_deadline_utc']))
        (self.root/'caddy-config'/'Caddyfile').write_text(caddyfile(self.config['hostname']),encoding='utf-8')
        tls_root = ROOT/'tls'
        tls_root.mkdir(exist_ok=True,mode=0o700)
        base = ['docker','run','--rm=false','--log-driver','none',
            *bind(self.root/'caddy-config','/etc/caddy'),*bind(tls_root,'/data',False),
            *bind(self.root/'web','/web'),self.config['caddy_image']]
        self.step([*base,'caddy','validate','--config','/etc/caddy/Caddyfile','--adapter','caddyfile'],timeout=60)
        self.caddy = 'cloudrag-i4-caddy-'+self.boot
        self.container(['docker','run','--rm=false','--name',self.caddy,'--network','host','--log-driver','none',
            *bind(self.root/'caddy-config','/etc/caddy'),*bind(tls_root,'/data',False),*bind(self.root/'web','/web'),
            self.config['caddy_image'],'caddy','run','--config','/etc/caddy/Caddyfile','--adapter','caddyfile'],self.caddy)
        while time.monotonic()-self.began < 900:
            try:
                tls = certificate(self.config['hostname'])
                with urllib.request.urlopen('https://'+self.config['hostname']+'/_stcore/health',timeout=5) as response:
                    if response.read() != b'ok':
                        raise ValueError('APP_NOT_HEALTHY')
                settings = json.loads((self.root/'meta'/'deployment.json').read_text())
                self.step(['docker','exec',self.app,'python','scripts/cloud_entrypoint.py','verify',
                           '--deployment','/deployment/deployment.json'],timeout=120)
                self.step(['docker','exec',self.app,'python','-c',
                    "import json; from scripts.cloud_entrypoint import configure,verify; "
                    "from src.ui.components.study_sessions import StudyStore; "
                    "d=json.load(open('/deployment/deployment.json')); configure(d); "
                    "StudyStore(d['session_root'],verify(d),d['purpose']).check_backups()"],timeout=120)
                if time.monotonic()-self.began > 900:
                    raise ValueError('READY_DEADLINE_900S')
                save_state(self.root/'ready.json',dict(status='READY',image_id=image['Id'],
                    url='https://'+self.config['hostname'],boot_id=self.boot,identity_verified=True,
                    environment_identity_sha256=settings['environment_identity_sha256'],
                    metadata_unreachable=True,backup_clear=True,tls_verified=True,tls=tls,
                    start_to_ready_s=time.monotonic()-self.began,ready_utc=datetime.now(timezone.utc).isoformat()))
                return
            except (OSError,ValueError):
                time.sleep(2)
        raise ValueError('READY_DEADLINE_900S')

    def run(self):
        try:
            self.prepare()
            while time.monotonic()-self.began < 165*60:
                if (self.root/'stop-request.json').exists():
                    break
                if any(child.poll() is not None for child in self.children):
                    raise ValueError('PRIVATE_AGENT_EXITED')
                time.sleep(2)
        except BaseException as error:
            save_state(self.root/'failure.json',dict(status='FAILED',error_type=type(error).__name__,
                reason=str(error) if isinstance(error,ValueError) and re_safe_reason(str(error)) else 'HOST_FAILED',
                boot_id=self.boot,elapsed_s=time.monotonic()-self.began))
        finally:
            failed_cleanup = 0
            for name in reversed(self.containers):
                try:
                    result = subprocess.run(['docker','stop','--time','15',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
                    failed_cleanup += result.returncode != 0
                except (OSError,subprocess.TimeoutExpired):
                    failed_cleanup += 1
            for child in self.children:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
            save_state(self.root/'stopped.json',dict(status='SERVICES_STOPPED' if not failed_cleanup else 'CLEANUP_ERRORS',
                container_stop_errors=failed_cleanup,boot_id=self.boot,
                elapsed_s=time.monotonic()-self.began,stopped_utc=datetime.now(timezone.utc).isoformat()))
            subprocess.run(['/sbin/shutdown','-h','now'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)


def re_safe_reason(value):
    return value.isascii() and value.replace('_','').isalnum() and value.upper() == value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings',required=True)
    arguments = parser.parse_args(argv)
    config = json.loads(Path(arguments.settings).read_text(encoding='utf-8'))
    Host(config).run()


if __name__ == '__main__':
    main()
