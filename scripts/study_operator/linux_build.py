"""Guest-only offline cached build and Linux checks, with independent hard STOP.

Only technical build evidence is uploaded; never copy a deployment/session tree.
The owner must download each generation and verify its SHA256 before acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request


ASSETS = '/srv/cloudrag/iteration3/i3-r1-build01/assets'


def persist(path,value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    fd = os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def metadata(field):
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/'+field,
                                     headers={'Metadata-Flavor':'Google'})
    with urllib.request.urlopen(request,timeout=5) as response:
        return response.read()


def stop():
    subprocess.run(['/usr/sbin/shutdown','-h','now'],stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL,timeout=15,check=True)


def check_commands(image,root,label):
    root = Path(root)
    base = ['docker','run','--rm=false','--log-driver','none','--network','none',
        '-e','TMPDIR=/test-tmp','-e','PYTHONUTF8=1','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1',
        '--mount','type=bind,source='+str(root/'persistent-test-tmp')+',target=/test-tmp',
        '--mount','type=bind,source='+ASSETS+'/data/models,target=/opt/cloudrag/repository/data/models,readonly',
        '--mount','type=bind,source='+ASSETS+'/data/indices,target=/opt/cloudrag/repository/data/indices,readonly',
        '--entrypoint','python']
    commands = []
    for name,args in [('pip-check',['-m','pip','check']),
        ('linux-full',['-m','pytest','-ra','-p','no:cacheprovider','--basetemp=/test-tmp/full']),
        ('linux-filtered',['-m','pytest','-ra','-p','no:cacheprovider','--basetemp=/test-tmp/filtered','-m','not slow and not gpu']),
        ('posix-durability',['-m','pytest','tests/test_session_durability.py','-ra','-p','no:cacheprovider','--basetemp=/test-tmp/durability']),
        ('packages',['-m','pip','freeze'])]:
        commands.append((name,[*base,'--name','cloudrag-i5-'+label+'-'+name,image,*args],900))
    return commands


class Guest:
    def __init__(self,job,*,invoke=subprocess.Popen):
        if (job['purpose'] != 'technical' or job['operation'] != 'I5_CPU_BUILD'
                or not job['label'].isalnum() or len(job['label']) > 16
                or job['bucket'] != 'cloudrag-study-103950017681-20261002'
                or not job['prefix'].startswith('iteration4/iteration5/') or '..' in job['prefix'].split('/')):
            raise ValueError('Build scope differs')
        self.job,self.invoke = job,invoke
        self.root = Path('/srv/cloudrag/iteration5/builds')/job['label']
        self.logs = self.root/'evidence'
        self.logs.mkdir(parents=True,exist_ok=False)
        self.deadline = time.monotonic()+5400
        self.active = None

    def step(self,label,argv,timeout=600,*,env=None):
        at,begin = datetime.now(timezone.utc).isoformat(),time.monotonic()
        timed_out = False
        with (self.logs/(label+'.stdout')).open('xb') as out,(self.logs/(label+'.stderr')).open('xb') as err:
            self.active = self.invoke(argv,stdout=out,stderr=err,start_new_session=True,env=env)
            try:
                self.active.wait(timeout=max(1,min(timeout,self.deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if self.active.poll() is None:
                    os.killpg(self.active.pid,signal.SIGTERM)
                    try:
                        self.active.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.active.pid,signal.SIGKILL)
                        self.active.wait(timeout=5)
        code = 124 if timed_out else self.active.returncode
        self.active = None
        persist(self.logs/(label+'.receipt.json'),dict(command=argv,started_utc=at,
            ended_utc=datetime.now(timezone.utc).isoformat(),duration_s=time.monotonic()-begin,
            exit_code=code,timed_out=timed_out))
        if code:
            raise ValueError('STEP_FAILED_'+label)

    def run(self):
        watchdog = threading.Timer(5400,stop)
        watchdog.daemon = True
        watchdog.start()
        status,image = 'FAILED',None
        try:
            if metadata('id').decode() != self.job['instance_id']:
                raise ValueError('INSTANCE_ID_DIFFERS')
            context = Path(self.job['context'])
            repo = context/'repository'
            self.step('source-commit',['git','-C',str(repo),'rev-parse','HEAD'])
            self.step('source-clean',['git','-C',str(repo),'status','--porcelain'])
            if ((self.logs/'source-commit.stdout').read_text().strip() != self.job['commit']
                    or (self.logs/'source-clean.stdout').read_text().strip()):
                raise ValueError('BUILD_SOURCE_DIFFERS')
            self.step('docker-version',['docker','version','--format','{{json .Server.Version}}'])
            self.step('docker-build-help',['docker','build','--help'])
            self.step('docker-run-help',['docker','run','--help'])
            tag = 'cloudrag-study:i5-'+self.job['label']
            self.step('build',['docker','build','--rm=false','--network=none','--pull=false','-t',tag,
                              '-f',str(repo/'Dockerfile'),str(context)],3600,
                      env=dict(os.environ,DOCKER_BUILDKIT='0'))
            self.step('image-inspect',['docker','image','inspect',tag])
            image = json.loads((self.logs/'image-inspect.stdout').read_bytes())[0]['Id']
            persist(self.logs/'image-id.json',dict(image_id=image,commit=self.job['commit'],image_tag=tag))
            (self.root/'persistent-test-tmp').mkdir()
            for label,argv,timeout in check_commands(image,self.root,self.job['label']):
                self.step(label,argv,timeout)
            self.step('persistent-mount',['findmnt','-T',str(self.root/'persistent-test-tmp'),'-o','SOURCE,TARGET,FSTYPE','--json'])
            self.step('isolated-runtime-user',['docker','run','--rm=false','--name','cloudrag-i5-'+self.job['label']+'-user',
                '--network','none','--log-driver','none','--user','10001:10001','--read-only',
                '--tmpfs','/tmp:rw,nosuid,nodev,size=512m','-e','HOME=/tmp','-e','USER=cloudrag','--entrypoint','python',image,'-c',
                'import getpass,torch._dynamo,subprocess; assert getpass.getuser()=="cloudrag"; assert not subprocess.check_output(["git","status","--porcelain"]).strip(); print("ISOLATED_RUNTIME_USER_AND_CLEAN_GIT_PASS")'],120)
            source_name = 'cloudrag-i5-'+self.job['label']+'-host-source'
            self.step('host-source-create',['docker','create','--name',source_name,'--network','none',
                                          '--log-driver','none','--entrypoint','true',image])
            host = Path('/srv/cloudrag/iteration5/code-'+self.job['label'])
            host.mkdir(parents=True,exist_ok=False)
            for folder in ('scripts','src','config'):
                self.step('host-source-'+folder,['docker','cp',source_name+':/opt/cloudrag/repository/'+folder,str(host)])
            files = {p.relative_to(host).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in host.rglob('*') if p.is_file()}
            persist(self.logs/'host-code-inventory.json',dict(image_id=image,commit=self.job['commit'],files=files))
            status = 'PASS'
        except BaseException as error:
            persist(self.logs/'failure.json',dict(status='FAILED',error_type=type(error).__name__,
                reason=str(error) if isinstance(error,ValueError) and str(error).startswith('STEP_FAILED_') else 'BUILD_OR_CHECK_FAILED'))
        finally:
            try:
                persist(self.logs/'receipt.json',dict(status=status,image_id=image,commit=self.job['commit'],
                    vm_id=self.job['instance_id'],finished_utc=datetime.now(timezone.utc).isoformat(),
                    no_model_generation=True,final_gpu_acceptance_not_inferred=True))
                context = Path(self.job['context'])
                sys.path.insert(0,str(context/'repository'))
                from scripts.study_operator.gcs import Storage
                def token():
                    return json.loads(metadata('service-accounts/default/token'))['access_token']
                storage = Storage(self.job['bucket'],token)
                receipts = [storage.create_technical(self.job['prefix']+'/'+p.name,p.read_bytes())
                            for p in sorted(self.logs.iterdir()) if p.is_file()]
                data = json.dumps(dict(status=status,files=receipts,image_id=image,commit=self.job['commit'])).encode()
                upload = storage.create_technical(self.job['prefix']+'/manifest.json',data)
                persist(self.root/'upload-receipt.json',upload)
            except BaseException as error:
                persist(self.root/'upload-failure.json',dict(status='OWNER_RECOVERY_REQUIRED',error_type=type(error).__name__))
            finally:
                watchdog.cancel()
                stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    args = parser.parse_args(argv)
    Guest(json.loads(Path(args.job).read_bytes())).run()


if __name__ == '__main__':
    main()
