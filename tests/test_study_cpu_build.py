import ast
import base64
import copy
import hashlib
import json

import pytest

from scripts.study_operator import cpu_build
from scripts.study_operator.policy import OperatorError, validate_minimal_iam, validate_storage_iam


def fixture(tmp_path,monkeypatch):
    resource = dict(type='vm',name='cloudrag-i5-restore-bootstrap01',id='1',zone='us-central1-a',
                    disposable=True,ownership_marker='own-marker')
    vm = dict(name=resource['name'],id='1',zone='zones/us-central1-a',status='TERMINATED',
        deletionProtection=True,description='own-marker',machineType='types/e2-standard-2',
        disks=[dict(autoDelete=False)],networkInterfaces=[{}],
        scheduling=dict(instanceTerminationAction='STOP',maxRunDuration=dict(seconds='7200')),
        serviceAccounts=[dict(email=cpu_build.SA,scopes=['https://www.googleapis.com/auth/devstorage.read_write'])])
    member = 'serviceAccount:'+cpu_build.SA
    sessions = dict(bindings=[dict(role=r,members=[member]) for r in
                            ('roles/storage.objectCreator','roles/storage.objectViewer')])
    technical = dict(bindings=[dict(role='roles/storage.objectCreator',members=[member],condition=dict(
        expression="resource.name.startsWith('projects/_/buckets/"+cpu_build.TECHNICAL+"/objects/iteration4/')"))])
    state = dict(status='ACTIVE',resources=[resource],cloud_cutoff_usd=90,independent_closure_verified=True,
        closure_reserved_utc='2026-10-09T23:01:39Z',cost=dict(estimated_spend_usd=12,reserved_retention_and_closure_usd=4))
    (tmp_path/'STATE.json').write_text(json.dumps(state))
    (tmp_path/'cpu-restoration-bootstrap01-proof.json').write_text(json.dumps(dict(
        status='CPU_RESTORATION_VERIFIED',synthetic=False,cpu_vm_id='1')))
    monkeypatch.setattr(cpu_build,'quote_archive',lambda *a:dict(usd_per_hour='.067',catalog_receipt_sha256='c'*64))
    class Cloud:
        def __init__(self):
            self.calls=[]
            self.vm=vm
            self.private_access=True
            self.transport_failure=False
            self.stop_failure=False
        def owner_token(self):
            raise AssertionError('No fixture credentials')
        def command(self,args,**kwargs):
            self.calls.append(args)
            if args[:3] == ['compute','instances','describe']:
                return copy.deepcopy(self.vm)
            if args[:3] == ['compute','networks','subnets']:
                return dict(privateIpGoogleAccess=self.private_access)
            if args[:2] == ['projects','get-iam-policy']:
                return {}
            if args[:3] == ['storage','buckets','get-iam-policy']:
                return sessions if args[3] == 'gs://'+cpu_build.SESSIONS else technical
            if args[:3] == ['compute','instances','start']:
                self.vm['status']='RUNNING'
            if args[:3] == ['compute','instances','stop']:
                if not self.stop_failure:
                    self.vm['status']='TERMINATED'
            if args[:3] == ['compute','instances','get-guest-attributes']:
                return ['fixture public key']
            if args[:2] == ['compute','ssh']:
                if self.transport_failure:
                    raise OperatorError('fixture transport failure')
                job=json.loads((tmp_path/'cpu-build-final01-job-input.json').read_bytes())
                return json.dumps(dict(status='CPU_BUILD_LAUNCHED_NOT_ACCEPTED',pid=42,
                                      instance_id='1',context_sha256=job['context_sha256']))
    cloud=Cloud()
    build=cpu_build.Build(tmp_path,cloud)
    bundle=tmp_path/'source.tar.gz'
    bundle.write_bytes(b'fixture context')
    inventory=dict(bundle=str(bundle),status='CLEAN_PUBLISHED_BUILD_CONTEXT_VERIFIED',
        bytes=bundle.stat().st_size,sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),commit='a'*40)
    return build,cloud,resource,inventory,sessions,technical


@pytest.mark.parametrize('change',[dict(id='replacement'),dict(status='RUNNING'),dict(deletionProtection=False),
    dict(zone='zones/us-west1-a'),dict(networkInterfaces=[dict(accessConfigs=[{}])]),
    dict(scheduling=dict(instanceTerminationAction='STOP',maxRunDuration=dict(seconds='10800'))),
    dict(disks=[dict(autoDelete=True)]),dict(guestAccelerators=[{}])])
def test_cpu_identity_rejects_unsafe_or_replaced_resource(tmp_path,monkeypatch,change):
    build,cloud,resource,*_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):
        cpu_build.checked_cpu(dict(cloud.vm,**change),resource)
    assert not cloud.calls


def test_launch_is_bounded_and_never_claims_acceptance(tmp_path,monkeypatch):
    build,cloud,resource,inventory,*_=fixture(tmp_path,monkeypatch)
    result=build.launch('final01',inventory,b'fixture guest')
    assert result['status']=='CPU_BUILD_LAUNCHED_NOT_ACCEPTED'
    state=json.loads((tmp_path/'STATE.json').read_bytes())
    assert state['builds_started']==1 and state['build_jobs']['final01']['status']=='RUNNING'
    assert state['open_exposures']['cpu-build-final01']['not_billed_spend']
    assert '+110' in (tmp_path/'cpu-build-final01-startup.sh').read_text()
    payload=cpu_build.stage_payload(json.loads((tmp_path/'cpu-build-final01-job-input.json').read_bytes()),b'fixture',b'guest')
    ast.parse(payload)
    assert b"filter='data'" in payload and b'start_new_session=True' in payload


def test_private_google_access_and_budget_block_before_start(tmp_path,monkeypatch):
    build,cloud,resource,inventory,*_=fixture(tmp_path,monkeypatch)
    cloud.private_access=False
    with pytest.raises(ValueError,match='Private Google Access'):
        build.launch('final01',inventory,b'guest')
    assert not any(a[2]=='start' for a in cloud.calls if a[:2]==['compute','instances'])
    assert 'builds_started' not in json.loads((tmp_path/'STATE.json').read_bytes())


def test_build_cap_and_restoration_identity_reject_before_mutation(tmp_path,monkeypatch):
    build,cloud,resource,inventory,*_=fixture(tmp_path,monkeypatch)
    build.state.update(lambda state:state.update(builds_started=3))
    with pytest.raises(ValueError,match='allowance'):
        build.launch('final01',inventory,b'guest')
    assert all(a[2]=='describe' for a in cloud.calls)
    (tmp_path/'cpu-restoration-bootstrap01-proof.json').write_text(json.dumps(dict(
        status='CPU_RESTORATION_VERIFIED',synthetic=False,cpu_vm_id='replacement')))
    with pytest.raises(ValueError,match='independently verified'):
        build.resource()


def test_transport_failure_stops_and_preserves_counted_attempt(tmp_path,monkeypatch):
    build,cloud,resource,inventory,*_=fixture(tmp_path,monkeypatch)
    cloud.transport_failure=True
    with pytest.raises(OperatorError,match='transport failure'):
        build.launch('final01',inventory,b'guest')
    assert cloud.vm['status']=='TERMINATED'
    assert cloud.calls[-1][2]=='describe'
    state=json.loads((tmp_path/'STATE.json').read_bytes())
    assert state['builds_started']==1 and state['build_jobs']['final01']['status']=='LAUNCH_FAILED_PRESERVED'


def objects(tmp_path,monkeypatch,build,*,bad_hash=False,bad_mount=False,missing=False):
    prefix='iteration4/iteration5/fixture/final01'
    job=dict(prefix=prefix,commit='a'*40,instance_id='1')
    (tmp_path/'cpu-build-final01-job-input.json').write_text(json.dumps(job))
    build.state.update(lambda state:state.update(build_jobs=dict(final01=dict(
        status='RUNNING',vm_id='1',commit='a'*40))))
    data={name:json.dumps(dict(exit_code=0)).encode() for name in
        ('pip-check.receipt.json','linux-full.receipt.json','linux-filtered.receipt.json',
         'posix-durability.receipt.json','isolated-runtime-user.receipt.json')}
    data['persistent-mount.stdout']=json.dumps(dict(filesystems=[dict(source='/dev/sda1',target='/',fstype='tmpfs' if bad_mount else 'ext4')])).encode()
    data['posix-durability.stdout']=b'================ 7 passed in 1.00s ================'
    if missing:
        del data['pip-check.receipt.json']
    rows=[dict(object=prefix+'/'+name,generation=str(i+1),bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
          for i,(name,value) in enumerate(data.items())]
    if bad_hash:
        rows[0]['sha256']='0'*64
    manifest=json.dumps(dict(status='PASS',commit=job['commit'],image_id='fixture-image',files=rows)).encode()
    class Storage:
        def __init__(self,*a):pass
        def objects(self,*a):
            return [dict(name=prefix+'/manifest.json',generation='99',size=len(manifest),
                md5Hash=base64.b64encode(hashlib.md5(manifest).digest()).decode())]
        def read(self,name,generation):
            if name.endswith('/manifest.json'):
                assert generation=='99'
                return manifest
            row=next(r for r in rows if r['object']==name)
            assert generation==row['generation']
            return data[name.split('/')[-1]]
    monkeypatch.setattr(cpu_build,'Storage',Storage)


def test_collect_checks_each_generation_and_stops_before_proof(tmp_path,monkeypatch):
    build,cloud,*_=fixture(tmp_path,monkeypatch)
    objects(tmp_path,monkeypatch,build)
    cloud.vm['status']='RUNNING'
    proof=build.collect('final01')
    assert proof['cpu_stopped_verified'] and proof['final_gpu_acceptance_not_inferred']
    assert proof['manifest']['generation']=='99'
    assert cloud.vm['status']=='TERMINATED'
    assert build.collect('final01')==proof  # Idempotent verified download, never overwrite different evidence.


@pytest.mark.parametrize('status',['INTENT','LAUNCH_ATTEMPTED','RUNNING'])
def test_delayed_collector_never_stops_cpu_reused_for_another_build(tmp_path,monkeypatch,status):
    build,cloud,*_=fixture(tmp_path,monkeypatch)
    objects(tmp_path,monkeypatch,build)
    build.state.update(lambda state:state['build_jobs'].update(final02=dict(
        status=status,vm_id='1',commit='b'*40)))
    cloud.vm['status']='RUNNING'
    with pytest.raises(ValueError,match='Stale collector'):
        build.collect('final01')
    assert not cloud.calls
    assert cloud.vm['status']=='RUNNING'
    assert not (tmp_path/'linux-build-final01-evidence').exists()


def test_collector_rejects_unrecorded_cpu_before_cloud_access(tmp_path,monkeypatch):
    build,cloud,*_=fixture(tmp_path,monkeypatch)
    objects(tmp_path,monkeypatch,build)
    build.state.update(lambda state:state['build_jobs']['final01'].update(vm_id='replacement'))
    with pytest.raises(ValueError,match='ownership'):
        build.collect('final01')
    assert not cloud.calls


@pytest.mark.parametrize('failure',['hash','mount','missing','stop'])
def test_collect_never_seals_false_pass(tmp_path,monkeypatch,failure):
    build,cloud,*_=fixture(tmp_path,monkeypatch)
    objects(tmp_path,monkeypatch,build,bad_hash=failure=='hash',bad_mount=failure=='mount',missing=failure=='missing')
    if failure=='stop':
        cloud.vm['status']='RUNNING'
        cloud.stop_failure=True
    with pytest.raises(ValueError):
        build.collect('final01')
    assert not (tmp_path/'linux-build-final01-download-proof.json').exists()


def test_storage_iam_does_not_fabricate_app_metadata_isolation(tmp_path,monkeypatch):
    build,cloud,resource,inventory,sessions,technical=fixture(tmp_path,monkeypatch)
    kwargs=dict(sa=cpu_build.SA,bucket=cpu_build.TECHNICAL)
    assert validate_storage_iam(cloud.vm,{},sessions,technical,**kwargs)==dict(storage_iam_verified=True)
    for reachable in (None,True):
        with pytest.raises(OperatorError,match='metadatos'):
            validate_minimal_iam(cloud.vm,{},sessions,technical,metadata_reachable=reachable,**kwargs)
