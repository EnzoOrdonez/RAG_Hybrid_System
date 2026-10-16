"""The owner and the actual qualified image must share the exact app lock."""
import hashlib
import json
from datetime import datetime, timezone

import pytest


def test_prebuild_matches_owner_lock_and_recipe_without_image():
    from scripts.study_operator.i7_streamlit_preflight import verify_before_cpu
    result = verify_before_cpu()
    assert result['status'] == 'STREAMLIT_BUILD_INPUTS_MATCH'
    assert result['owner_version'] == result['lock_version'] == '1.54.0'
    assert len(result['recipe_sha256']) == 64
    assert 'Dockerfile' in result['files'] and 'requirements-lock.txt' in result['files']


def image_evidence(root, version='1.54.0'):
    image = 'sha256:'+'a'*64
    commit = 'b'*40
    (root/'installation.json').write_text(json.dumps({'iteration':7, 'image_id':image, 'commit':commit}))
    directory = root/'cpu/linux-build-finali7-evidence'
    directory.mkdir(parents=True)
    files = {'packages.stdout':f'streamlit=={version}\n'.encode(),
             'packages.receipt.json':json.dumps({'exit_code':0,
                 'command':['docker','run','--network','none','--entrypoint','python',
                            image,'-m','pip','freeze']}).encode()}
    rows = []
    for name, data in files.items():
        (directory/name).write_bytes(data)
        rows.append({'object':'technical/'+name, 'bytes':len(data),
                     'sha256':hashlib.sha256(data).hexdigest()})
    proof = {'status':'TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED', 'image_id':image,
             'commit':commit, 'files':rows}
    (root/'cpu/linux-build-finali7-download-proof.json').write_text(json.dumps(proof))
    return directory


def test_owner_lock_and_qualified_image_streamlit_match(tmp_path):
    from scripts.study_operator.i7_streamlit_preflight import verify
    image_evidence(tmp_path)
    result = verify(tmp_path)
    assert result['status'] == 'STREAMLIT_VERSIONS_MATCH'
    assert result['owner_version'] == result['lock_version'] == result['image_version'] == '1.54.0'


@pytest.mark.parametrize('fault', ['owner', 'image', 'lock', 'missing', 'tampered', 'identity'])
@pytest.mark.parametrize('step', ['ready','smoke'])
def test_phase_b_blocks_streamlit_fault_before_cloud(tmp_path, monkeypatch, capsys, fault, step):
    from scripts.study_operator import cloud_client, i7_browser_preflight, i7_phase_b
    from scripts.study_operator import i7_streamlit_preflight
    class November(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(i7_phase_b, 'datetime', November)
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: pytest.fail('Cloud constructed before Streamlit admission'))
    directory = image_evidence(tmp_path, '1.53.0' if fault == 'image' else '1.54.0')
    if fault == 'owner':
        monkeypatch.setattr(i7_streamlit_preflight, 'version', lambda name: '1.53.0')
    if fault == 'lock':
        lock = tmp_path/'requirements-lock.txt'
        lock.write_text('streamlit==1.53.0\n')
        monkeypatch.setattr(i7_streamlit_preflight, 'LOCK', lock)
    if fault == 'missing':
        (directory/'packages.stdout').unlink()
    if fault == 'tampered':
        (directory/'packages.stdout').write_text('streamlit==1.53.0\n')
    if fault == 'identity':
        path = tmp_path/'installation.json'
        config = json.loads(path.read_bytes())
        config['image_id'] = 'sha256:'+'c'*64
        path.write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        i7_phase_b.main(['--live','--root',str(tmp_path),'--sdk','unused',step])
    error = capsys.readouterr().err
    assert ('STREAMLIT_VERSION_MISMATCH' if fault in ('owner','image','lock')
            else 'QUALIFIED_I7_STREAMLIT_EVIDENCE_REQUIRED') in error


def test_local_phase_b_check_admits_equal_versions_without_cloud(tmp_path, monkeypatch):
    from scripts.study_operator import cloud_client, i7_browser_preflight, i7_phase_b, i7_readiness
    image_evidence(tmp_path)
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(i7_readiness, 'evaluate', lambda *a, **kw: {'status':'LOCAL_RUNBOOK_READY','blockers':[]})
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: pytest.fail('Local check constructed cloud'))
    result = i7_phase_b.validate_steps(root=tmp_path, step='gpu-restore')
    assert result['status'] == 'LOCAL_RUNBOOK_READY'
    assert result['blockers'] == []
    assert result['streamlit']['status'] == 'STREAMLIT_VERSIONS_MATCH'


def test_missing_image_is_named_in_local_phase_b_check(tmp_path, monkeypatch):
    from scripts.study_operator import i7_browser_preflight, i7_phase_b, i7_readiness
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(i7_readiness, 'evaluate', lambda *a, **kw: {'status':'LOCAL_RUNBOOK_READY','blockers':[]})
    result = i7_phase_b.validate_steps(root=tmp_path, step='gpu-restore')
    assert result['status'] == 'LOCAL_RUNBOOK_PARTIAL'
    assert result['blockers'] == [{'owner':'code','code':'QUALIFIED_I7_STREAMLIT_EVIDENCE_REQUIRED'}]


@pytest.mark.parametrize('fault', ['owner','recipe','lock'])
def test_cpu_preflight_blocks_before_cloud(tmp_path, monkeypatch, capsys, fault):
    from scripts.study_operator import cloud_client, i7_browser_preflight, i7_phase_b, i7_streamlit_preflight as check
    for name in check.RECIPE_FILES:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((check.ROOT/name).read_bytes())
    monkeypatch.setattr(check, 'ROOT', tmp_path)
    monkeypatch.setattr(check, 'LOCK', tmp_path/'requirements-lock.txt')
    if fault == 'owner':
        monkeypatch.setattr(check, 'version', lambda name: '1.53.0')
    else:
        path = tmp_path/('Dockerfile' if fault == 'recipe' else 'requirements-lock.txt')
        path.write_bytes(path.read_bytes()+b'\n# changed after recipe pin\n')
    class November(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026,11,1,tzinfo=timezone.utc)
    monkeypatch.setattr(i7_phase_b, 'datetime', November)
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: pytest.fail('CPU cloud constructed before preflight rejection'))
    with pytest.raises(SystemExit):
        i7_phase_b.main(['--live','--root',str(tmp_path),'--sdk','unused','cpu-build'])
    assert ('STREAMLIT_VERSION_MISMATCH' if fault == 'owner' else 'I7_BUILD_RECIPE_HASH_MISMATCH') in capsys.readouterr().err


def test_cpu_preflight_allows_first_build_without_image(tmp_path, monkeypatch):
    from scripts.study_operator import cloud_client, i7_browser_preflight, i7_phase_b, i7_readiness
    class November(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026,11,1,tzinfo=timezone.utc)
    class Admitted(Exception):
        pass
    def cloud(*a, **kw):
        raise Admitted
    monkeypatch.setattr(i7_phase_b, 'datetime', November)
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(i7_readiness, 'evaluate', lambda *a, **kw: {'blockers':[]})
    monkeypatch.setattr(cloud_client, 'Cloud', cloud)
    with pytest.raises(Admitted):
        i7_phase_b.main(['--live','--root',str(tmp_path),'--sdk','unused','cpu-build'])


def cpu_owner(root, *, stop_fails=False, replacement=False):
    """Exercise the real owned-CPU STOP logic with only the API doubled."""
    from types import SimpleNamespace
    name = 'cloudrag-i5-restore-i7cpu01'
    vm = {'type':'vm','disposable':True,'name':name,'id':'11',
          'zone':'us-central1-a','ownership_marker':'synthetic-owner'}
    disk = dict(vm, type='disk', name=name+'-boot', id='22')
    cpu = root/'cpu'
    (cpu/'STATE.json').write_text(json.dumps({'resources':[vm,disk]}))
    (cpu/'cpu-restoration-i7cpu01-proof.json').write_text(json.dumps({
        'status':'CPU_RESTORATION_VERIFIED','synthetic':False,'cpu_vm_id':'11',
        'restored_disk_id':'22','source_snapshot_id':'33','zone':'us-central1-a',
        'all_expected_files_verified':True,'image_config_verified':True,
        'model_manifest_and_blobs_verified':True}))
    events, receipts = [], []
    class Cloud:
        status = 'RUNNING'
        def command(self, args, **kw):
            if args[:3] == ['compute','instances','describe']:
                events.append('describe')
                return dict(vm, id='replacement' if replacement else '11', status=self.status,
                    machineType='e2-standard-2', deletionProtection=True,
                    description='synthetic-owner', disks=[{'autoDelete':False}],
                    scheduling={'instanceTerminationAction':'STOP','maxRunDuration':{'seconds':'7200'}},
                    networkInterfaces=[])
            assert args[:3] == ['compute','instances','stop'], 'Only owned CPU STOP allowed'
            events.append('stop')
            if stop_fails:
                raise RuntimeError('synthetic API failure')
            self.status = 'TERMINATED'
            return {}
    return SimpleNamespace(root=root, cloud=Cloud(),
        reserve=lambda *a: pytest.fail('No GPU reservation after rejection'),
        receipt=lambda kind,value: receipts.append(value)), events, receipts


@pytest.mark.parametrize('fault', ['image','owner','missing'])
def test_gpu_rejection_verifies_cpu_stop_without_reserving(tmp_path, monkeypatch, fault):
    from scripts.study_operator.i7_cpu_pipeline import gpu_restore
    from scripts.study_operator import i7_streamlit_preflight as check
    directory = image_evidence(tmp_path, '1.53.0' if fault == 'image' else '1.54.0')
    if fault == 'owner':
        monkeypatch.setattr(check, 'version', lambda name: '1.53.0')
    if fault == 'missing':
        (directory/'packages.stdout').unlink()
    owner, events, receipts = cpu_owner(tmp_path)
    with pytest.raises(ValueError, match='STREAMLIT_VERSION_MISMATCH|QUALIFIED_I7_STREAMLIT_EVIDENCE_REQUIRED'):
        gpu_restore(owner)
    assert events == ['describe','stop','describe']
    assert owner.cloud.status == 'TERMINATED'
    assert receipts[-1]['status'] == 'STREAMLIT_BLOCKED_CPU_STOPPED'


@pytest.mark.parametrize('fault', ['api','replacement'])
def test_unconfirmed_cpu_stop_still_blocks_gpu(tmp_path, fault):
    from scripts.study_operator.i7_cpu_pipeline import gpu_restore
    image_evidence(tmp_path, '1.53.0')
    owner, events, receipts = cpu_owner(tmp_path, stop_fails=fault=='api', replacement=fault=='replacement')
    with pytest.raises(ValueError, match='STREAMLIT_BLOCKED_CPU_STOP_UNCONFIRMED'):
        gpu_restore(owner)
    assert receipts[-1]['status'] == 'STREAMLIT_BLOCKED_CPU_STOP_UNCONFIRMED'
    assert events == (['describe','stop'] if fault=='api' else ['describe'])


def test_matching_image_admission_reads_census_without_cloud(tmp_path):
    from scripts.study_operator.i7_cpu_pipeline import verify_image_or_stop_cpu
    image_evidence(tmp_path)
    owner, events, receipts = cpu_owner(tmp_path)
    result = verify_image_or_stop_cpu(owner)
    assert result['status'] == 'STREAMLIT_VERSIONS_MATCH'
    assert events == []
    assert receipts == [result]


def test_live_gpu_rejection_stops_cpu_before_other_admissions(tmp_path, monkeypatch):
    from scripts.study_operator import cloud_client, i7, i7_browser_preflight, i7_phase_b, i7_readiness
    image_evidence(tmp_path, '1.53.0')
    owner, events, receipts = cpu_owner(tmp_path)
    class November(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026,11,1,tzinfo=timezone.utc)
    monkeypatch.setattr(i7_phase_b, 'datetime', November)
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: owner.cloud)
    monkeypatch.setattr(i7, 'Operator', lambda *a, **kw: owner)
    monkeypatch.setattr(i7_readiness, 'evaluate', lambda *a, **kw: pytest.fail('Reject image and stop CPU first'))
    with pytest.raises(ValueError, match='STREAMLIT_VERSION_MISMATCH'):
        i7_phase_b.main(['--live','--root',str(tmp_path),'--sdk','unused','gpu-restore'])
    assert events == ['describe','stop','describe']
    assert receipts[-1]['cpu_status'] == 'TERMINATED'


def test_mismatched_owner_can_still_run_close(tmp_path, monkeypatch):
    from scripts.study_operator import cloud_client, i7, i7_browser_preflight, i7_phase_b, i7_workflows
    from scripts.study_operator import i7_streamlit_preflight as check
    class November(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026,11,1,tzinfo=timezone.utc)
    monkeypatch.setattr(i7_phase_b, 'datetime', November)
    monkeypatch.setattr(check, 'version', lambda name: '1.53.0')
    monkeypatch.setattr(i7_browser_preflight, 'verify', lambda: {})
    monkeypatch.setattr(cloud_client, 'Cloud', lambda *a, **kw: object())
    monkeypatch.setattr(i7, 'Operator', lambda *a, **kw: object())
    monkeypatch.setattr(i7_workflows, 'verify_tool_source', lambda owner: None)
    monkeypatch.setattr(i7_phase_b, 'dispatch', lambda owner,step,spec: {'status':'CLOSED'})
    assert i7_phase_b.main(['--live','--root',str(tmp_path),'--sdk','unused','close']) == 0
