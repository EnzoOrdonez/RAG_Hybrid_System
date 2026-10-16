
import pytest

from scripts.study_operator.i7_phase_b import profile, validate_steps


def test_gate_profile_does_not_extend_cold_trials():
    assert profile('gate') == {'i7_profile':'gate','native_stop_s':10800}
    assert profile('stimulus') == {'i7_profile':'standard','native_stop_s':10800}
    with pytest.raises(ValueError):
        profile('participants')


def test_runbook_steps_have_one_executable_entrypoint_and_dependencies():
    result = validate_steps()
    assert result['status'] == 'LOCAL_RUNBOOK_PARTIAL'
    assert {row['owner'] for row in result['blockers']}=={'code','Enzo'}
    assert result['steps'] >= 9


def test_incomplete_prerequisites_block_gate_before_any_cloud_call(tmp_path):
    from scripts.study_operator.i7_phase_b import gate
    from scripts.study_operator.i7_gate_guest import dispatch_gate
    class Owner:
        root=tmp_path
        def __getattr__(self,name):
            pytest.fail('No owner/cloud effect while prerequisites are absent')
    with pytest.raises(ValueError,match='ENZO_PUBLISHED'):
        gate(Owner())
    active={'boot_root':'unused','config':{'purpose':'technical'}}
    with pytest.raises(ValueError,match='ENZO_PUBLISHED'):
        dispatch_gate(active,{'operation':'gate-start'},execute=Owner(),controller=Owner(),maintenance=Owner())


def test_local_config_never_targets_inherited_vm(tmp_path,monkeypatch):
    import json

    from scripts.study_operator import i7_config
    from tests.study_helpers import configured
    source={}
    # Deployment validation is covered separately; here verify the cloning contract.
    source.update(primary_vm={'name':'inherited','id':'123'},instance_id='123',primary_disk={'id':'456'})
    path=tmp_path/'old.json'
    path.write_text(json.dumps(source))
    prices=tmp_path/'prices'
    prices.mkdir()
    (prices/'receipt.json').write_text(json.dumps({'at':'2026-11-01T00:00:00+00:00'}))
    config,csv,protocol=configured(tmp_path)
    monkeypatch.setattr(i7_config,'quote_archive',lambda *args:{'usd_per_hour':'.7','catalog_receipt_sha256':'a'*64})
    monkeypatch.setattr(i7_config,'checked_config',lambda config:config)
    monkeypatch.setattr('src.ui.components.study_protocol.verify_draw',lambda path:protocol)
    root=tmp_path/'new-i7'
    (tmp_path/'rag_freeze_baseline.json').write_text('{}')
    assert i7_config.create(root,{'source_installation':str(path),'price_archive':str(prices),
        'reviewed_config':str(config),'inherited_package':str(tmp_path)},commit='b'*40)['cloud_measured'] is False
    saved=json.loads((root/'installation.json').read_bytes())
    assert saved['primary_vm']['name']=='cloudrag-i7-unallocated'
    assert saved['primary_vm']['id']=='0' and 'primary_disk' not in saved and 'instance_id' not in saved
    assert len(set(saved['period_ids'].values()))==2


def test_cleaner_requires_ack_and_preserves_only_service_payload(tmp_path):
    from scripts.study_operator.i7_preparation import file_sha
    from scripts.study_operator.i7_service_disk import sanitize
    root = tmp_path/'new-service-clone'
    root.mkdir()
    config = {}
    for name,key in [('assets','asset_root'),('models','ollama_models'),('code','host_code'),
                     ('reviewed','reviewed_config'),('jit','cuda_cache')]:
        folder = root/'srv/cloudrag'/name
        folder.mkdir(parents=True)
        (folder/'file').write_bytes(name.encode())
        config[key]='/srv/cloudrag/'+name
    tls = root/'srv/cloudrag/iteration5/tls'
    tls.mkdir(parents=True)
    (tls/'certificate').write_bytes(b'unchanged')
    for name,key in [('manifest','artifact_manifest'),('prereg','preregistration_file')]:
        path = root/'srv/cloudrag'/name
        path.write_bytes(b'frozen')
        config[key]='/srv/cloudrag/'+name
    old = root/'srv/cloudrag/iteration3/runtime/old-log'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'legacy IP')
    archive, images = tmp_path/'private.tar',tmp_path/'images.tar'
    archive.write_bytes(b'archived original bytes')
    images.write_bytes(b'pinned images')
    job={'installation':config,'legacy_archive':str(archive),'legacy_archive_ack_sha256':'0'*64,
         'image_archive':str(images),'image_archive_sha256':file_sha(images)}
    with pytest.raises(ValueError,match='ACK'):
        sanitize(root,job,verify_target=lambda *args:None)
    assert old.exists()
    job['legacy_archive_ack_sha256']=file_sha(archive)
    result=sanitize(root,job,verify_target=lambda *args:None)
    assert result['status']=='CLEAN_SERVICE_DISK_PREPARED_NOT_BOOT_VALIDATED'
    assert not old.exists()
    assert (root/'srv/cloudrag/i7/service/assets/file').read_bytes()==b'assets'
    assert (root/'srv/cloudrag/iteration5/tls/certificate').read_bytes()==b'unchanged'
    assert archive.read_bytes()==b'archived original bytes'


def test_gpu_restore_checks_snapshot_before_allocating_capacity(tmp_path,monkeypatch):
    from scripts.study_operator.i7_cpu_pipeline import gpu_restore
    monkeypatch.setattr('scripts.study_operator.i7_cpu_pipeline.verify_image_or_stop_cpu',lambda owner:None)
    class Cloud:
        def command(self,args,**kwargs):
            return {'id':'replacement','status':'READY'}
    class Operator:
        def __init__(self):
            self.state={'qualified_service_snapshot':{'name':'qualified','id':'original'}}
            self.cloud=Cloud()
        def reserve(self,hours):
            pytest.fail('No reservation before source snapshot readback')
    with pytest.raises(ValueError,match='SNAPSHOT_CHANGED'):
        gpu_restore(Operator())


def test_close_before_gpu_allocation_never_targets_placeholder(tmp_path,monkeypatch):
    from scripts.study_operator.i7_workflows import dispatch
    from scripts.study_operator import i7_cpu_pipeline
    class Owner:
        config={'primary_vm':{'id':'0'}}
        state={}
        def stop(self):
            pytest.fail('Do not stop unallocated GPU identity')
        def guard(self):
            pytest.fail('Do not guard nonexistent reservation')
    monkeypatch.setattr(i7_cpu_pipeline,'cleanup_cpu',lambda owner:{'status':'NO_QUALIFIED_CPU_CLEANUP_PLAN'})
    assert dispatch(Owner(),'close')['status']=='CLOSED'
    Owner.state={'creation_intents':[{'status':'UNKNOWN'}]}
    assert dispatch(Owner(),'close')['status']=='BLOCKED_UNKNOWN_RESOURCE'
