import hashlib
import json

from scripts.study_operator import guest_bridge, session_control
from scripts.study_operator.i7_restore import probe
from tests.test_study_session_restore import recovery_request


def test_owner_probe_drives_bridge_control_container_and_identical_export(tmp_path,monkeypatch):
    request, deployment, full = recovery_request(tmp_path)
    primary = tmp_path/'period'/'sessions'
    primary.mkdir(parents=True)
    (primary/'_i4_root.json').write_text(json.dumps({'schema_version':1,'purpose':'technical'}))
    boot = tmp_path/'boot'
    boot.mkdir()
    config = {'schema_version': 1,'project': 'pure-loop-474323-a8','zone': 'us-central1-a','machine_type': 'g2-standard-4',
        'sessions_bucket': 'cloudrag-study-i4-103950017681-20261004','technical_bucket': 'cloudrag-study-103950017681-20261002',
        'purpose': 'technical','image_id': 'sha256:'+'a'*64,'commit': 'b'*40,'model_digest': 'c'*64,'period_id': 'd'*32,
        'static_ip': '203.0.113.10','hostname': '203.0.113.10.sslip.io','ollama_image': 'o@sha256:'+'e'*64,
        'caddy_image': 'c@sha256:'+'f'*64,'asset_root': '/srv/cloudrag/assets','ollama_models': '/srv/cloudrag/models',
        'host_code': '/srv/cloudrag/code'}
    active = {'session_root': str(primary),'boot_root': str(boot),'app_container': 'owned','config': config}
    monkeypatch.setattr(guest_bridge.os,'chown',lambda *a:None,raising=False)
    calls = []
    def execute(argv,payload=None,**kwargs):
        calls.append(argv)
        if argv[:2] == ['docker','inspect']:
            return [{'State':{'Running':False}}]
        assert argv[:2] == ['docker','run'] and argv[argv.index('--network')+1] == 'none'
        assert argv[-1] == 'scripts.study_operator.session_control'
        mount = next(s for s in argv if s.endswith('target=/sessions'))
        target = mount.split('source=',1)[1].split(',target=',1)[0]
        return session_control.dispatch(payload,dict(deployment,session_root=target),maintenance=True)
    monkeypatch.setattr(guest_bridge,'execute',execute)
    def bridge(payload):
        if payload == {'operation':'maintenance'}:
            (boot/'maintenance.json').write_text('{}')
            return {'status':'MAINTENANCE','admission_closed':True}
        return guest_bridge.disk_dispatch(active,payload)
    original = {p.name:p.read_bytes() for p in primary.iterdir()}
    result = probe(bridge,request)
    assert result['status'] == 'RESTORE_DEPLOYMENT_VERIFIED'
    assert result['export_sha256'] == hashlib.sha256(full).hexdigest()
    restored = list((primary.parent/'recoveries').glob('*/sessions/*/full_session.json'))
    assert len(restored) == 1 and restored[0].read_bytes() == full
    assert {p.name:p.read_bytes() for p in primary.iterdir()} == original
    assert any('CLOUDRAG_MAINTENANCE=1' in command for command in calls)
