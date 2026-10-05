import hashlib
import json

import pytest

from scripts.study_operator.bootstrap_failure import failure_summary, validate_summary


def test_failed_boot_summary_excludes_session_paths_and_raw_command_data(tmp_path):
    boot = tmp_path / '11111111-1111-4111-8111-111111111111'
    boot.mkdir()
    (boot / 'failure.json').write_text(json.dumps(dict(reason='HOST_COMMAND_FAILED', error_type='ValueError')))
    (boot / 'command-0001.json').write_text(json.dumps(dict(command=['docker', 'secret-free-query', '203.0.113.9'],
        exit_code=1, duration_s=2, started_utc='2026-10-05T00:00:00+00:00')))
    (boot / 'sessions').mkdir()
    (boot / 'sessions' / 'private.json').write_text('must never be read')
    value = failure_summary(boot, instance_id='123', image_id='sha256:'+'b'*64, commit='c'*40)
    text = json.dumps(value)
    assert 'secret-free-query' not in text and '203.0.113.9' not in text and 'must never' not in text
    assert value['commands'][0]['exit_code'] == 1
    assert value['commands'][0]['operation'] == 'HOST_COMMAND'
    assert value['commands'][0]['argv_sha256'] == hashlib.sha256(
        json.dumps(['docker', 'secret-free-query', '203.0.113.9'], sort_keys=True).encode()).hexdigest()
    validate_summary(value, instance_id='123', image_id='sha256:'+'b'*64, commit='c'*40)
    for key, replacement in [('instance_id','other'), ('image_id','sha256:'+'a'*64), ('commit','a'*40)]:
        with pytest.raises(ValueError):
            validate_summary(dict(value, **{key:replacement}),instance_id='123',image_id='sha256:'+'b'*64,commit='c'*40)
    with pytest.raises(ValueError):
        validate_summary(dict(value, user_agent='private'),instance_id='123',image_id='sha256:'+'b'*64,commit='c'*40)


def test_failure_reason_cannot_export_arbitrary_exception_text(tmp_path):
    boot = tmp_path / '11111111-1111-4111-8111-111111111111'
    boot.mkdir()
    (boot/'failure.json').write_text(json.dumps(dict(reason='private text 203.0.113.9',error_type='private user agent')))
    value = failure_summary(boot,instance_id='123',image_id='sha256:'+'b'*64,commit='c'*40)
    assert value['failure'] == dict(reason='HOST_FAILED',error_type='UnknownError')


@pytest.mark.parametrize('upload_fails',[False,True])
def test_failed_host_publishes_creator_only_summary_then_always_shuts_down(tmp_path,monkeypatch,upload_fails):
    from scripts.study_operator import backup_agent, host_runtime
    from scripts.study_operator.gcs import Storage

    host=host_runtime.Host.__new__(host_runtime.Host)
    host.root=tmp_path/'11111111-1111-4111-8111-111111111111'
    host.root.mkdir()
    host.boot=host.root.name
    host.began=host_runtime.time.monotonic()
    host.children=[]
    host.containers=[]
    host.config=dict(instance_id='123',image_id='sha256:'+'b'*64,commit='c'*40,technical_bucket='technical')
    def fail():
        raise ValueError('HOST_COMMAND_FAILED')
    host.prepare=fail
    calls=[]
    monkeypatch.setattr(backup_agent,'access_token',lambda:'fixture-token-only')
    def upload(self,name,data):
        calls.append('upload')
        assert name=='iteration4/failed-boots/123/'+host.boot+'.json'
        assert json.loads(data)['failure']['reason']=='HOST_COMMAND_FAILED'
        if upload_fails:
            raise RuntimeError('must never bypass shutdown')
        return dict(object=name,generation='42',sha256=hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(Storage,'create_technical',upload)
    monkeypatch.setattr(host_runtime.subprocess,'run',lambda argv,**kwargs:calls.append(argv))
    host.run()
    assert calls==['upload',['/sbin/shutdown','-h','now']]
    if upload_fails:
        assert json.loads((host.root/'failure-upload.json').read_text())['status']=='TECHNICAL_UPLOAD_FAILED'
