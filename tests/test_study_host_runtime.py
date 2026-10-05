import base64
import hashlib
import json
import subprocess

import pytest

from scripts.study_operator.gcs import Storage
from scripts.study_operator.host_runtime import metadata_unreachable, re_safe_reason


@pytest.mark.parametrize('observed,exit_code', [({'169.254.169.254':True,'metadata.google.internal':False},0),
    ({'169.254.169.254':False,'metadata.google.internal':True},0), ({},0), ({},1)])
def test_actual_container_metadata_probe_rejects_either_route(observed, exit_code):
    def invoke(argv, **options):
        assert argv[:3] == ['docker','exec','owned-app']
        assert '169.254.169.254' in argv[-1] and 'metadata.google.internal' in argv[-1]
        assert options['timeout'] == 15
        return subprocess.CompletedProcess(argv,exit_code,json.dumps(observed).encode(),b'')
    with pytest.raises(ValueError,match='METADATA_ISOLATION_NOT_VERIFIED'):
        metadata_unreachable('owned-app',invoke=invoke)


def test_metadata_probe_and_technical_error_redaction():
    result = metadata_unreachable('owned-app',invoke=lambda argv,**options:
        subprocess.CompletedProcess(argv,0,b'{"169.254.169.254":false,"metadata.google.internal":false}',b''))
    assert result['metadata_unreachable']
    assert re_safe_reason('READY_DEADLINE_900S')
    assert not re_safe_reason('private question or client IP 203.0.113.4')


def test_creator_only_technical_upload_never_requests_read_permission(monkeypatch):
    storage = Storage('technical-bucket',lambda:'memory-token')
    data = b'fixed technical metadata'
    calls = []
    def request(path,**options):
        calls.append((path,options))
        assert options['method'] == 'POST' and options['params']['ifGenerationMatch'] == 0
        return json.dumps(dict(name='iteration4/job/ready.json',generation='123',size=len(data),
            md5Hash=base64.b64encode(hashlib.md5(data).digest()).decode(),timeCreated='2026-10-05T00:00:00Z')).encode()
    monkeypatch.setattr(storage,'request',request)
    receipt = storage.create_technical('iteration4/job/ready.json',data)
    assert len(calls) == 1 and receipt['owner_download_verification_pending']
    assert receipt['sha256'] == hashlib.sha256(data).hexdigest()


def test_creator_only_write_rejects_unsafe_prefix_and_wrong_checksum(monkeypatch):
    storage = Storage('technical-bucket',lambda:'unused')
    with pytest.raises(ValueError,match='prefix'):
        storage.create_technical('iteration3/retained/evidence.json',b'data')
    monkeypatch.setattr(storage,'request',lambda *args,**options:
        b'{"name":"iteration4/new.json","generation":"123","size":"4","md5Hash":"wrong"}')
    with pytest.raises(ValueError,match='checksum'):
        storage.create_technical('iteration4/new.json',b'data')
