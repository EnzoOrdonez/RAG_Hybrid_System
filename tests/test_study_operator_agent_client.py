import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from scripts import cloud_storage
from scripts.study_operator import agent_client
from src.ui.components.session_storage import atomic_json


@pytest.fixture
def session(tmp_path):
    root = tmp_path / ('a' * 32)
    atomic_json(root / 'full_session.json', {'synthetic': True})
    digest = hashlib.sha256((root / 'full_session.json').read_bytes()).hexdigest()
    atomic_json(root / 'export_manifest.json', {'files': {'full_session.json': digest}})
    receipt = dict(status='complete', sha256=digest, destination='gs://test-bucket/study/period/P999/' + root.name,
                   objects={name: dict(object='study/period/P999/' + root.name + '/' + name, generation='123',
                              sha256=hashlib.sha256((root / name).read_bytes()).hexdigest())
                            for name in ['full_session.json', 'export_manifest.json']})
    return root, receipt


def connection(monkeypatch, receipt, status=200):
    calls = []

    class Connection:
        def __init__(self, path):
            assert path == '/private/backup.sock'

        def request(self, method, path, body, headers):
            assert method == 'POST' and path == '/backup'
            calls.append(json.loads(body))

        def getresponse(self):
            stream = io.BytesIO(json.dumps(receipt).encode())
            return SimpleNamespace(status=status, read=stream.read)

        def close(self):
            pass

    monkeypatch.setattr(agent_client, 'UnixConnection', Connection)
    return calls


def test_application_uses_agent_without_cloud_credentials(session, monkeypatch):
    root, receipt = session
    calls = connection(monkeypatch, receipt)
    monkeypatch.setenv('CLOUDRAG_BACKUP_SOCKET', '/private/backup.sock')
    monkeypatch.setenv('CLOUDRAG_ISOLATED_SERVICE', '1')
    monkeypatch.setattr(cloud_storage.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('app reached metadata or GCS'))
    result = cloud_storage.backup_session(root, cloud_storage.Bucket('test-bucket'), 'study/period')
    assert result == receipt and calls == [dict(session_id=root.name, bucket='test-bucket', prefix='study/period', sha256=receipt['sha256'])]
    assert json.loads((root / 'backup_state.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('fault', ['generation', 'sha256', 'destination', 'http'])
def test_invalid_agent_receipt_stays_pending(session, monkeypatch, fault):
    root, receipt = session
    if fault == 'generation':
        receipt['objects']['full_session.json']['generation'] = ''
    elif fault == 'sha256':
        receipt['sha256'] = '0' * 64
    elif fault == 'destination':
        receipt['destination'] = 'gs://other-bucket/other-study'
    connection(monkeypatch, receipt, 503 if fault == 'http' else 200)
    monkeypatch.setenv('CLOUDRAG_BACKUP_SOCKET', '/private/backup.sock')
    with pytest.raises(ValueError):
        cloud_storage.backup_session(root, cloud_storage.Bucket('test-bucket'), 'study/period')
    assert json.loads((root / 'backup_state.json').read_text())['status'] == 'pending'


def test_study_and_isolated_app_cannot_fall_back_to_metadata(monkeypatch):
    monkeypatch.setenv('CLOUDRAG_STUDY_PURPOSE', 'study')
    monkeypatch.setattr(cloud_storage.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('metadata accessed'))
    with pytest.raises(ValueError, match='host backup agent'):
        cloud_storage.Bucket('test-bucket').request('https://storage.googleapis.com/example')
