import hashlib
import json

import pytest

from scripts.study_operator.backup_agent import backup


class Storage:
    def __init__(self):
        self.data = {}

    def put(self, name, data):
        self.data[name] = data
        return {'object': name, 'generation': '123', 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


@pytest.fixture
def prepared(tmp_path):
    sid = 'a' * 32
    root = tmp_path / 'sessions'
    session = root / sid
    session.mkdir(parents=True)
    data = json.dumps(dict(session_id=sid, assignment={'participant_id': 'P999'}, purpose='smoke', stage='complete')).encode()
    digest = hashlib.sha256(data).hexdigest()
    (session / 'full_session.json').write_bytes(data)
    (session / 'export_manifest.json').write_text(json.dumps({'files': {'full_session.json': digest}}))
    settings = dict(sessions_root=str(root), private_inventory_root=str(tmp_path / 'private'),
                    sessions_bucket='synthetic-bucket', prefix='smoke/iteration4', purpose='smoke')
    request = dict(session_id=sid, bucket='synthetic-bucket', prefix='smoke/iteration4', sha256=digest)
    return settings, request


def test_host_backup_is_fixed_scope_generation_bound_and_privately_inventoried(prepared):
    settings, request = prepared
    storage = Storage()
    result = backup(settings, request, storage)
    assert result['status'] == 'complete' and result['sha256'] == request['sha256']
    assert all(name.startswith('smoke/iteration4/P999/' + request['session_id'] + '/') for name in storage.data)
    assert len(result['objects']) == 2
    assert all(row['generation'] == '123' for row in result['objects'].values())
    assert backup(settings, request, storage) == result


@pytest.mark.parametrize('fault,value', [('bucket', 'other-bucket'), ('prefix', 'other-study'),
                                        ('session_id', '../escape'), ('sha256', '0' * 64)])
def test_host_never_accepts_arbitrary_destination_or_changed_data(prepared, fault, value):
    settings, request = prepared
    storage = Storage()
    with pytest.raises(ValueError):
        backup(settings, dict(request, **{fault: value}), storage)
    assert not storage.data
