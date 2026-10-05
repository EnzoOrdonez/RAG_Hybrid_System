import hashlib
import json

import pytest

from scripts.study_operator.policy import OperatorError
from scripts.study_operator.session_control import read_plan, restore_closed
from src.ui.components.study_sessions import StudyStore, StudySession
from tests.study_helpers import configured
from tests.test_study_sessions import finish


def backup(tmp_path):
    _,_,protocol = configured(tmp_path)
    store = StudyStore(tmp_path/'source',protocol,'technical')
    store.freeze()
    session = store.admit(store.issue('P900',cell=1,profile='without_experience'))
    finish(session)
    full = session.export().read_bytes()
    manifest = (session.path.parent/'export_manifest.json').read_bytes()
    objects = {name:dict(generation=str(index),sha256=hashlib.sha256(data).hexdigest())
               for index,(name,data) in enumerate([('full_session.json',full),('export_manifest.json',manifest)],1)}
    return store,full,manifest,objects


def test_generation_verified_closed_session_restores_in_empty_app_and_exports_identically(tmp_path):
    original,full,manifest,objects = backup(tmp_path)
    new = StudyStore(tmp_path/'new-app',original.protocol,'technical')
    result = restore_closed(new,full,manifest,objects)
    session = StudySession.load(new,result['session_id'])
    assert session.export().read_bytes() == full
    assert (session.path.parent/'export_manifest.json').read_bytes() == manifest
    assert result['export_sha256'] == hashlib.sha256(full).hexdigest()
    assert all(invite['revoked'] and invite['expires_at'] == 0 for invite in new._read()['invitations'].values())
    assert new._read()['active'] is None
    with pytest.raises(OperatorError,match='vacía'):
        restore_closed(new,full,manifest,objects)


@pytest.mark.parametrize('fault',['hash','generation','protocol','purpose','stage'])
def test_restore_wrong_backup_never_creates_checkpoint(tmp_path,fault):
    original,full,manifest,objects = backup(tmp_path)
    payload = json.loads(full)
    if fault == 'hash':
        objects['full_session.json']['sha256'] = '0'*64
    elif fault == 'generation':
        objects['full_session.json']['generation'] = 'not-generation'
    else:
        key = {'protocol':'protocol_fingerprint','purpose':'purpose','stage':'stage'}[fault]
        payload[key] = 'wrong'
        full = json.dumps(payload).encode()
    new = StudyStore(tmp_path/'new-app',original.protocol,'technical')
    with pytest.raises(OperatorError,match='inválidos'):
        restore_closed(new,full,manifest,objects)
    assert not list(new.root.glob('*/study_checkpoint.json'))


def test_private_disk_download_rejects_changed_file_before_copy(tmp_path):
    path = tmp_path/'full_session.json'
    path.write_bytes(b'synthetic')
    plan = dict(root=str(tmp_path),files=[dict(path=str(path),relative=path.name,root=str(tmp_path),kind='session',
        bytes=len(b'synthetic'),sha256=hashlib.sha256(b'synthetic').hexdigest())])
    rows = read_plan(plan)
    assert rows[0]['content_base64'] and rows[0]['sha256'] == plan['files'][0]['sha256']
    path.write_bytes(b'changed')
    with pytest.raises(OperatorError,match='cambió'):
        read_plan(plan)
