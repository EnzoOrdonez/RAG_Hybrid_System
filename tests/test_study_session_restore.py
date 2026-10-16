import hashlib
import json
import base64

import pytest

from scripts.study_operator.policy import OperatorError
from scripts.study_operator.session_control import read_plan, restore_closed
from src.ui.components.study_sessions import StudyStore, StudySession
from src.ui.components.session_storage import SessionStorageError
from tests.study_helpers import configured
from tests.test_study_sessions import finish


def recovery_request(tmp_path):
    original, full, manifest, objects = backup(tmp_path)
    config = tmp_path/'sealed'
    config.mkdir()
    (config/'study.json').write_bytes((tmp_path/'config.json').read_bytes())
    (config/'assignments.csv').write_bytes((tmp_path/'assignments.csv').read_bytes())
    protocol = original.protocol
    (config/'draw_seal.json').write_text(json.dumps(dict(schema_version=1,
        seeds=protocol['config']['randomization'], hashes=protocol['hashes'],
        fingerprint=protocol['fingerprint'])), encoding='utf-8')
    root = tmp_path/'recoveries'/'new'/'sessions'
    root.mkdir(parents=True)
    (root/'_i4_root.json').write_text(json.dumps(dict(schema_version=1,purpose='technical')))
    request = dict(operation='restore', full_session_base64=base64.b64encode(full).decode(),
        manifest_base64=base64.b64encode(manifest).decode(), objects=objects)
    deployment = dict(session_root=str(root), config_dir=str(config), purpose='technical')
    return request, deployment, full


def test_dispatch_restores_and_replays_in_new_recovery_store(tmp_path):
    from scripts.study_operator.session_control import dispatch

    request, deployment, full = recovery_request(tmp_path)
    result = dispatch(request, deployment, maintenance=True)
    assert result['status'] == 'RESTORED_VERIFIED'
    from pathlib import Path

    root = Path(deployment['session_root'])
    assert (root/'_study_protocol.json').is_file()
    assert (root/result['session_id']/'full_session.json').read_bytes() == full
    assert dispatch(request, deployment, maintenance=True)['replayed'] is True


@pytest.mark.parametrize('fault', ['maintenance', 'seal', 'hash', 'generation', 'occupied'])
def test_dispatch_restore_preserves_guards_and_never_writes_bad_checkpoint(tmp_path, fault):
    from pathlib import Path
    from scripts.study_operator.session_control import dispatch

    request, deployment, _ = recovery_request(tmp_path)
    root = Path(deployment['session_root'])
    if fault == 'seal':
        seal = Path(deployment['config_dir'])/'draw_seal.json'
        seal.write_text('{}', encoding='utf-8')
    elif fault == 'hash':
        request['objects']['full_session.json']['sha256'] = '0'*64
    elif fault == 'generation':
        request['objects']['full_session.json']['generation'] = 'invalid'
    elif fault == 'occupied':
        (root/'_admissions.json').write_text('{}', encoding='utf-8')
    with pytest.raises((ValueError, OperatorError, SessionStorageError)):
        dispatch(request, deployment, maintenance=fault != 'maintenance')
    assert not list(root.glob('*/study_checkpoint.json'))
    if fault in {'maintenance', 'seal', 'occupied'}:
        assert not (root/'_study_protocol.json').exists()


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
