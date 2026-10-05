import base64
import hashlib
import json
from pathlib import Path

import pytest

from scripts.study_operator import guest_bridge
from scripts.study_operator.managed_stores import managed_stores, recovery_counts, snapshot_safe
from scripts.study_operator.policy import OperatorError
from scripts.study_operator.session_control import read_plan, restore_closed
from scripts.study_operator.session_data import clean, inventory
from src.ui.components.study_sessions import StudyStore
from tests.test_study_session_restore import backup


def setup_stores(tmp_path):
    (tmp_path/'source-data').mkdir()
    source,full,manifest,objects = backup(tmp_path/'source-data')
    period = tmp_path/'periods'/('a'*32)
    primary = period/'sessions'
    recovered = period/'recoveries'/hashlib.sha256(full).hexdigest()/'sessions'
    for folder in (primary,recovered):
        folder.mkdir(parents=True)
        (folder/'_i4_root.json').write_text(json.dumps(dict(schema_version=1,purpose='technical')))
        (folder.parent/'private-inventory').mkdir()
        restore_closed(StudyStore(folder,source.protocol,'technical'),full,manifest,objects)
    active = dict(session_root=str(primary),config=dict(purpose='technical'))
    return active,full,manifest,objects


def controller(active,request,**options):
    root = active['session_root']
    operation = request['operation']
    if operation == 'inventory':
        return dict(plan=inventory(root,code=request.get('code'),
                                  private_inventory=str(Path(root).parent/'private-inventory')))
    if operation == 'download-disk':
        return dict(files=read_plan(request['plan']))
    if operation == 'clean-disk':
        return dict(result=clean(request['plan'],request['verified_downloads']))
    raise AssertionError(operation)


def test_withdraw_covers_primary_and_fresh_recovery_and_distinguishes_paths(tmp_path,monkeypatch):
    active,_,_,_ = setup_stores(tmp_path)
    monkeypatch.setattr(guest_bridge,'controller',controller)
    assert len(managed_stores(active['session_root'],'technical')) == 2
    assert recovery_counts(active['session_root'],'technical') == dict(session_count=1,invitation_count=1)
    plan = guest_bridge.disk_dispatch(active,dict(operation='inventory',code='P900'))['plan']
    assert plan['session_count'] == 2 and len(plan['stores']) == 2
    rows = guest_bridge.disk_dispatch(active,dict(operation='download-disk',plan=plan))['files']
    assert {row['store_id'] for row in rows} == {item['store_id'] for item in plan['stores']}
    result = guest_bridge.disk_dispatch(active,dict(operation='clean-disk',plan=plan,verified_downloads=rows))
    assert result['result']['empty']
    assert all(not inventory(folder)['session_codes'] for folder in managed_stores(active['session_root'],'technical').values())
    assert snapshot_safe(tmp_path/'periods')['status'] == 'ALL_I4_PERIODS_EMPTY'


def test_snapshot_refuses_closed_data_and_recovery_copies(tmp_path):
    active,_,_,_ = setup_stores(tmp_path)
    with pytest.raises(OperatorError,match='Persisten datos'):
        snapshot_safe(tmp_path/'periods')
    primary = inventory(active['session_root'])
    clean(primary,primary['files'])
    with pytest.raises(OperatorError,match='Persisten datos'):
        snapshot_safe(tmp_path/'periods')


def test_recovery_replay_requires_exact_export_generation_and_checkpoint(tmp_path):
    source,full,manifest,objects = backup(tmp_path)
    new = StudyStore(tmp_path/'empty-app',source.protocol,'technical')
    result = restore_closed(new,full,manifest,objects,allow_replay=True)
    assert restore_closed(new,full,manifest,objects,allow_replay=True)['replayed']
    checkpoint = new.root/result['session_id']/'study_checkpoint.json'
    value = json.loads(checkpoint.read_text())
    value['attempts'][0]['answer'] = 'synthetic changed checkpoint'
    checkpoint.write_text(json.dumps(value))
    with pytest.raises(OperatorError,match='Checkpoint restaurado'):
        restore_closed(new,full,manifest,objects,allow_replay=True)


def test_restore_selects_new_app_store_without_overwriting_primary(tmp_path,monkeypatch):
    active,full,manifest,objects = setup_stores(tmp_path)
    before = {str(path):path.read_bytes() for path in Path(active['session_root']).rglob('*') if path.is_file()}
    observed = []
    monkeypatch.setattr(guest_bridge.os,'chown',lambda *args:None,raising=False)
    monkeypatch.setattr(guest_bridge,'controller',lambda value,request,**options: observed.append(value) or dict(status='RESTORED_VERIFIED'))
    request = dict(operation='restore',full_session_base64=base64.b64encode(full).decode(),
                   manifest_base64=base64.b64encode(manifest).decode(),objects=objects)
    result = guest_bridge.disk_dispatch(active,request)
    assert result['new_app_instance'] and result['original_store_unchanged']
    assert observed[0]['session_root'] != active['session_root']
    assert before == {str(path):path.read_bytes() for path in Path(active['session_root']).rglob('*') if path.is_file()}


def test_unknown_recovery_or_changed_copy_set_blocks_deletion(tmp_path,monkeypatch):
    active,_,_,_ = setup_stores(tmp_path)
    monkeypatch.setattr(guest_bridge,'controller',controller)
    plan = guest_bridge.disk_dispatch(active,dict(operation='inventory',code='P900'))['plan']
    plan['stores'].append(plan['stores'][-1])
    with pytest.raises(ValueError,match='MANAGED_STORE_INVENTORY_CHANGED'):
        guest_bridge.disk_dispatch(active,dict(operation='clean-disk',plan=plan,verified_downloads=[]))
    root = Path(active['session_root']).parent/'recoveries'/'unknown'
    root.mkdir()
    with pytest.raises(OperatorError,match='desconocida'):
        managed_stores(active['session_root'],'technical')
