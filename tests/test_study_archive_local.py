import hashlib
import json

import pytest

from scripts.study_operator.archive_local import verified_closed_copies
from scripts.study_operator.deletion import execute
from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_deletion import MemoryStorage


def test_archive_downloads_generations_then_cleans_disk_without_deleting_gcs(tmp_path):
    storage = MemoryStorage()
    original = dict(storage.data)
    cleanups = []
    def cleanup(rows):
        assert len(list(tmp_path.rglob('*.download'))) == 2 and len(rows) == 2
        assert not storage.deleted and storage.data == original
        cleanups.append(rows)
        return dict(empty=True)
    first = execute(storage,'study/period/P999/',tmp_path,dry_run=False,disk_cleanup=cleanup,retain_remote=True)
    second = execute(storage,'study/period/P999/',tmp_path,dry_run=False,disk_cleanup=cleanup,retain_remote=True)
    assert first['status'] == second['status'] == 'ARCHIVED_LOCAL_VERIFIED'
    assert first['remote_objects_retained'] == 2 and first['deleted'] == 0
    assert len(cleanups) == 2 and storage.data == original and not storage.deleted
    with pytest.raises(OperatorError,match='Propósito de transacción'):
        execute(storage,'study/period/P999/',tmp_path,dry_run=False,disk_cleanup=cleanup)


def test_archive_corrupt_download_never_cleans_disk_or_deletes_gcs(tmp_path):
    storage = MemoryStorage()
    storage.read = lambda *args:b''
    with pytest.raises(OperatorError,match='incompleta'):
        execute(storage,'study/period/P999/',tmp_path,dry_run=False,retain_remote=True,
                disk_cleanup=lambda rows:pytest.fail('No disk cleanup before verified downloads'))
    assert not storage.deleted and len(storage.data) == 2


def copies(tmp_path):
    sid = 'a'*32
    rows, remote, objects = [], [], {}
    def add(name,data,kind='session'):
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        rows.append(dict(store_id='primary',relative=name,kind=kind,local_path=str(path)))
    for name,data in [('full_session.json',b'{"synthetic":true}'),('export_manifest.json',b'{"files":{}}')]:
        add(sid+'/'+name,data)
        objects[name] = dict(object='periods/p/P999/'+sid+'/'+name,generation='123',sha256=hashlib.sha256(data).hexdigest())
        remote.append(objects[name])
    add(sid+'/backup_state.json',json.dumps(dict(status='complete',objects=objects)).encode())
    add(sid+'/study_checkpoint.json',json.dumps(dict(stage='complete')).encode())
    add('_admissions.json',json.dumps(dict(active=None)).encode(),kind='shared')
    return rows,remote


@pytest.mark.parametrize('fault',['open','active','missing-generation','changed-local'])
def test_archive_requires_closed_verified_backup_for_each_copy(tmp_path,fault):
    rows,remote = copies(tmp_path)
    assert verified_closed_copies(rows,remote)['closed_backups_verified'] == 1
    if fault == 'open':
        (tmp_path/('a'*32)/'study_checkpoint.json').write_text('{"stage":"tasks"}')
    elif fault == 'active':
        (tmp_path/'_admissions.json').write_text('{"active":"synthetic"}')
    elif fault == 'missing-generation':
        remote.pop()
    else:
        (tmp_path/('a'*32)/'full_session.json').write_bytes(b'changed')
    with pytest.raises(OperatorError):
        verified_closed_copies(rows,remote)
