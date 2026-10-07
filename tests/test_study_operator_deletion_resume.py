import hashlib
import json

import pytest

from scripts.study_operator.deletion import execute
from scripts.study_operator.policy import OperatorError
from test_study_operator_deletion import MemoryStorage


PREFIX = 'study/period/P999/'


def test_resume_after_one_generation_deleted_uses_original_verified_downloads(tmp_path):
    storage = MemoryStorage()
    delete = storage.delete
    calls = []

    def interrupted(name, generation):
        calls.append(name)
        if len(calls) == 2:
            raise ConnectionError('synthetic interruption')
        delete(name, generation)

    storage.delete = interrupted
    with pytest.raises(ConnectionError):
        execute(storage, PREFIX, tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    assert len(storage.data) == 1
    storage.delete = delete
    storage.read = lambda *args: pytest.fail('verified generations must not be downloaded again')
    result = execute(storage, PREFIX, tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    assert result['status'] == 'DELETED_VERIFIED' and len(result['verified_downloads']) == 2
    assert not storage.data


def test_resume_rejects_changed_local_backup_after_partial_deletion(tmp_path):
    storage = MemoryStorage()
    delete = storage.delete

    def interrupted(name, generation):
        delete(name, generation)
        raise ConnectionError('after delete, before recording receipt')

    storage.delete = interrupted
    with pytest.raises(ConnectionError):
        execute(storage, PREFIX, tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    next(tmp_path.rglob('*.download')).write_bytes(b'corrupt')
    storage.delete = lambda *args: pytest.fail('no additional deletion allowed')
    with pytest.raises(OperatorError, match='alterada'):
        execute(storage, PREFIX, tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})


def test_disk_cleanup_failure_retries_original_inventory(tmp_path):
    storage = MemoryStorage()
    with pytest.raises(OperatorError, match='no completo'):
        execute(storage, PREFIX, tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': False})
    assert not storage.data
    disk = []
    result = execute(storage, PREFIX, tmp_path, dry_run=False,
                     disk_cleanup=lambda rows: disk.extend(rows) or {'empty': True})
    assert result['disk']['empty'] and len(disk) == 2
    for item in disk:
        file = tmp_path / item['relative'] / (item['generation'] + '.download')
        assert hashlib.sha256(file.read_bytes()).hexdigest() == item['sha256']
    assert json.loads(next(tmp_path.glob('deletion-*.json')).read_text())['stage'] == 'COMPLETE'


def test_resume_rejects_new_remote_objects_after_disk_failure(tmp_path):
    storage = MemoryStorage()
    with pytest.raises(OperatorError):
        execute(storage, PREFIX, tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': False})
    storage.data[PREFIX + 'new/full_session.json'] = b'new session'
    with pytest.raises(OperatorError, match='inventario'):
        execute(storage, PREFIX, tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    assert storage.data


def test_complete_replay_does_not_reaccredit_a_tampered_local_download(tmp_path):
    storage = MemoryStorage()
    execute(storage, PREFIX, tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    next(tmp_path.rglob('*.download')).write_bytes(b'changed after completed transaction')
    with pytest.raises(OperatorError, match='alterada'):
        execute(storage, PREFIX, tmp_path, dry_run=False,
            disk_cleanup=lambda rows: pytest.fail('corrupt backup must fail before disk cleanup'))
