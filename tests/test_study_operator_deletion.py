
import pytest

from scripts.study_operator.deletion import execute
from scripts.study_operator.policy import OperatorError


class MemoryStorage:
    def __init__(self):
        self.data = {'study/period/P999/synthetic/full_session.json': b'{"synthetic":true}',
                     'study/period/P999/synthetic/export_manifest.json': b'{"files":{}}'}
        self.deleted = []
        self.soft = []

    def objects(self, prefix, *, versions=False, soft_deleted=False):
        if soft_deleted:
            return self.soft
        return [{'name': name, 'generation': '100', 'size': len(data)}
                for name, data in sorted(self.data.items()) if name.startswith(prefix)]

    def read(self, name, generation):
        assert generation == '100'
        return self.data[name]

    def delete(self, name, generation):
        self.deleted.append((name, generation))
        del self.data[name]


def test_dry_run_is_read_only_and_soft_delete_blocks(tmp_path):
    storage = MemoryStorage()
    result = execute(storage, 'study/period/P999/', tmp_path, dry_run=True)
    assert result['status'] == 'DRY_RUN' and len(result['objects']) == 2
    assert not storage.deleted and not list(tmp_path.iterdir())
    storage.soft = [{'name': 'retained copy'}]
    with pytest.raises(OperatorError, match='soft-deleted'):
        execute(storage, 'study/period/P999/', tmp_path)
    assert not storage.deleted


def test_every_generation_downloaded_before_any_remote_delete(tmp_path):
    storage = MemoryStorage()
    original = storage.delete

    def delete(name, generation):
        files = list(tmp_path.rglob('*.download'))
        assert len(files) == 2
        assert sum(p.stat().st_size for p in files) == len(b'{"synthetic":true}') + len(b'{"files":{}}')
        original(name, generation)

    storage.delete = delete
    result = execute(storage, 'study/period/P999/', tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    assert result['status'] == 'DELETED_VERIFIED'
    assert result['remote_versions_empty'] and result['remote_soft_deleted_empty']
    assert all(len(row['sha256']) == 64 for row in result['verified_downloads'])


def test_corrupt_download_prevents_all_remote_deletion(tmp_path):
    storage = MemoryStorage()
    storage.read = lambda name, generation: b''
    with pytest.raises(OperatorError, match='incompleta'):
        execute(storage, 'study/period/P999/', tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    assert not storage.deleted


def test_missing_disk_agent_and_unknown_object_fail_closed(tmp_path):
    storage = MemoryStorage()
    with pytest.raises(OperatorError, match='limpieza'):
        execute(storage, 'study/period/P999/', tmp_path, dry_run=False)
    storage.data['study/period/P999/uninventoried.log'] = b'unknown'
    with pytest.raises(OperatorError, match='desconocido'):
        execute(storage, 'study/period/P999/', tmp_path)
    assert not storage.deleted


def test_other_participant_is_preserved_and_replay_is_safe(tmp_path):
    storage = MemoryStorage()
    other = 'study/period/P998/synthetic/full_session.json'
    storage.data[other] = b'other'
    first = execute(storage, 'study/period/P999/', tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    second = execute(storage, 'study/period/P999/', tmp_path, dry_run=False, disk_cleanup=lambda rows: {'empty': True})
    assert first['deleted'] == 2 and second['deleted'] == 0
    assert storage.data == {other: b'other'}
