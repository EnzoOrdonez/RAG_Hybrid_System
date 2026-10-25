import pytest

from src.ui.components.session_storage import SessionStorageError, atomic_json
from src.ui.components.study_sessions import StudyStore
from tests.study_helpers import configured


def test_closed_checkpoint_blocks_admission_before_export_exists(tmp_path, monkeypatch):
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol)
    store.freeze()
    token = store.issue('P01')
    other = store.root / ('b' * 32)
    atomic_json(other / 'study_checkpoint.json', {'stage': 'complete'})
    monkeypatch.setenv('CLOUDRAG_BACKUP_BUCKET', 'test-private-bucket')
    with pytest.raises(SessionStorageError, match='backup is missing'):
        store.admit(token)
    with pytest.raises(SessionStorageError, match='backup is missing'):
        store.issue('P02')
    assert not (other / 'full_session.json').exists()
