import json

import pytest

from src.ui.components.session_storage import SessionStorageError
from src.ui.components.study_sessions import StudyStore
from tests.study_helpers import configured
from tests.test_study_sessions import finish


def test_preissued_invitation_cannot_enter_after_actual_backup_failure(tmp_path, monkeypatch):
    from src.ui.components import study_backup

    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol)
    store.freeze()
    first = store.admit(store.issue('P01'))
    invitation = store.issue('P02')
    finish(first)
    export = first.export()

    def failed_copy(*args):
        raise OSError('copy interrupted')

    monkeypatch.setattr(study_backup.shutil, 'copytree', failed_copy)
    with pytest.raises(OSError, match='copy interrupted'):
        study_backup.backup_export(export.parent, tmp_path / 'backup', same_physical_disk=lambda *_: False)
    before = store.path.read_bytes()
    with pytest.raises(SessionStorageError, match='backup'):
        store.admit(invitation)
    assert store.path.read_bytes() == before
    assert list(store.root.glob('*/' + first.path.name)) == [first.path]


@pytest.mark.parametrize('state', ['{', '[]', '{"status": "unknown"}'])
def test_unreadable_or_unknown_backup_state_fails_closed(tmp_path, state):
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol)
    store.freeze()
    token = store.issue('P01')
    prior = store.root / ('a' * 32)
    prior.mkdir()
    (prior / 'backup_state.json').write_text(state, encoding='utf-8')
    with pytest.raises(SessionStorageError):
        store.admit(token)


def test_verified_backup_allows_preissued_invitation_without_reissue(tmp_path):
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol)
    store.freeze()
    token = store.issue('P01')
    prior = store.root / ('a' * 32)
    prior.mkdir()
    status = prior / 'backup_state.json'
    status.write_text(json.dumps({'status': 'pending'}), encoding='utf-8')
    with pytest.raises(SessionStorageError):
        store.admit(token)
    status.write_text(json.dumps({'status': 'complete'}), encoding='utf-8')
    assert store.admit(token).data['assignment']['participant_id'] == 'P01'
