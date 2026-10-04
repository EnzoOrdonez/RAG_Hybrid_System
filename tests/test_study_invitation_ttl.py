import hashlib

import pytest

from src.ui.components import study_sessions as study
from tests.study_helpers import configured


@pytest.fixture
def store(tmp_path):
    _, _, protocol = configured(tmp_path)
    result = study.StudyStore(tmp_path / 'sessions', protocol)
    result.freeze()
    return result


def test_hash_only_registration_expires_at_24h(store, monkeypatch):
    monkeypatch.setattr(study.time, 'time', lambda: 1000)
    token = 'synthetic token; no real invitation'
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    store.register_invitation_hash(token_hash, 'P01')
    assert token not in store.path.read_text(encoding='utf-8')
    assert store._read()['invitations'][token_hash]['expires_at'] == 87400
    monkeypatch.setattr(study.time, 'time', lambda: 87399)
    session = store.admit(token)
    assert store.admit(token).session_id == session.session_id
    monkeypatch.setattr(study.time, 'time', lambda: 87400)
    with pytest.raises(ValueError, match='Invitación'):
        store.admit(token)


def test_legacy_no_ttl_and_plaintext_registration_fail_closed(store):
    token = store.issue('P01')
    data = store._read()
    del data['invitations'][hashlib.sha256(token.encode()).hexdigest()]['expires_at']
    study.atomic_json(store.path, data)
    with pytest.raises(ValueError, match='Invitación'):
        store.admit(token)
    with pytest.raises(ValueError, match='hash'):
        store.register_invitation_hash('plaintext token', 'P02')


def test_revocation_blocks_admitted_session_and_completed_token(store):
    token = store.issue('P01')
    session = store.admit(token)
    store.revoke('P01')
    with pytest.raises(ValueError, match='Invitación'):
        store.admit(token)
    with pytest.raises(study.SessionConflict):
        store.assert_active(session.session_id)
    token2 = store.issue('P02')
    session2 = store.admit(token2)
    store.complete_admission(session2.session_id)
    with pytest.raises(ValueError, match='Invitación'):
        store.admit(token2)


def test_duplicate_hash_or_participant_never_creates_second_session(store):
    digest = hashlib.sha256(b'synthetic').hexdigest()
    store.register_invitation_hash(digest, 'P01')
    for pid, value in [('P01', hashlib.sha256(b'other').hexdigest()), ('P02', digest)]:
        with pytest.raises(ValueError, match='already'):
            store.register_invitation_hash(value, pid)
    assert len(store._read()['invitations']) == 1
