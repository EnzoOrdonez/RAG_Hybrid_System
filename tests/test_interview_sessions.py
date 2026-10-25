import json

import pytest

from src.ui.components import session_manager as sm


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sm, "_get_evaluation_queries", lambda: [
        {"query_id": f"q{i}", "question": f"Question {i}"} for i in range(30)])
    result = sm.EvaluationSession("P01", "Intermedio")
    result.state = "evaluating"
    result.save_checkpoint()
    return result


def test_participant_path_cannot_escape_root(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "SESSIONS_DIR", tmp_path)
    with pytest.raises(ValueError):
        sm.EvaluationSession("../escape", "Intermedio")


def test_failed_query_cannot_be_rated(session):
    with pytest.raises(ValueError):
        session.record_rating(3, 3, 1, 2, 3, 4)


def test_answer_survives_reload_and_rating_is_idempotent(session):
    aid = session.begin_attempt({"llm_model": "fake"}, 1.0)
    session.finish_attempt(aid, answer="Exact answer [Source: aws/s3]", sources=[{"url": "https://example.org"}],
                           chunks=[{"chunk_id": "c1", "text": "evidence"}], verification={"method": "nli"})
    loaded = sm.EvaluationSession.load_checkpoint(session.session_id)
    assert loaded.pending_attempt["answer"] == "Exact answer [Source: aws/s3]"
    assert loaded.pending_attempt["sources"][0]["url"] == "https://example.org"
    loaded.record_rating(3, 4, 1, 2, 3, 4)
    loaded.save_checkpoint()
    with pytest.raises(ValueError):
        loaded.record_rating(3, 4, 1, 2, 3, 4)


def test_stale_tab_cannot_overwrite_checkpoint(session):
    other = sm.EvaluationSession.load_checkpoint(session.session_id)
    session.begin_attempt({}, 1)
    with pytest.raises(sm.SessionConflict):
        other.save_checkpoint()


def test_corrupt_checkpoint_is_not_treated_as_new(session):
    path = session.get_session_dir() / "session_checkpoint.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(sm.SessionStorageError):
        sm.EvaluationSession.load_checkpoint(session.session_id)
    assert path.read_text(encoding="utf-8") == "{"


@pytest.mark.parametrize("field,value", [("current_query_index", -1), ("state", "unknown"), ("query_sets", {})])
def test_valid_json_with_invalid_session_state_is_preserved(session, field, value):
    path = session.get_session_dir() / "session_checkpoint.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(sm.SessionStorageError):
        sm.EvaluationSession.load_checkpoint(session.session_id)
    assert path.read_bytes() == before


def test_atomic_write_failure_preserves_previous_checkpoint(session, monkeypatch):
    from src.ui.components import session_storage
    path = session.get_session_dir() / "session_checkpoint.json"
    before = path.read_bytes()
    monkeypatch.setattr(session_storage.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError):
        session.save_checkpoint()
    assert path.read_bytes() == before


def test_invitation_admission_and_completed_session_do_not_overwrite(session):
    from src.ui.components.session_storage import InvitationStore
    store = InvitationStore(sm.SESSIONS_DIR)
    token = store.issue("P02")
    admitted = store.admit(token, "Intermedio")
    with pytest.raises(ValueError):
        store.admit("P02", "Intermedio")
    second = store.issue("P03")
    with pytest.raises(sm.SessionConflict):
        store.admit(second, "Intermedio")
    admitted.state = "complete"
    admitted.save_checkpoint()
    assert store.admit(token, "Intermedio").session_id == admitted.session_id
    assert store.admit(second, "Intermedio").participant_id == "P03"


def test_error_attempt_stays_pending_without_rating(session):
    aid = session.begin_attempt({}, 1)
    session.finish_attempt(aid, error="generation_failed")
    assert session.pending_attempt["status"] == "error"
    with pytest.raises(ValueError):
        session.record_rating(2, 2, 1, 2, 3, 4)
    assert not session.ratings
