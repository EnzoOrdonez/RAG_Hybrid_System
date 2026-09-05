import json
from types import SimpleNamespace

import pytest
from filelock import FileLock, Timeout

from src.pipeline.pipeline_config import SURVEY_DEPLOY
from src.ui.components import session_manager as sm
from src.ui.components.evaluation_service import answer_query, recover_interrupted
from src.ui.components.session_storage import InvitationStore


@pytest.fixture
def active(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sm, "_get_evaluation_queries", lambda: [
        {"query_id": f"q{i}", "question": f"Question {i}"} for i in range(30)])
    store = InvitationStore(tmp_path)
    session = store.admit(store.issue("P01"), "Intermedio")
    session.state = "evaluating"
    session.save_checkpoint()
    return session


@pytest.mark.parametrize("error,answer,method", [("offline", "Error", "nli"), (None, "", "nli"),
                                               (None, "answer", "mixed"), (None, "answer", "keyword_fallback")])
def test_technical_failures_are_persisted_and_not_ratable(active, error, answer, method):
    response = SimpleNamespace(answer=answer, error=error, confidence="LOW", sources=[],
                               retrieved_chunks=[], hallucination_report={"method": method})
    pipeline = SimpleNamespace(config=SURVEY_DEPLOY, query=lambda q: response,
                               llm=SimpleNamespace(seed=42, cache_enabled=False))
    answer_query(active, lambda key: pipeline)
    saved = sm.EvaluationSession.load_checkpoint(active.session_id)
    assert saved.pending_attempt["status"] == "error"
    assert saved.pending_attempt["answer"] is None
    with pytest.raises(ValueError):
        saved.record_rating(1, 1, 1, 2, 3, 4)


def test_recovery_cannot_interrupt_an_inflight_request(active):
    active.begin_attempt({}, 1)
    with FileLock(str(sm.SESSIONS_DIR / "_inference.lock")):
        with pytest.raises(Timeout):
            recover_interrupted(active)
    recover_interrupted(active)
    assert active.pending_attempt["error"] == "interrupted"


def test_partial_export_is_not_complete_and_retry_preserves_answers(active, monkeypatch):
    active.state = "open_questions"
    active.ratings = [{"attempt_id": str(i), "utility_rating": 3} for i in range(30)]
    active.attempts = [{"attempt_id": str(i), "status": "rated", "answer": f"Answer {i}"} for i in range(30)]
    active.sus_responses = [1, 5] * 5
    active.save_checkpoint()
    original = sm.atomic_json
    def fail_full(path, value):
        if path.name == "full_session.json":
            raise OSError("disk full")
        return original(path, value)
    monkeypatch.setattr(sm, "atomic_json", fail_full)
    with pytest.raises(OSError):
        active.export_results()
    assert sm.EvaluationSession.load_checkpoint(active.session_id).state == "open_questions"
    monkeypatch.setattr(sm, "atomic_json", original)
    active.export_results()
    path = active.get_session_dir() / "full_session.json"
    before = path.read_bytes()
    assert json.loads(before)["sus_score"] == 0
    active.export_results()
    assert path.read_bytes() == before


def test_revoked_admission_cannot_generate(active):
    InvitationStore(sm.SESSIONS_DIR).abandon(active.session_id)
    with pytest.raises(sm.SessionConflict):
        answer_query(active, lambda _: pytest.fail("must not construct pipeline"))
