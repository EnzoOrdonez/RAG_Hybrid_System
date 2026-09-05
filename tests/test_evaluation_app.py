"""Real Streamlit execution from invitation through export, with no model execution."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from src.ui.components import session_manager as sm
from src.ui.components.session_storage import InvitationStore
from src.pipeline.pipeline_config import SURVEY_DEPLOY


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def test_evaluation_full_flow_and_reload(tmp_path, monkeypatch):
    from src.ui.components import index_loader
    monkeypatch.setenv("CLOUDRAG_MODE", "participant")
    monkeypatch.setattr(sm, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sm, "_get_evaluation_queries", lambda: [
        {"query_id": f"q{i}", "question": f"Question {i}"} for i in range(30)])
    calls = []
    def query(question):
        calls.append(question)
        return SimpleNamespace(answer="Saved answer", error=None, confidence="MEDIUM",
                               sources=[{"provider": "aws", "service": "s3"}],
                               retrieved_chunks=[{"chunk_id": "c1", "text": "evidence"}],
                               hallucination_report={"method": "nli"})
    fake = SimpleNamespace(config=SURVEY_DEPLOY, query=query,
                           llm=SimpleNamespace(seed=42, cache_enabled=False, num_ctx=4096))
    monkeypatch.setattr(index_loader, "load_hybrid_index", lambda: object())
    monkeypatch.setattr(index_loader, "load_pipeline", lambda *a, **k: fake)
    token = InvitationStore(tmp_path).issue("P01")
    app = AppTest.from_file(str(Path(__file__).parents[1] / "src/ui/app.py"), default_timeout=10).run()
    assert not app.exception
    assert not app.radio  # no operator navigation
    app.text_input[0].set_value(token)
    app.checkbox[0].check()
    _button(app, "Comenzar Evaluacion").click().run()
    assert not app.exception
    for _ in range(3):
        _button(app, "Buscar respuesta").click().run()
        _button(app, "Next Practice Question").click().run()
    _button(app, "Begin Evaluation").click().run()
    for index in range(30):
        _button(app, "Buscar respuesta").click().run()
        assert not app.exception
        if index == 0:
            before = len(calls)
            # A fresh browser reconnects with its invitation to the pending answer.
            app = AppTest.from_file(str(Path(__file__).parents[1] / "src/ui/app.py"), default_timeout=10).run()
            app.text_input[0].set_value(token)
            app.checkbox[0].check()
            _button(app, "Comenzar Evaluacion").click().run()
            assert len(calls) == before
            assert any(element.value == "Saved answer" for element in app.markdown)
        _button(app, "Siguiente pregunta →").click().run()
        if index in (9, 19):
            saved = app.session_state.eval_session
            saved.break_started_at = time.time() - 121
            saved.save_checkpoint()
            app.run()
            _button(app, "Estoy listo (continuar)").click().run()
    assert not app.exception
    _button(app, "Enviar cuestionario").click().run()
    _button(app, "Finalizar evaluacion").click().run()
    assert not app.exception
    session = app.session_state.eval_session
    assert session.state == "complete"
    exported = json.loads((session.get_session_dir() / "full_session.json").read_text(encoding="utf-8"))
    assert len(exported["attempts"]) == len(exported["ratings"]) == 30
    assert all(attempt["answer"] == "Saved answer" for attempt in exported["attempts"])
    assert len(calls) == 33  # three unrecorded practice questions, no duplicate generation
    assert not any("System order" in element.value for element in app.markdown)
