from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.ui.components import session_manager as sm
from src.ui.components.session_storage import InvitationStore

ROOT = Path(__file__).resolve().parents[1]


def test_operator_modules_are_not_autodiscovered():
    assert not [p for p in (ROOT / "src/ui/pages").glob("*.py") if p.name != "__init__.py"]


def test_participant_registers_only_interview_before_and_after_login(tmp_path, monkeypatch):
    monkeypatch.setenv("CLOUDRAG_MODE", "participant")
    monkeypatch.setattr(sm, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sm, "_get_evaluation_queries", lambda: [
        {"query_id": f"q{i}", "question": f"Question {i}"} for i in range(30)])
    token = InvitationStore(tmp_path).issue("P900")
    registered = []
    real_navigation = st.navigation

    def observed_navigation(pages, **kwargs):
        registered.append(([page.title for page in pages], kwargs.get("position")))
        return real_navigation(pages, **kwargs)

    monkeypatch.setattr(st, "navigation", observed_navigation)
    app = AppTest.from_file(str(ROOT / "src/ui/app.py"), default_timeout=10).run()
    assert not app.exception
    assert registered == [(["Evaluation Mode"], "hidden")]
    assert not app.radio
    app.text_input[0].set_value(token)
    app.checkbox[0].check()
    next(b for b in app.button if b.label == "Comenzar Evaluacion").click().run()
    assert not app.exception
    assert len(registered) >= 2
    assert all(value == (["Evaluation Mode"], "hidden") for value in registered)
    assert any(h.value == "Training Phase" for h in app.subheader)
