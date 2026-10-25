from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.study_helpers import configured
from src.ui.components.study_sessions import StudyStore

ROOT = Path(__file__).resolve().parents[1]


def test_operator_modules_are_not_autodiscovered():
    assert not [p for p in (ROOT / "src/ui/pages").glob("*.py") if p.name != "__init__.py"]


def test_participant_registers_only_interview_before_and_after_login(tmp_path, monkeypatch):
    c, a, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol, 'technical')
    store.freeze()
    token = store.issue('P900', cell=1, profile='without_experience')
    for key, value in dict(CLOUDRAG_MODE='participant', CLOUDRAG_STUDY_CONFIG=c,
            CLOUDRAG_STUDY_ASSIGNMENTS=a, CLOUDRAG_STUDY_SESSION_DIR=store.root,
            CLOUDRAG_STUDY_PURPOSE='technical').items():
        monkeypatch.setenv(key, str(value))
    registered = []
    real = st.navigation
    def observed(pages, **kwargs):
        registered.append(([page.title for page in pages], kwargs.get('position')))
        return real(pages, **kwargs)
    monkeypatch.setattr(st, 'navigation', observed)
    app = AppTest.from_file(str(ROOT / 'src/ui/app.py')).run()
    assert not app.exception
    app.text_input[0].set_value(token)
    next(b for b in app.button if b.label == 'Entrar').click().run()
    assert not app.exception
    assert len(registered) >= 2
    assert all(x == (['Sesión'], 'hidden') for x in registered)
    assert not app.sidebar.radio
