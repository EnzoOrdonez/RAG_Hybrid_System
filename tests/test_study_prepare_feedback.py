"""«Preparar sesión» gives wait feedback and survives a second click (F-023, ADR-0036).

The real preparation takes about 52 s. Before F-023 the page showed nothing while it ran, and a
second click started another Streamlit run (fastReruns) that hit FileLock(timeout=0) and showed
the generic «No se pudo continuar» error.
"""
from types import SimpleNamespace

from filelock import Timeout
from streamlit.testing.v1 import AppTest

from src.ui.components import study_runtime as runtime
from src.ui.components.study_sessions import StudyStore
from src.ui.views import study_page
from tests.study_helpers import configured

CODE = ('import streamlit as st\nfrom src.ui.views.study_page import render\n'
        'st.navigation([st.Page(render)]).run()\nst.stop()')


def logged_in(tmp_path, monkeypatch, *, ready, prepare):
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol, 'technical')
    store.freeze()
    token = store.issue('P900', cell=1, profile='without_experience')
    monkeypatch.setattr(runtime, 'store_from_env', lambda: store)
    monkeypatch.setattr(runtime, 'preparation', lambda root: SimpleNamespace(ready=ready, pipeline=None))
    monkeypatch.setattr(runtime, 'prepare', prepare)
    app = AppTest.from_string(CODE).run()
    app.text_input[0].set_value(token)
    next(b for b in app.button if b.label == 'Entrar').click().run()
    assert not app.exception
    return app


def labels(app):
    return [b.label for b in app.button]


def infos(app):
    return [x.value for x in app.info]


def test_concurrent_preparation_shows_progress_not_the_generic_error(tmp_path, monkeypatch):
    def locked(session):
        raise Timeout('_inference.lock')

    app = logged_in(tmp_path, monkeypatch, ready=lambda scope: False, prepare=locked)
    next(b for b in app.button if b.label == 'Preparar sesión').click().run()

    assert not app.exception
    assert not app.error
    assert study_page.IN_PROGRESS_TEXT in infos(app)
    assert 'Comprobar preparación' in labels(app)
    assert app.session_state[study_page.PREPARING_KEY] is False


def test_preparation_in_progress_does_not_offer_or_start_a_second_one(tmp_path, monkeypatch):
    calls = []
    app = logged_in(tmp_path, monkeypatch, ready=lambda scope: False, prepare=calls.append)
    app.session_state[study_page.PREPARING_KEY] = True
    app.run()

    assert 'Preparar sesión' not in labels(app)
    assert 'Comprobar preparación' in labels(app)
    assert study_page.IN_PROGRESS_TEXT in infos(app)
    assert calls == []


def test_successful_preparation_runs_once_and_opens_familiarization(tmp_path, monkeypatch):
    state = {'ready': False, 'calls': 0}

    def prepare(session):
        state['calls'] += 1
        state['ready'] = True

    app = logged_in(tmp_path, monkeypatch, ready=lambda scope: state['ready'], prepare=prepare)
    next(b for b in app.button if b.label == 'Preparar sesión').click().run()

    assert not app.exception
    assert not app.error
    assert state['calls'] == 1
    assert 'Probar consulta' in labels(app)
    assert app.session_state[study_page.PREPARING_KEY] is False


def test_unexpected_preparation_failure_still_shows_the_neutral_error(tmp_path, monkeypatch):
    def broken(session):
        raise RuntimeError('warmup failed')

    app = logged_in(tmp_path, monkeypatch, ready=lambda scope: False, prepare=broken)
    next(b for b in app.button if b.label == 'Preparar sesión').click().run()

    assert not app.exception
    assert [x.value for x in app.error] == ['No se pudo continuar. Contacta al coordinador antes de reintentar.']
    assert app.session_state[study_page.PREPARING_KEY] is False
