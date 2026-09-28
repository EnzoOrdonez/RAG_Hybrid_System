import ast
import json
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from src.ui.components import study_runtime as runtime
from src.ui.components.study_sessions import StudyStore
from tests.study_helpers import configured


def click(app, label):
    next(b for b in app.button if b.label == label).click().run()
    assert not app.exception


def test_two_blocks_app_disconnect_export_and_practice_privacy(tmp_path, monkeypatch):
    c, a, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol, 'technical')
    store.freeze()
    token = store.issue('P900', cell=1, profile='without_experience')
    monkeypatch.setattr(runtime, 'store_from_env', lambda: store)
    calls = []
    def pipeline(condition, scope):
        def query(q):
            calls.append((condition, q))
            return SimpleNamespace(answer='PRACTICE_ONLY' if q == protocol['config']['familiarization'] else 'Answer',
                sources=[dict(provider=None, service='N/A', section='Source')], error=None,
                confidence='HIGH', hallucination_report={'method': 'nli'})
        return SimpleNamespace(query=query)
    monkeypatch.setattr(runtime, 'preparation', lambda root: SimpleNamespace(ready=lambda scope: True, pipeline=pipeline))
    monkeypatch.setattr(runtime, 'render_wait', lambda start: None)
    code = 'import streamlit as st\nfrom src.ui.views.study_page import render\nst.navigation([st.Page(render)]).run()\nst.stop()'
    app = AppTest.from_string(code).run()
    app.text_input[0].set_value(token)
    click(app, 'Entrar')
    for block in range(2):
        click(app, 'Probar consulta')
        click(app, 'Comenzar tareas')
        for task in range(4):
            if task == 3:
                assert any(protocol['config']['free_instruction'] == x.value for x in app.markdown)
                app.text_area[0].set_value('FREE QUERY')
            click(app, 'Consultar')
            if block == 0 and task == 0:
                before = len(calls)
                app = AppTest.from_string(code).run()
                app.text_input[0].set_value(token)
                click(app, 'Entrar')
                assert len(calls) == before
            assert not any('None' in x.value or 'N/A' in x.value for x in app.markdown)
            if block == 1:
                assert not app.expander
            click(app, 'Continuar')
        assert len(app.radio) == 20
        for r in app.radio:
            r.set_value(3)
        click(app, 'Guardar respuestas del bloque')
    assert len(app.radio) == 3, [(r.label, r.options) for r in app.radio]
    for r in app.radio:
        r.set_value('A')
    app.text_area[0].set_value('Comparison')
    click(app, 'Continuar al cierre')
    app.radio[0].set_value('Sistema A')
    click(app, 'Finalizar')
    exported = next(store.root.glob('*/full_session.json'))
    payload = json.loads(exported.read_text(encoding='utf-8'))
    assert payload['stage'] == 'complete' and len(payload['attempts']) == 8
    assert len(payload['instruments']) == 2 and len(calls) == 10
    assert 'PRACTICE_ONLY' not in exported.read_text(encoding='utf-8')
    assert protocol['config']['familiarization'] not in exported.read_text(encoding='utf-8')


def test_participant_templates_have_no_condition_hints():
    path = Path(__file__).parents[1] / 'src/ui/views/study_page.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    forbidden = ('recuperación', 'rag', 'alucinación', 'híbrido', 'léxico', 'semántico', 'denso')
    # Text literals in the participant template, including errors/help.
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not any(word in node.value.casefold() for word in forbidden)


def test_empty_sus_blocks_login(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, 'store_from_env', lambda: (_ for _ in ()).throw(ValueError('SUS')))
    app = AppTest.from_string('from src.ui.views.study_page import render; render()').run()
    assert app.error and not app.text_input and not app.exception


def test_stale_block_form_has_neutral_error_and_no_duplicate_instruments(tmp_path, monkeypatch):
    from src.ui.components import study_service as service
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', protocol)
    store.freeze()
    token = store.issue('P01')
    session = store.admit(token)
    session.familiarization_done()
    for _ in range(4):
        service.answer(session, lambda _: SimpleNamespace(query=lambda q: SimpleNamespace(
            answer='ok', error=None, confidence='HIGH', sources=[], hallucination_report={})), 'free')
        session.shown()
        session.acknowledge()
    monkeypatch.setattr(runtime, 'store_from_env', lambda: store)
    app = AppTest.from_string('import streamlit as st\nfrom src.ui.views.study_page import render\nst.navigation([st.Page(render)]).run()\nst.stop()').run()
    app.text_input[0].set_value(token)
    click(app, 'Entrar')
    # Another tab advances the revision after this form was rendered.
    store.admit(token).incident()
    for r in app.radio:
        r.set_value(3)
    click(app, 'Guardar respuestas del bloque')
    assert app.error and not app.exception
    assert store.admit(token).data['instruments'] == []
