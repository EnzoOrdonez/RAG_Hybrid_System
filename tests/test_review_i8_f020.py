"""F-020 (intento 7, el login no ocurrió tras click('Entrar')): reproducción local y parche I8-06 propuesto.

- Caracterización con Chromium real (se omite si no hay Chromium): el recorrido exacto de i7_browser.run
  hace login contra src/ui/app.py. Un rechazo de admit() en el servidor produce el síntoma del intento 7:
  rerun enviado y completado, formulario intacto y mensaje genérico.
- I8-06 (xfail estricto): el volcado no guarda sync.requests/completed ni observe(), así que no
  distingue esos casos. Parche: docs/review/2026-11-02-i8/patches/004-i8-06-dump-sync-and-observe.patch.
"""
import contextlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

from tests.test_study_i7_attempt8 import TOKEN, Page

HARNESS = Path(__file__).resolve().parents[1]/'docs/review/2026-11-02-i8/f020/harness.py'


def harness():
    spec = importlib.util.spec_from_file_location('f020_harness', HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not Path(module.CHROME).exists():
        pytest.skip('Chromium del contenedor no disponible (F020_CHROME)')
    pytest.importorskip('playwright.sync_api')
    return module


def test_f020_exact_login_flow_logs_in_locally(tmp_path):
    record = harness().attempt('a_fill_click', 'app', tmp_path)
    assert record['error'] is None and record['login'] and record['stage'] == 'familiarization'
    reruns = [f for f in record['frames'] if f.get('type') == 'rerun_script']
    assert [w['value_is_token'] for w in reruns[-1]['widgets'] if w['kind'] == 'string_value'] == [True]


def test_f020_server_side_refusal_looks_like_a_click_without_effect(tmp_path):
    record = harness().attempt('e_expired', 'app', tmp_path)
    assert record['error'] is None and not record['login'] and record['sessions_created'] == 0
    assert record['requests'] == record['completed'] == 2       # el clic sí envió y completó un rerun
    assert 'Entrar' in record['buttons_after']
    assert record['error_text'] == ['No se pudo abrir la sesión. Contacta al coordinador.']


@pytest.mark.xfail(strict=True, reason='I8-06: el volcado no guarda sync.requests/completed ni observe(); '
                                       'no separa clic no enviado, rerun sin terminar y rechazo de admit()')
def test_i8_06_failure_dump_records_sync_counters_and_app_state(tmp_path, monkeypatch):
    from scripts.study_operator import i7_browser
    page = Page()

    class Browser:
        def new_context(self, **kwargs):
            return types.SimpleNamespace(new_page=lambda: page)

        def close(self):
            pass
    chromium = types.SimpleNamespace(executable_path='chrome', launch=lambda **kwargs: Browser())
    module = types.ModuleType('playwright.sync_api')
    module.sync_playwright = lambda: contextlib.nullcontext(types.SimpleNamespace(chromium=chromium))
    monkeypatch.setitem(sys.modules, 'playwright', types.ModuleType('playwright'))
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', module)
    monkeypatch.setattr(i7_browser, 'runtime_identity', lambda pw: {'version': 'fake'})
    identity = {'boot_id': 'boot'}
    with pytest.raises(RuntimeError, match='ORIGINAL_UI_FAILURE'):
        i7_browser.run('https://localhost', TOKEN, {'config': {}}, identity,
                       lambda: {'identity': identity, 'data': None}, failure_dir=tmp_path/'failures')
    (folder,) = (tmp_path/'failures').iterdir()
    assert all(TOKEN.encode() not in p.read_bytes() for p in folder.iterdir())
    diagnostics = json.loads((folder/'diagnostics.json').read_bytes())
    assert diagnostics['sync'] == {'requests': 0, 'completed': 0, 'form_text_entries': 0, 'error': None}
    assert diagnostics['observe'] == {'identity_matches': True, 'data': None}


def test_f020_harness_never_writes_the_token():
    """El arnés afirma internamente que el token no aparece en su registro; aquí se revisan los resultados versionados."""
    folder = HARNESS.parent/'results'
    files = list(folder.rglob('*.json'))
    assert files
    for path in files:
        record = json.loads(path.read_bytes())
        assert set(record) >= {'variant', 'server', 'login', 'frames'}
        assert os.path.basename(path) == record['server']+'-'+record['variant']+'.json'


def test_f020b_button_chunk_failure_reproduces_attempt7_call_log(tmp_path):
    """Intento 7: fill('Invitación') funciona y click('Entrar') agota el tiempo sin 'locator resolved'.
    Mecanismo reproducido: el chunk JS de stFormSubmitButton falla (abort) o queda pendiente (hang)."""
    module = harness()
    spec = importlib.util.spec_from_file_location('f020_chunk', HARNESS.with_name('chunk.py'))
    chunk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunk)
    chunk.main(tmp_path/'chunk.json')
    rows = {r['mode']: r for r in json.loads((tmp_path/'chunk.json').read_bytes())}
    assert rows['none']['click'] == 'ok' and not rows['none']['exact_match_attempt7']
    for mode, signature in (('abort', 'Failed to fetch dynamically imported module'), ('hang', 'Missing Submit Button')):
        row = rows[mode]
        assert row['exact_match_attempt7'], row
        assert row['call_log'] == ['- waiting for get_by_role("button", name="Entrar", exact=True)']
        assert signature in row['aria'] and 'textbox "Invitación"' in row['aria']
    assert module.CHROME
