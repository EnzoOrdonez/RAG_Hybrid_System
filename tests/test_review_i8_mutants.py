"""Auditoría del intento 8: tests que matan mutantes supervivientes de los tests de 3bb9ca8.

Cada test nombra el mutante (docs/review/2026-11-02-i8/MUTACION.md) que la suite de 3bb9ca8 no detectaba.
Pasan sobre 3bb9ca8; fallan con el mutante aplicado.
"""
import contextlib
import sys
import types
from pathlib import Path

import pytest

from tests.test_study_i7_attempt8 import TOKEN, Group, Owner, Page, smoke_env


class NoFields(Page):
    """input_value() falla: solo quedan la redacción ARIA por rol y los secretos privados."""

    def __init__(self, aria, **kwargs):
        super().__init__(**kwargs)
        self.aria = aria

    def locator(self, selector):
        if selector == 'body':
            return Group(aria=self.aria)
        if selector == 'label':
            return Group([])
        raise RuntimeError('fields unavailable')


def dumped(tmp_path, page, private=(TOKEN,)):
    from scripts.study_operator.i7_browser import dump_failure
    folder = dump_failure(page, tmp_path, private, {}, None)
    return {p.name: p.read_bytes() for p in Path(folder).iterdir()}


def test_m11_smoke_stops_vm_on_keyboard_interrupt(tmp_path, monkeypatch):
    def browse(*args, **kwargs):
        raise KeyboardInterrupt
    smoke = smoke_env(monkeypatch, browse)
    owner = Owner(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        smoke.run(owner)
    assert owner.calls[-1] == 'stop'


def test_m13_privacy_stops_vm_on_keyboard_interrupt(tmp_path, monkeypatch):
    from scripts.study_operator import i7_phase_b, i7_workflows
    monkeypatch.setattr(i7_workflows, 'wait_ready', lambda operator: operator.calls.append('ready'))
    def privacy(operator):
        raise KeyboardInterrupt
    monkeypatch.setattr(i7_workflows, 'privacy', privacy)
    owner = Owner(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        i7_phase_b.dispatch(owner, 'privacy', {})
    assert owner.calls == ['stop', 'start:smoke', 'ready', 'stop']


def test_m17_child_lines_of_a_value_role_are_redacted_without_known_value(tmp_path):
    files = dumped(tmp_path, NoFields('- textbox "Nota":\n  - text: PRIVATE-CHILD-NOTE\n- button "Entrar"'))
    assert b'PRIVATE-CHILD-NOTE' not in files['aria.yaml']
    assert b'button "Entrar"' in files['aria.yaml']
    assert b'input_values' in files['dump-errors.json']


def test_m37_combobox_and_spinbutton_values_are_redacted_by_role(tmp_path):
    files = dumped(tmp_path, NoFields('- combobox "Perfil": SECRET-COMBO\n- spinbutton "Edad": 4242'))
    assert b'SECRET-COMBO' not in files['aria.yaml'] and b'4242' not in files['aria.yaml']


def test_m19_m20_whitespace_normalised_echo_of_a_typed_value_is_redacted(tmp_path):
    typed = 'linea uno\n   linea dos privada'
    page = Page()
    page.fields[1].value = typed
    page.locator = lambda selector, base=page.locator: (
        Group(aria='- paragraph: "Tu consulta: linea uno linea dos privada"') if selector == 'body'
        else base(selector))
    files = dumped(tmp_path, page)
    assert b'linea dos privada' not in files['aria.yaml']


def test_m26_dump_bounds_each_browser_call_to_dump_timeout(tmp_path):
    from scripts.study_operator.i7_browser import DUMP_TIMEOUT_MS
    page = Page()
    dumped(tmp_path, page)
    assert page.timeouts == [DUMP_TIMEOUT_MS] and DUMP_TIMEOUT_MS <= 5000


def test_m28_run_passes_the_invitation_token_to_the_dump(tmp_path, monkeypatch):
    """El token solo aparece en la consola (ningún campo lo contiene): debe venir de los secretos de run()."""
    from scripts.study_operator import i7_browser

    class ConsolePage(Page):
        def __init__(self):
            super().__init__()
            self.fields = []

        def goto(self, url, **kwargs):
            for handler in self.handlers.get('console', []):
                handler(types.SimpleNamespace(type='log', text='invite='+TOKEN))
            raise RuntimeError('ORIGINAL_UI_FAILURE')
    page = ConsolePage()

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
        i7_browser.run('https://localhost', TOKEN, {'config': {}}, identity, lambda: {'identity': identity},
                       failure_dir=tmp_path/'failures')
    (folder,) = (tmp_path/'failures').iterdir()
    console = (folder/'console.json').read_bytes()
    assert b'invite=<redacted>' in console and TOKEN.encode() not in console
