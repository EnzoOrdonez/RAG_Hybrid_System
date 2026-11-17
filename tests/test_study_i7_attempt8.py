"""Attempt 8: configurable synthetic smoke code, VM stop on failure, fast socket errors, redacted dumps."""
import contextlib
import json
import sys
import time
import types
from datetime import datetime, timezone

import pytest

from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_deployment import config as deployment_config

TOKEN = 'synthetic-invitation-token-123'
TYPED = 'synthetic typed question'


class Owner:
    def __init__(self, root, **config):
        self.root, self.calls = root, []
        self.config = dict({'image_id': 'sha256:'+'a'*64, 'commit': 'b'*40, 'hostname': 'localhost',
                            'local_reviewed_config': 'reviewed.json'}, **config)
        self.state = {'ready': {'boot_id': 'boot'}, 'purpose': 'smoke'}

    def stop(self):
        self.calls.append('stop')

    def start(self, purpose='technical'):
        self.calls.append('start:'+purpose)

    def invite(self, code, **kwargs):
        self.calls.append('invite:'+code)
        return TOKEN

    def bridge(self, *args, **kwargs):
        return {}

    def receipt(self, kind, value):
        self.calls.append('receipt:'+kind)
        return dict(value, kind=kind)

    def now(self):
        return datetime(2026, 11, 1, tzinfo=timezone.utc)

    def delete_sessions(self, operation, *, code=None, dry_run=True):
        self.calls.append(('delete', operation, code, dry_run))
        return dict.fromkeys(('remote_versions_empty', 'zero_retention_policy_verified',
                              'normal_objects_empty', 'disk_empty'), True)

    def restore(self, code, session_id, **kwargs):
        self.calls.append(('restore', code, session_id))
        return {'status': 'RESTORED_VERIFIED', 'export_sha256': 'e'*64, 'new_app_instance': True,
                'original_store_unchanged': True, 'invitation_revoked': True}


def test_smoke_participant_defaults_to_p999_and_accepts_other_synthetic_codes():
    from scripts.study_operator.i7_smoke import smoke_participant
    assert smoke_participant({}) == 'P999'
    assert smoke_participant({'smoke_participant': 'P998'}) == 'P998'


@pytest.mark.parametrize('code', ['P12', 'PX', 'P899', 'P1234567', 998])
def test_smoke_participant_rejects_malformed_or_real_codes(code):
    from scripts.study_operator.deployment import checked_config
    from scripts.study_operator.i7_smoke import smoke_participant
    with pytest.raises(OperatorError):
        smoke_participant({'smoke_participant': code})
    with pytest.raises(OperatorError):
        checked_config(dict(deployment_config(), smoke_participant=code))


def test_installation_check_accepts_optional_synthetic_smoke_participant():
    from scripts.study_operator.deployment import checked_config
    assert checked_config(dict(deployment_config(), smoke_participant='P998'))['smoke_participant'] == 'P998'
    assert 'smoke_participant' not in checked_config(deployment_config())


def smoke_env(monkeypatch, browse):
    from scripts.study_operator import i7_browser, i7_smoke, i7_workflows
    from src.ui.components import study_protocol
    monkeypatch.setattr(i7_smoke, 'validate_receipt', lambda path: {})
    monkeypatch.setattr(i7_workflows, 'wait_ready', lambda operator: operator.calls.append('ready'))
    monkeypatch.setattr(study_protocol, 'verify_draw', lambda path: {'config': {}})
    monkeypatch.setattr(i7_browser, 'run', browse)
    return i7_smoke


def test_smoke_stops_vm_when_browser_fails_and_passes_participant_and_dump_dir(tmp_path, monkeypatch):
    seen = {}
    def browse(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError('UI_BROKEN')
    smoke = smoke_env(monkeypatch, browse)
    owner = Owner(tmp_path, smoke_participant='P998')
    with pytest.raises(RuntimeError, match='UI_BROKEN'):
        smoke.run(owner)
    assert owner.calls == ['stop', 'start:smoke', 'ready', 'invite:P998', 'stop']
    assert seen['participant'] == 'P998'
    assert seen['failure_dir'] == tmp_path/'private'/'browser-failures'


def test_smoke_success_returns_receipt_and_keeps_vm_for_restore_step(tmp_path, monkeypatch):
    smoke = smoke_env(monkeypatch, lambda *a, **k: {'status': 'BROWSER_EXPORT_VERIFIED'})
    owner = Owner(tmp_path)
    result = smoke.run(owner)
    assert result['status'] == 'BROWSER_EXPORT_VERIFIED' and result['boot_id'] == 'boot'
    assert owner.calls == ['stop', 'start:smoke', 'ready', 'invite:P999', 'receipt:smoke']


def test_invalid_smoke_participant_fails_before_any_vm_call(tmp_path, monkeypatch):
    smoke = smoke_env(monkeypatch, lambda *a, **k: pytest.fail('browser reached'))
    owner = Owner(tmp_path, smoke_participant='P100')
    with pytest.raises(OperatorError):
        smoke.run(owner)
    assert owner.calls == []


def test_privacy_step_stops_vm_when_canary_fails(tmp_path, monkeypatch):
    from scripts.study_operator import i7_phase_b, i7_workflows
    monkeypatch.setattr(i7_workflows, 'wait_ready', lambda operator: operator.calls.append('ready'))
    def privacy(operator):
        raise ValueError('CANARY_INCOMPLETE')
    monkeypatch.setattr(i7_workflows, 'privacy', privacy)
    owner = Owner(tmp_path)
    with pytest.raises(ValueError, match='CANARY_INCOMPLETE'):
        i7_phase_b.dispatch(owner, 'privacy', {})
    assert owner.calls == ['stop', 'start:smoke', 'ready', 'stop']


def test_tls_stops_running_vm_before_first_start(tmp_path, monkeypatch):
    from scripts.study_operator import i7_phase_b, i7_workflows
    monkeypatch.setattr(i7_workflows, 'wait_ready', lambda operator: operator.calls.append('ready'))
    monkeypatch.setattr(i7_workflows, 'check_tls_sequence', lambda attempts: {'status': 'TLS', 'n': len(attempts)})
    owner = Owner(tmp_path)
    assert i7_phase_b.tls(owner)['n'] == 3
    assert owner.calls[:2] == ['stop', 'start:technical']
    assert owner.calls.count('start:technical') == 3 and owner.calls[-1] == 'receipt:tls'


def test_withdraw_and_restore_use_configured_participant(tmp_path):
    from scripts.study_operator.i7_phase_b import restore_latest, withdraw_purge
    owner = Owner(tmp_path, smoke_participant='P998')
    withdraw_purge(owner)
    assert [c[2] for c in owner.calls if c[0] == 'delete' and c[1] == 'withdraw'] == ['P998', 'P998']
    folder = tmp_path/'receipts'/'smoke'
    folder.mkdir(parents=True)
    (folder/'0.json').write_text(json.dumps({'session_id': 'c'*32, 'export_sha256': 'e'*64,
        'objects': {'full_session.json': {'generation': '1'}, 'export_manifest.json': {'generation': '2'}}}))
    restore_latest(owner)
    assert ('restore', 'P998', 'c'*32) in owner.calls


class Element:
    def __init__(self, text='', *, value='', kind='text', label=None, visible=True, disabled=False):
        self.text, self.value, self.kind, self.label = text, value, kind, label
        self.visible, self.disabled = visible, disabled

    def is_visible(self):
        return self.visible

    def is_disabled(self):
        return self.disabled

    def inner_text(self):
        return self.text

    def input_value(self):
        return self.value

    def get_attribute(self, name):
        return {'type': self.kind, 'aria-label': self.label}.get(name)


class Group:
    def __init__(self, items=(), aria=''):
        self.items, self.aria = list(items), aria

    def all(self):
        return self.items

    def all_inner_texts(self):
        return [item.text for item in self.items]

    def aria_snapshot(self):
        return self.aria


class Page:
    url = 'https://localhost/?invite=hidden'

    def __init__(self, *, broken_screenshot=False, fail_on_goto=False):
        self.handlers, self.timeouts, self.waits, self.roles = {}, [], 0, []
        self.broken_screenshot, self.fail_on_goto = broken_screenshot, fail_on_goto
        self.fields = [Element(value=TOKEN, kind='password', label='Invitación'),
                       Element(value=TYPED, label='Tu consulta')]
        self.buttons = [Element('Entrar'), Element('Oculto', visible=False), Element('Consultar', disabled=True)]

    def on(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    def set_default_timeout(self, value):
        self.timeouts.append(value)

    def wait_for_timeout(self, value):
        self.waits += 1

    def goto(self, url, **kwargs):
        raise RuntimeError('ORIGINAL_UI_FAILURE')

    def screenshot(self, *, path, full_page):
        if self.broken_screenshot:
            raise OSError('screenshot failed')
        assert full_page
        open(path, 'wb').write(b'png')

    def locator(self, selector):
        if selector == 'body':
            return Group(aria='- textbox "Invitación":\n  - text: '+TOKEN+'\n- textbox "Tu consulta": "'
                         +TYPED+'"\n- \'textbox "a: b"\': other\n- button "Entrar"\n- text: echo '+TOKEN)
        if selector == 'label':
            return Group([Element('Invitación'), Element('Tu consulta')])
        return Group(self.fields)

    def get_by_role(self, role, **kwargs):
        self.roles.append((role, kwargs))
        return Group(self.buttons)


def written(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def test_failure_dump_writes_four_redacted_files(tmp_path):
    from scripts.study_operator.i7_browser import dump_failure
    events = {'console': [{'type': 'log', 'text': 'token='+TOKEN}], 'pageerror': [],
              'websockets': ['wss://localhost/_stcore/stream'], 'websocket_closed': ['wss://localhost/_stcore/stream']}
    folder = dump_failure(Page(), tmp_path/'failures', (TOKEN, None), events, 'STREAMLIT_SOCKET_CLOSED')
    assert folder.parent == tmp_path/'failures'
    files = written(folder)
    assert set(files) == {'screenshot.png', 'aria.yaml', 'controls.json', 'console.json'}
    text = b''.join(files[name] for name in ('aria.yaml', 'controls.json', 'console.json')).decode()
    assert TOKEN not in text and TYPED not in text and 'other' not in text and 'invite=' not in text
    assert files['aria.yaml'].decode().count('<redacted>') >= 4
    controls = json.loads(files['controls.json'])
    assert controls['path'] == '/'
    assert controls['buttons'] == [{'text': 'Entrar', 'disabled': False}, {'text': 'Consultar', 'disabled': True}]
    assert controls['labels'] == ['Invitación', 'Tu consulta']
    assert controls['inputs'][0] == {'type': 'password', 'aria_label': 'Invitación', 'disabled': False}
    console = json.loads(files['console.json'])
    assert console['sync_error'] == 'STREAMLIT_SOCKET_CLOSED' and console['websocket_closed']
    assert console['console'][0]['text'] == 'token=<redacted>'
    with pytest.raises(FileExistsError):
        folder.mkdir(exist_ok=False)


def test_failure_dump_errors_never_escape(tmp_path):
    from scripts.study_operator.i7_browser import dump_failure
    folder = dump_failure(Page(broken_screenshot=True), tmp_path, [TOKEN], {}, None)
    files = written(folder)
    assert 'screenshot.png' not in files and 'aria.yaml' in files
    assert json.loads(files['dump-errors.json']) == {'failed': ['screenshot.png']}


def test_browser_failure_is_dumped_and_original_error_reraised(tmp_path, monkeypatch):
    from scripts.study_operator import i7_browser
    page = Page(broken_screenshot=True)
    class Browser:
        def new_context(self, **kwargs):
            return types.SimpleNamespace(new_page=lambda: page)
        def close(self):
            page.closed = True
    chromium = types.SimpleNamespace(executable_path='chrome', launch=lambda **kwargs: Browser())
    module = types.ModuleType('playwright.sync_api')
    module.sync_playwright = lambda: contextlib.nullcontext(types.SimpleNamespace(chromium=chromium))
    monkeypatch.setitem(sys.modules, 'playwright', types.ModuleType('playwright'))
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', module)
    monkeypatch.setattr(i7_browser, 'runtime_identity', lambda pw: {'version': 'fake'})
    identity = {'boot_id': 'boot'}
    with pytest.raises(RuntimeError, match='ORIGINAL_UI_FAILURE'):
        i7_browser.run('https://localhost', TOKEN, {'config': {}}, identity, lambda: {'identity': identity},
                       participant='P998', failure_dir=tmp_path/'failures')
    assert page.timeouts[0] == 60000 and page.closed
    (folder,) = (tmp_path/'failures').iterdir()
    assert {'aria.yaml', 'controls.json', 'console.json', 'dump-errors.json'} <= set(written(folder))
    assert all(TOKEN.encode() not in data for data in written(folder).values())


def test_button_wait_polls_and_raises_socket_error_promptly():
    from scripts.study_operator.i7_browser import StreamlitSync, visible
    page = Page()
    page.buttons = Element('Consultar', visible=False)
    page.get_by_role = lambda role, **kwargs: (page.roles.append((role, kwargs)), page.buttons)[1]
    sync = StreamlitSync(page, 600)
    def tick(value):
        page.waits += 1
        if page.waits == 3:
            sync.failed('STREAMLIT_SOCKET_CLOSED')
    page.wait_for_timeout = tick
    started = time.monotonic()
    with pytest.raises(ValueError, match='STREAMLIT_SOCKET_CLOSED'):
        visible(page, sync, 'Consultar')
    assert time.monotonic()-started < 5 and page.waits == 3
    assert page.roles == [('button', {'name': 'Consultar', 'exact': True})]


def test_button_wait_returns_exact_button_once_visible():
    from scripts.study_operator.i7_browser import StreamlitSync, visible
    page = Page()
    button = Element('Continuar', visible=False)
    page.get_by_role = lambda role, **kwargs: button
    sync = StreamlitSync(page, 600)
    def tick(value):
        button.visible = True
    page.wait_for_timeout = tick
    assert visible(page, sync, 'Continuar') is button
    with pytest.raises(ValueError, match='BUTTON_NOT_VISIBLE'):
        button.visible = False
        page.wait_for_timeout = lambda value: None
        visible(page, sync, 'Continuar', 0.05)


def test_button_visibility_waits_90_s_by_default_and_900_s_only_when_asked():
    from scripts.study_operator import i7_browser
    class RecordingSync:
        def __init__(self, timeout_s):
            self.timeout_s, self.seen = timeout_s, []
        def until(self, predicate, timeout_s=None, code=None):
            self.seen.append(timeout_s)
    class Page:
        def get_by_role(self, role, name, exact):
            return Element(name)
    cloud, local = RecordingSync(600), RecordingSync(25)
    i7_browser.visible(Page(), cloud, 'Entrar')
    i7_browser.visible(Page(), local, 'Entrar')
    i7_browser.visible(Page(), cloud, 'Probar consulta', 900)
    assert i7_browser.BUTTON_WAIT_S == 90
    assert cloud.seen == [90, 900]
    assert local.seen == [25]
