"""I8 hardening (001-004 and I8-09): each test fails on 3bb9ca8 without its change."""
import contextlib
import json
import sys
import types

import pytest

from tests.test_study_i7_attempt8 import TOKEN, Owner, Page, smoke_env, written

HOST = '203.0.113.7.sslip.io'  # TEST-NET-3 documentation address, not a real VM


# 001: the local branch that line 90 clones must exist and match HEAD before any payment.
def test_build_requires_local_branch_before_payment(tmp_path, monkeypatch):
    from scripts.study_operator import i7_cpu_pipeline, i7_streamlit_preflight
    head = 'c'*40
    monkeypatch.setattr(i7_streamlit_preflight, 'verify_before_cpu', lambda: {})
    monkeypatch.setattr(i7_cpu_pipeline.subprocess, 'check_output',
                        lambda cmd, text: head if 'rev-parse' in cmd else head+'\trefs/heads/fix/interview-readiness\n')
    monkeypatch.setattr(i7_cpu_pipeline.subprocess, 'run',
                        lambda *a, **k: types.SimpleNamespace(stdout='', returncode=1))
    owner = Owner(tmp_path, commit=head)
    (tmp_path/'cpu').mkdir()  # without the guard the next check (attempt exists) fires instead
    with pytest.raises(ValueError, match='PUBLISHED_I7_SOURCE_REQUIRED'):
        i7_cpu_pipeline.build(owner, {})
    assert owner.calls == ['receipt:streamlit-prebuild']


class FailingStart(Owner):
    def start(self, purpose='technical'):
        self.calls.append('start:'+purpose)
        raise TimeoutError('instances start timed out')  # the VM may still boot


# 002: a failed start('smoke') still stops the VM, in privacy and in smoke.
def test_privacy_stops_vm_when_start_itself_fails(tmp_path, monkeypatch):
    from scripts.study_operator import i7_phase_b
    owner = FailingStart(tmp_path)
    with pytest.raises(TimeoutError):
        i7_phase_b.dispatch(owner, 'privacy', {})
    assert owner.calls == ['stop', 'start:smoke', 'stop']


def test_smoke_stops_vm_when_start_itself_fails(tmp_path, monkeypatch):
    smoke = smoke_env(monkeypatch, lambda *a, **k: pytest.fail('browser reached'))
    owner = FailingStart(tmp_path)
    with pytest.raises(TimeoutError):
        smoke.run(owner)
    assert owner.calls == ['stop', 'start:smoke', 'stop']


class Request:
    def __init__(self, method, url, failure=None):
        self.method, self.url, self.failure = method, url, failure


class NetworkPage(Page):
    """goto emits one finished, one failed and one pending request, then fails."""
    def goto(self, url, **kwargs):
        done, lost, hung = (Request('GET', 'https://'+HOST+path) for path in
                            ('/index.html', '/static/js/chunk1.js', '/static/js/chunk2.js'))
        lost.failure = 'net::ERR_QUIC_PROTOCOL_ERROR'
        for request in (done, lost, hung):
            for handler in self.handlers.get('request', []):
                handler(request)
        for handler in self.handlers.get('requestfinished', []):
            handler(done)
        for handler in self.handlers.get('requestfailed', []):
            handler(lost)
        raise RuntimeError('ORIGINAL_UI_FAILURE')


class ConsolePage(Page):
    """The browser logs the origin (as Streamlit does on socket errors), then the UI fails."""
    def goto(self, url, **kwargs):
        for handler in self.handlers.get('console', []):
            handler(types.SimpleNamespace(type='error', text='WebSocket wss://'+HOST+'/_stcore/stream failed'))
        raise RuntimeError('ORIGINAL_UI_FAILURE')


def run_failing(tmp_path, monkeypatch, page, observe):
    from scripts.study_operator import i7_browser
    launches = []
    class Browser:
        def new_context(self, **kwargs):
            return types.SimpleNamespace(new_page=lambda: page)
        def close(self):
            pass
    chromium = types.SimpleNamespace(executable_path='chrome', launch=lambda **k: launches.append(k) or Browser())
    module = types.ModuleType('playwright.sync_api')
    module.sync_playwright = lambda: contextlib.nullcontext(types.SimpleNamespace(chromium=chromium))
    monkeypatch.setitem(sys.modules, 'playwright', types.ModuleType('playwright'))
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', module)
    monkeypatch.setattr(i7_browser, 'runtime_identity', lambda pw: {'version': 'fake'})
    with pytest.raises(RuntimeError, match='ORIGINAL_UI_FAILURE'):
        i7_browser.run('https://'+HOST, TOKEN, {'config': {}}, {'boot_id': 'boot'}, observe,
                       failure_dir=tmp_path/'failures')
    (folder,) = (tmp_path/'failures').iterdir()
    return written(folder), launches


# 003: neither the sslip hostname nor its embedded public IP reaches the dump.
def test_dump_redacts_sslip_host_and_ip(tmp_path, monkeypatch):
    observe = lambda: {'identity': {'boot_id': 'boot'}, 'data': {'origin': 'https://'+HOST, 'ip': '203.0.113.7'}}
    files, _ = run_failing(tmp_path, monkeypatch, ConsolePage(), observe)
    assert b'wss://<redacted>/_stcore/stream' in files['console.json']
    assert all(HOST.encode() not in d and b'203.0.113.7' not in d for d in files.values())


# 004: diagnostics.json carries sync counters and the server-side observe() state.
def test_dump_writes_sync_counters_and_observe_state(tmp_path, monkeypatch):
    observe = lambda: {'identity': {'boot_id': 'boot'}, 'data': {'stage': 'consent'}}
    files, _ = run_failing(tmp_path, monkeypatch, Page(), observe)
    diagnostics = json.loads(files['diagnostics.json'])
    assert diagnostics['sync'] == {'requests': 0, 'completed': 0, 'form_text_entries': 0, 'error': None}
    assert diagnostics['observe'] == {'identity_matches': True, 'data': {'stage': 'consent'}}


def test_dump_survives_observe_failure(tmp_path, monkeypatch):
    calls = []
    def observe():
        calls.append(1)
        if len(calls) > 1:
            raise PermissionError('locked')
        return {'identity': {'boot_id': 'boot'}}
    files, _ = run_failing(tmp_path, monkeypatch, Page(), observe)
    assert json.loads(files['diagnostics.json'])['observe'] == {'observe_error': 'PermissionError'}


# I8-09: failed and pending requests are logged (path only), finished ones are not.
def test_dump_logs_failed_and_pending_requests(tmp_path, monkeypatch):
    observe = lambda: {'identity': {'boot_id': 'boot'}}
    files, _ = run_failing(tmp_path, monkeypatch, NetworkPage(), observe)
    network = json.loads(files['diagnostics.json'])['network']
    assert [(f['path'], f['failure']) for f in network['failed']] == [
        ('/static/js/chunk1.js', 'net::ERR_QUIC_PROTOCOL_ERROR')]
    assert [p['path'] for p in network['pending']] == ['/static/js/chunk2.js']
    assert network['pending_total'] == 1 and network['failed_total'] == 1
    assert all(HOST.encode() not in d for d in files.values())


def test_network_log_is_bounded_and_never_waits():
    from scripts.study_operator.i7_browser import NETWORK_LOG_LIMIT, network_log
    network = {'pending': {i: ('GET', '/p%d' % i, float(i)) for i in range(NETWORK_LOG_LIMIT+10)},
               'failed': [{'path': '/f%d' % i} for i in range(NETWORK_LOG_LIMIT+5)]}
    log = network_log(network, now=100.0)
    assert len(log['pending']) == NETWORK_LOG_LIMIT and log['pending'][0] == {'method': 'GET', 'path': '/p0', 'age_s': 100.0}
    assert len(log['failed']) == NETWORK_LOG_LIMIT and log['failed'][-1] == {'path': '/f%d' % (NETWORK_LOG_LIMIT+4)}
    assert log['pending_total'] == NETWORK_LOG_LIMIT+10 and log['failed_total'] == NETWORK_LOG_LIMIT+5


# Transport (commit 2, not backed by evidence): Caddy serves only h1/h2 and Chromium never tries QUIC.
def test_caddy_disables_http3():
    from scripts.study_operator.deployment import caddyfile
    text = caddyfile(HOST)
    assert 'servers {\n        protocols h1 h2\n    }' in text and 'h3' not in text


def test_browser_launches_with_quic_disabled(tmp_path, monkeypatch):
    observe = lambda: {'identity': {'boot_id': 'boot'}}
    _, launches = run_failing(tmp_path, monkeypatch, Page(), observe)
    assert launches and '--disable-quic' in launches[0]['args']
