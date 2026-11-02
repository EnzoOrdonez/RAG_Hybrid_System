"""Auditoría del intento 8 (3bb9ca8). Defectos demostrables como xfail estricto; el código bajo prueba no se toca.

I8-01  cpu-build clona la rama LOCAL fix/interview-readiness por nombre después de admit_paid.
I8-02  smoke/privacy: start() queda fuera del try, así que un timeout de `instances start` no para la VM.
I8-03  el volcado del navegador escribe la IP pública real (hostname <IP>.sslip.io).
I8-04  si stop() falla en el except, la causa original del smoke queda oculta (patrón B-2).
"""
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tests.test_study_i7_attempt8 import TOKEN, Owner, Page, smoke_env

PUBLIC_IP = '203.0.113.45'


# --- I8-01 -----------------------------------------------------------------------------------

def git(cwd, *args):
    return subprocess.run(['git', '-c', 'user.name=review-i8', '-c', 'user.email=review-i8',
                           '-c', 'init.defaultBranch=main', '-c', 'commit.gpgsign=false', *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class Paid(Exception):
    pass


def operator_checkout(tmp_path, *, local_branch):
    """Bare origin with fix/interview-readiness fast-forwarded to HEAD; operator on fix/i7-attempt8."""
    origin, work = tmp_path/'origin.git', tmp_path/'work'
    git(tmp_path, 'init', '--bare', str(origin))
    git(tmp_path, 'clone', str(origin), str(work))
    (work/'a.txt').write_text('b470d3e')
    git(work, 'checkout', '-b', 'fix/interview-readiness')
    git(work, 'add', '.')
    git(work, 'commit', '-m', 'base')
    old = git(work, 'rev-parse', 'HEAD')
    git(work, 'push', 'origin', 'fix/interview-readiness')
    git(work, 'checkout', '-b', 'fix/i7-attempt8')
    (work/'a.txt').write_text('3bb9ca8')
    git(work, 'commit', '-am', 'attempt 8')
    head = git(work, 'rev-parse', 'HEAD')
    git(work, 'push', 'origin', 'HEAD:refs/heads/fix/interview-readiness')  # fast-forward remoto
    if local_branch == 'missing':
        git(work, 'branch', '-D', 'fix/interview-readiness')
    assert git(work, 'ls-remote', 'origin', 'refs/heads/fix/interview-readiness').split()[0] == head
    return work, old, head


@pytest.mark.parametrize('local_branch', ['stale', 'missing'])
@pytest.mark.xfail(strict=True, reason='I8-01: la rama local fix/interview-readiness se clona tras admit_paid '
                                       '(obsoleta: construye b470d3e; ausente: git clone rc 128 con cpu/ creado)')
def test_i8_01_cpu_build_refuses_before_payment_unless_local_branch_is_head(tmp_path, monkeypatch, local_branch):
    from scripts.study_operator import i7_cpu_pipeline, pricing
    from scripts.study_operator import cpu_restoration
    work, old, head = operator_checkout(tmp_path, local_branch=local_branch)
    preflight = types.ModuleType('scripts.study_operator.i7_streamlit_preflight')
    preflight.verify_before_cpu = lambda: {'files': {}}
    preflight.verify_recipe = lambda path: None
    monkeypatch.setitem(sys.modules, 'scripts.study_operator.i7_streamlit_preflight', preflight)
    monkeypatch.setattr(i7_cpu_pipeline, '__file__', str(work/'scripts'/'study_operator'/'i7_cpu_pipeline.py'))
    monkeypatch.setattr(cpu_restoration, 'verified_closure', lambda path: None)
    quote = {'usd_per_hour': '1', 'catalog_receipt_sha256': 'r'}
    monkeypatch.setattr(pricing, 'quote_archive', lambda *a: quote)
    monkeypatch.setattr(pricing, 'disk_quote_archive', lambda *a: {'estimated_usd_gib_h': '0'})

    class Operator:
        root = tmp_path/'root'
        config = {'commit': head, 'rates': {'price_receipt_sha256': 'r', 'compute_usd_h': 1.0}}
        state = {}

        def receipt(self, kind, value):
            return value

        def admit_paid(self, exposure):
            raise Paid('admit_paid alcanzado')
    Operator.root.mkdir()
    # Comportamiento deseado: rechazo SIN gasto, antes de admit_paid.
    with pytest.raises(ValueError):
        i7_cpu_pipeline.build(Operator(), {'inherited_package': str(tmp_path), 'price_archive': str(tmp_path)})


def test_i8_01_characterization_remote_not_fast_forwarded_fails_before_payment(tmp_path, monkeypatch):
    """Escenario A de §2.5 (remoto en b470d3e): la guarda de :46 sí corta antes de pagar. Esto no es defecto."""
    from scripts.study_operator import i7_cpu_pipeline
    work, old, head = operator_checkout(tmp_path, local_branch='stale')
    git(work, 'push', '--force', 'origin', old+':refs/heads/fix/interview-readiness')
    preflight = types.ModuleType('scripts.study_operator.i7_streamlit_preflight')
    preflight.verify_before_cpu = lambda: {'files': {}}
    preflight.verify_recipe = lambda path: None
    monkeypatch.setitem(sys.modules, 'scripts.study_operator.i7_streamlit_preflight', preflight)
    monkeypatch.setattr(i7_cpu_pipeline, '__file__', str(work/'scripts'/'study_operator'/'i7_cpu_pipeline.py'))
    operator = types.SimpleNamespace(root=tmp_path, config={'commit': head}, receipt=lambda k, v: v,
                                     admit_paid=lambda x: pytest.fail('admit_paid alcanzado'))
    with pytest.raises(ValueError, match='PUBLISHED_I7_SOURCE_REQUIRED'):
        i7_cpu_pipeline.build(operator, {})


# --- I8-02 -----------------------------------------------------------------------------------

class StartTimesOut(Owner):
    """gcloud `instances start` matado por timeout=300 (i7.py:383): la operación GCE sigue y la VM arranca."""

    def start(self, purpose='technical'):
        self.calls.append('start:'+purpose)
        raise subprocess.TimeoutExpired(['gcloud', 'compute', 'instances', 'start'], 300)


@pytest.mark.xfail(strict=True, reason='I8-02: i7_smoke.py:35 llama a start() fuera del try; sin stop() tras el timeout')
def test_i8_02_smoke_stops_vm_when_start_times_out(tmp_path, monkeypatch):
    smoke = smoke_env(monkeypatch, lambda *a, **k: pytest.fail('browser reached'))
    owner = StartTimesOut(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        smoke.run(owner)
    assert owner.calls[-1] == 'stop', owner.calls


@pytest.mark.xfail(strict=True, reason='I8-02: i7_phase_b.py:247 llama a start() fuera del try; sin stop() tras el timeout')
def test_i8_02_privacy_stops_vm_when_start_times_out(tmp_path, monkeypatch):
    from scripts.study_operator import i7_phase_b
    owner = StartTimesOut(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        i7_phase_b.dispatch(owner, 'privacy', {})
    assert owner.calls[-1] == 'stop', owner.calls


# --- I8-03 -----------------------------------------------------------------------------------

class PublicPage(Page):
    url = 'https://'+PUBLIC_IP+'.sslip.io/'

    def locator(self, selector):
        if selector == 'body':
            from tests.test_study_i7_attempt8 import Group
            # Forma real de aria_snapshot() en Chromium para un enlace (verificado con Chromium 1243).
            return Group(aria='- link "enlace":\n  - /url: https://'+PUBLIC_IP+'.sslip.io/x\n- button "Entrar"')
        return super().locator(selector)


@pytest.mark.xfail(strict=True, reason='I8-03: console.json guarda ws.url (wss://<IP>.sslip.io) y aria.yaml las URL; '
                                       'scrub() solo conoce token, IP/UA del canario y valores de campos')
def test_i8_03_failure_dump_never_writes_public_ip(tmp_path, monkeypatch):
    """Recorrido real de producción: run() con url https://<IP>.sslip.io falla y vuelca el diagnóstico."""
    import contextlib
    from scripts.study_operator import i7_browser
    stream = 'wss://'+PUBLIC_IP+'.sslip.io/_stcore/stream'

    class FailingPage(PublicPage):
        def goto(self, url, **kwargs):
            socket = types.SimpleNamespace(url=stream, on=lambda *a: None)
            for handler in self.handlers.get('websocket', []):
                handler(socket)  # Streamlit abre su WebSocket contra el mismo host
            for handler in self.handlers.get('console', []):
                handler(types.SimpleNamespace(type='error', text='WebSocket connection to '+stream+' failed'))
            raise RuntimeError('ORIGINAL_UI_FAILURE')
    page = FailingPage()

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
        i7_browser.run('https://'+PUBLIC_IP+'.sslip.io', TOKEN, {'config': {}}, identity,
                       lambda: {'identity': identity}, failure_dir=tmp_path/'failures')
    (folder,) = (tmp_path/'failures').iterdir()
    leaked = sorted(p.name for p in folder.iterdir() if PUBLIC_IP.encode() in p.read_bytes())
    assert leaked == []


def test_i8_03_characterization_token_and_canary_never_written(tmp_path):
    """Lo que sí cumple 3bb9ca8 (también verificado con Chromium real, ver docs/review/2026-11-02-i8)."""
    from scripts.study_operator.i7_browser import dump_failure
    canary_ip, ua = '198.51.100.7', 'Mozilla/5.0 review-i8'
    events = {'console': [{'type': 'log', 'text': 'x '+TOKEN+' '+canary_ip+' '+ua}], 'pageerror': [TOKEN],
              'websockets': [], 'websocket_closed': []}
    folder = dump_failure(Page(), tmp_path, (TOKEN, canary_ip, ua, ua+' CloudRAG-IP/'+canary_ip), events, TOKEN)
    for path in Path(folder).iterdir():
        data = path.read_bytes()
        assert TOKEN.encode() not in data and canary_ip.encode() not in data and ua.encode() not in data, path.name


# --- I8-04 -----------------------------------------------------------------------------------

class StopFails(Owner):
    def stop(self):
        self.calls.append('stop')
        if self.calls.count('stop') > 1:
            raise ValueError('STOP_NOT_VERIFIED')


@pytest.mark.xfail(strict=True, reason='I8-04: el stop() del except BaseException (i7_smoke.py:50) reemplaza la causa '
                                       'real; solo queda en __context__ (mismo patrón que B-2)')
def test_i8_04_smoke_failure_cause_survives_stop_failure(tmp_path, monkeypatch):
    def browse(*args, **kwargs):
        raise RuntimeError('UI_BROKEN_ENTRAR')
    smoke = smoke_env(monkeypatch, browse)
    with pytest.raises(Exception) as caught:
        smoke.run(StopFails(tmp_path))
    assert 'UI_BROKEN_ENTRAR' in str(caught.value)
