"""F-020: reproducción local sin GCP. Almacén temporal sintético; src/ sin tocar. Uso: harness.py OUTDIR [variantes...]

Servidores (env SERVERS): app = src/ui/app.py con los flags de cloud_entrypoint.streamlit_command;
fixture = tests/study_browser_fixture.py; app_relay = app detrás de dos network_relay (como app_runtime + Caddy, sin TLS).
Variantes: a_fill_click (= i7_browser.run), b_fill_blur_wait_click, c_enter, d_type_delay_click,
e_expired y f_conflict (rechazo de admit() en el servidor)."""
import hashlib, json, os, secrets, socket, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
CHROME = os.environ.get('F020_CHROME', '/opt/pw-browsers/chromium-1243/chrome-linux64/chrome')

def make_store(root):
    from tests.study_helpers import configured
    from src.ui.components.study_sessions import StudyStore
    config, assignments, protocol = configured(root)
    store = StudyStore(root/'sessions', protocol, purpose='smoke')
    store.freeze()
    token = secrets.token_urlsafe(32)                 # lifecycle.invite
    store.register_invitation_hash(hashlib.sha256(token.encode()).hexdigest(), 'P999',
                                   cell=1, profile='without_experience')   # cloud_entrypoint invite
    return config, assignments, store, token

def serve(root, config, assignments, server):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    env = dict(os.environ, PYTHONPATH=str(REPO), CLOUDRAG_TEST_ROOT=str(root),
               CLOUDRAG_STUDY_CONFIG=str(config), CLOUDRAG_STUDY_ASSIGNMENTS=str(assignments),
               CLOUDRAG_STUDY_SESSION_DIR=str(root/'sessions'), CLOUDRAG_STUDY_PURPOSE='smoke',
               CLOUDRAG_MODE='participant', CLOUDRAG_SMOKE_PARTICIPANT='P999', CLOUDRAG_DEMO_GPU='0',
               CLOUDRAG_BACKUP_BUCKET='synthetic-bucket', CLOUDRAG_BACKUP_PREFIX='synthetic',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    script = {'app': 'src/ui/app.py', 'fixture': 'tests/study_browser_fixture.py', 'app_relay': 'src/ui/app.py'}[server]
    # Same flags as scripts/cloud_entrypoint.py:streamlit_command (port changed; serverAddress -> localhost)
    argv = [sys.executable, '-m', 'streamlit', 'run', script, '--server.address', '127.0.0.1',
            '--server.port', str(port), '--server.headless', 'true', '--server.fileWatcherType', 'none',
            '--browser.gatherUsageStats', 'false', '--browser.serverAddress', 'localhost',
            '--browser.serverPort', str(port)]
    log = (root/'server.log').open('wb')
    proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=log)
    for _ in range(300):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=1); break
        except OSError:
            if proc.poll() is not None: raise RuntimeError('server died')
            time.sleep(.1)
    if server != 'app_relay':
        return proc, f'http://localhost:{port}'
    # Production chain: proxy -> unix /web/streamlit.sock -> relay (app_runtime.relays) -> 127.0.0.1:8501
    import threading
    from scripts.study_operator.network_relay import server as relay
    sock = str(root/'streamlit.sock')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); front = s.getsockname()[1]
    inner = relay(sock, ('127.0.0.1', port))            # app_runtime.py:17
    outer = relay(('127.0.0.1', front), sock)           # stands in for Caddy's reverse_proxy (no TLS)
    for r in (inner, outer):
        threading.Thread(target=r.serve_forever, daemon=True).start()
    return proc, f'http://localhost:{front}'

def attempt(variant, server, out):
    from playwright.sync_api import sync_playwright
    from streamlit.proto.BackMsg_pb2 import BackMsg
    from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
    from scripts.study_operator.i7_browser import StreamlitSync, visible
    root = Path(tempfile.mkdtemp(prefix='f020-'))
    config, assignments, store, token = make_store(root)
    if variant == 'e_expired':                 # server-side rejection: invitation past expires_at
        from src.ui.components.session_storage import atomic_json
        data = store._read()
        for row in data['invitations'].values():
            row['expires_at'] = row['issued_at'] - 1
        atomic_json(store.path, data)
    if variant == 'f_conflict':                # server-side rejection: another active session
        from src.ui.components.session_storage import atomic_json
        data = store._read(); data['active'] = 'f'*32; atomic_json(store.path, data)
    proc, url = serve(root, config, assignments, server)
    rec = {'variant': variant, 'server': server, 'frames': [], 'error': None}
    t0 = time.monotonic()
    def sent(payload):
        if not isinstance(payload, bytes): return
        m = BackMsg.FromString(payload); kind = m.WhichOneof('type'); row = {'t': round(time.monotonic()-t0, 3), 'dir': 'sent', 'type': kind}
        if kind == 'rerun_script':
            ws = m.rerun_script.widget_states.widgets
            row['widgets'] = [{'id_suffix': w.id[-24:], 'kind': w.WhichOneof('value'),
                               'value_is_token': (w.string_value == token) if w.WhichOneof('value') == 'string_value' else None,
                               'trigger': w.trigger_value if w.WhichOneof('value') == 'trigger_value' else None}
                              for w in ws]
        rec['frames'].append(row)
    def received(payload):
        if not isinstance(payload, bytes): return
        m = ForwardMsg.FromString(payload); kind = m.WhichOneof('type')
        if kind in ('new_session', 'script_finished', 'session_status_changed', 'session_event'):
            row = {'t': round(time.monotonic()-t0, 3), 'dir': 'recv', 'type': kind}
            if kind == 'script_finished': row['status'] = m.script_finished
            rec['frames'].append(row)
    try:
        with sync_playwright() as pw:
            b = pw.chromium.launch(executable_path=CHROME, headless=True)
            page = b.new_context().new_page()
            def ws_open(ws):
                if '/_stcore/stream' in ws.url:
                    rec['frames'].append({'t': round(time.monotonic()-t0, 3), 'dir': 'ws', 'type': 'open'})
                    ws.on('framesent', sent); ws.on('framereceived', received)
                    ws.on('close', lambda *_: rec['frames'].append({'t': round(time.monotonic()-t0, 3), 'dir': 'ws', 'type': 'close'}))
            page.on('websocket', ws_open)
            sync = StreamlitSync(page, 25)
            page.set_default_timeout(20000)
            page.goto(url, wait_until='domcontentloaded')
            sync.settled()
            field = page.get_by_label('Invitación', exact=True)
            if variant == 'a_fill_click':              # exactly i7_browser.run
                field.fill(token)
                sync.action(lambda: visible(page, sync, 'Entrar').click())
            elif variant == 'b_fill_blur_wait_click':
                field.fill(token); field.blur(); page.wait_for_timeout(1500)
                sync.action(lambda: visible(page, sync, 'Entrar').click())
            elif variant == 'c_enter':
                field.fill(token)
                sync.action(lambda: field.press('Enter'))
            elif variant in ('e_expired', 'f_conflict'):
                field.fill(token)
                sync.action(lambda: visible(page, sync, 'Entrar').click())
            elif variant == 'd_type_delay_click':
                field.press_sequentially(token, delay=30)
                sync.action(lambda: visible(page, sync, 'Entrar').click())
            page.wait_for_timeout(1500)
            rec['requests'], rec['completed'] = sync.requests, sync.completed
            rec['buttons_after'] = [x.inner_text() for x in page.get_by_role('button').all() if x.is_visible()]
            rec['error_text'] = page.locator('[data-testid=stAlert]').all_inner_texts()
            b.close()
    except Exception as e:
        rec['error'] = type(e).__name__+': '+str(e).replace(token, '<token>')[:300]
    finally:
        proc.terminate(); proc.wait(timeout=15)
    checkpoints = list((root/'sessions').glob('*/study_checkpoint.json'))
    rec['sessions_created'] = len(checkpoints)
    rec['stage'] = json.loads(checkpoints[0].read_bytes())['stage'] if checkpoints else None
    rec['login'] = bool(checkpoints)
    log = (root/'server.log').read_text(errors='replace').replace(token, '<token>')
    rec['server_log_tail'] = [l for l in log.splitlines() if l.strip()][-6:]
    blob = json.dumps(rec)
    assert token not in blob
    (Path(out)/f'{server}-{variant}.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    return rec

if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    servers = os.environ.get('SERVERS', 'app,fixture').split(',')
    variants = sys.argv[2:] or ['a_fill_click', 'b_fill_blur_wait_click', 'c_enter', 'd_type_delay_click']
    for server in servers:
        for v in variants:
            r = attempt(v, server, out)
            reruns = [f for f in r['frames'] if f.get('type') == 'rerun_script']
            print(server, v, 'login=', r['login'], 'sessions=', r['sessions_created'], 'stage=', r['stage'],
                  'reruns_sent=', len(reruns), 'completed=', r.get('completed'), 'err=', r['error'],
                  'buttons=', r.get('buttons_after'), 'alerts=', r.get('error_text'), flush=True)
            for f in reruns: print('   rerun', f['t'], f.get('widgets'))
