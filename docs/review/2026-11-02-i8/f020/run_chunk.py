"""3bb9ca8 i7_browser.run (unchanged) over TLS with the submit-button chunk aborted or hung; shows what the dump records."""
import os, subprocess, sys, tempfile, time, ssl, urllib.request, socket, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H
from chunk import chunk_of
mode = sys.argv[1]
import playwright._impl._browser_type as bt
orig = bt.BrowserType.launch
async def launch(self, **kw):
    kw['executablePath'] = H.CHROME
    return await orig(self, **kw)
bt.BrowserType.launch = launch
from playwright.sync_api._generated import Browser
orig_ctx = Browser.new_context
target = chunk_of('stFormSubmitButton')
def new_context(self, **kw):
    ctx = orig_ctx(self, **kw)
    ctx.route('**/' + target, (lambda r: r.abort()) if mode == 'abort' else (lambda r: None))
    return ctx
Browser.new_context = new_context
from scripts.study_operator import i7_browser
i7_browser.runtime_identity = lambda pw: {'version': 'container-1243'}
i7_browser.BUTTON_WAIT_S = 10            # 90 s in 3bb9ca8; shortened only to keep the run short
root = Path(tempfile.mkdtemp(prefix='f020r-'))
config, assignments, store, token = H.make_store(root)
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=localhost',
                '-keyout',str(root/'c.key'),'-out',str(root/'c.crt')],check=True,capture_output=True)
with socket.socket() as s:
    s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
env = dict(os.environ, PYTHONPATH=str(H.REPO), CLOUDRAG_STUDY_CONFIG=str(config), CLOUDRAG_STUDY_ASSIGNMENTS=str(assignments),
           CLOUDRAG_STUDY_SESSION_DIR=str(root/'sessions'), CLOUDRAG_STUDY_PURPOSE='smoke', CLOUDRAG_MODE='participant',
           CLOUDRAG_SMOKE_PARTICIPANT='P999', CLOUDRAG_BACKUP_BUCKET='synthetic-bucket', CLOUDRAG_BACKUP_PREFIX='synthetic',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
log = (root/'server.log').open('wb')
proc = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'src/ui/app.py', '--server.address', '127.0.0.1',
    '--server.port', str(port), '--server.headless', 'true', '--server.fileWatcherType', 'none',
    '--browser.gatherUsageStats', 'false', '--browser.serverAddress', 'localhost', '--browser.serverPort', str(port),
    '--server.sslCertFile', str(root/'c.crt'), '--server.sslKeyFile', str(root/'c.key')], cwd=H.REPO, env=env, stdout=log, stderr=log)
for _ in range(300):
    try:
        urllib.request.urlopen(f'https://localhost:{port}/_stcore/health', context=ssl._create_unverified_context(), timeout=1); break
    except OSError:
        time.sleep(.1)
identity = {'boot_id': 'local', 'image_id': 'sha256:'+'a'*64, 'source_commit': 'b'*40}
def observe():
    paths = list((root/'sessions').glob('*/study_checkpoint.json'))
    return {'identity': identity, 'data': json.loads(paths[0].read_bytes()) if paths else None}
from src.ui.components.study_protocol import load_protocol
try:
    i7_browser.run(f'https://localhost:{port}', token, load_protocol(config, assignments), identity, observe,
                   local_test=True, participant='P999', failure_dir=root/'failures')
    print('RUN_OK')
except Exception as e:
    print('RUN_RAISED', type(e).__name__, str(e).replace(token, '<token>')[:160])
finally:
    proc.terminate(); proc.wait(timeout=15)
(folder,) = (root/'failures').iterdir()
files = {p.name: p.read_bytes() for p in folder.iterdir()}
print('dump', sorted(files), 'token_in_dump', any(token.encode() in v for v in files.values()))
aria = files['aria.yaml'].decode()
print('ARIA_SIGNATURE', [l.strip() for l in aria.splitlines() if 'alert' in l or 'Missing Submit' in l or 'dynamically imported' in l or 'Entrar' in l or 'Invitación' in l])
console = json.loads(files['console.json'])
print('CONSOLE_ERRORS', [c['text'][:110] for c in console['console'] if c['type'] == 'error'][:3], 'sync_error', console['sync_error'])
print('BUTTONS', json.loads(files['controls.json'])['buttons'])
