"""i7_browser.run (unchanged) against src/ui/app.py over Streamlit TLS on localhost; synthetic store."""
import json, os, socket, subprocess, sys, tempfile, time, ssl, urllib.request
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from harness import make_store, REPO, CHROME
from datetime import datetime, timedelta, timezone
import playwright._impl._browser_type as bt
orig = bt.BrowserType.launch
async def launch(self, **kw):
    kw['executablePath'] = CHROME
    return await orig(self, **kw)
bt.BrowserType.launch = launch
from scripts.study_operator import i7_browser
i7_browser.runtime_identity = lambda pw: {'version': 'container-1243'}
root = Path(tempfile.mkdtemp(prefix='f020x-'))
config, assignments, store, token = make_store(root)
if os.environ.get('F020_EXPIRE')=='1':
    from src.ui.components.session_storage import atomic_json
    d=store._read()
    for r in d['invitations'].values(): r['expires_at']=r['issued_at']-1
    atomic_json(store.path,d)
subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=localhost',
                '-keyout',str(root/'c.key'),'-out',str(root/'c.crt')],check=True,capture_output=True)
with socket.socket() as s:
    s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
env = dict(os.environ, PYTHONPATH=str(REPO), CLOUDRAG_STUDY_CONFIG=str(config), CLOUDRAG_STUDY_ASSIGNMENTS=str(assignments),
           CLOUDRAG_STUDY_SESSION_DIR=str(root/'sessions'), CLOUDRAG_STUDY_PURPOSE='smoke', CLOUDRAG_MODE='participant',
           CLOUDRAG_SMOKE_PARTICIPANT='P999', CLOUDRAG_BACKUP_BUCKET='synthetic-bucket', CLOUDRAG_BACKUP_PREFIX='synthetic',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
log = (root/'server.log').open('wb')
proc = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'src/ui/app.py', '--server.address', '127.0.0.1',
    '--server.port', str(port), '--server.headless', 'true', '--server.fileWatcherType', 'none',
    '--browser.gatherUsageStats', 'false', '--browser.serverAddress', 'localhost', '--browser.serverPort', str(port),
    '--server.sslCertFile', str(root/'c.crt'), '--server.sslKeyFile', str(root/'c.key')], cwd=REPO, env=env, stdout=log, stderr=log)
for _ in range(300):
    try:
        urllib.request.urlopen(f'https://localhost:{port}/_stcore/health', context=ssl._create_unverified_context(), timeout=1); break
    except OSError:
        time.sleep(.1)
identity = {'boot_id': 'local', 'image_id': 'sha256:'+'a'*64, 'source_commit': 'b'*40}
stages = []
def observe():
    paths = list((root/'sessions').glob('*/study_checkpoint.json'))
    data = json.loads(paths[0].read_bytes()) if paths else None
    if data and (not stages or stages[-1] != data['stage']): stages.append(data['stage'])
    return {'identity': identity, 'data': data}
from src.ui.components.study_protocol import load_protocol
try:
    i7_browser.run(f'https://localhost:{port}', token, load_protocol(config, assignments), identity, observe,
                   local_test=True, participant='P999', failure_dir=root/'failures')
    print('RUN_OK')
except Exception as e:
    print('RUN_RAISED', type(e).__name__, str(e).replace(token, '<token>')[:200])
finally:
    proc.terminate(); proc.wait(timeout=15)
print('stages_seen', stages)
import json as _j
for d in sorted((root/'failures').glob('*')) if (root/'failures').exists() else []:
    files = {p.name: p.read_bytes() for p in d.iterdir()}
    print('dump', sorted(files), 'token_in_dump', any(token.encode() in v for v in files.values()))
    print(json.loads(files['controls.json'])['buttons'] if 'controls.json' in files else None)
    print('diagnostics', files.get('diagnostics.json', b'').decode()[:400]); print('ARIA', [l for l in files['aria.yaml'].decode().splitlines() if 'sesi' in l or 'Entrar' in l])
