"""Real HTTPS participant UI, synthetic backend only."""
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.study_helpers import configured


@contextmanager
def local_app(tmp_path, *, canary_probe=False):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    from src.ui.components.study_protocol import load_protocol
    from src.ui.components.study_sessions import StudyStore
    config,assignments,_=configured(tmp_path)
    protocol=load_protocol(config,assignments)
    store=StudyStore(tmp_path/'sessions',protocol,purpose='smoke')
    store.freeze()
    token=store.issue('P999',cell=1,profile='without_experience')
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    name=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,'localhost')])
    certificate=(x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(datetime.now(timezone.utc)-timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc)+timedelta(days=1)).sign(key,hashes.SHA256()))
    cert,pem=tmp_path/'local.crt',tmp_path/'local.key'
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    pem.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port=sock.getsockname()[1]
    env=dict(os.environ,CLOUDRAG_TEST_ROOT=str(tmp_path),CLOUDRAG_TEST_CANARY_TRACE=str(int(canary_probe)),
             PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    log=(tmp_path/'server.log').open('wb')
    process=subprocess.Popen([sys.executable,'-m','streamlit','run',str(Path(__file__).with_name('study_browser_fixture.py')),
        '--server.headless=true','--server.address=127.0.0.1','--server.port='+str(port),
        '--server.sslCertFile='+str(cert),'--server.sslKeyFile='+str(pem),
        '--browser.gatherUsageStats=false'],env=env,stdout=log,stderr=log)
    identity={'boot_id':'synthetic-local-boot','image_id':'sha256:'+'a'*64,'source_commit':'b'*40}
    def observe():
        paths=list(store.root.glob('*/study_checkpoint.json'))
        result={'identity':identity,'data':json.loads(paths[0].read_bytes()) if paths else None}
        if paths and (paths[0].parent/'full_session.json').exists() and (paths[0].parent/'backup_state.json').exists():
            full=(paths[0].parent/'full_session.json').read_bytes()
            result.update(export_base64=base64.b64encode(full).decode(),export_sha256=hashlib.sha256(full).hexdigest(),
                backup=json.loads((paths[0].parent/'backup_state.json').read_bytes()))
        return result
    try:
        import ssl
        import urllib.request
        for _ in range(100):
            try:
                with urllib.request.urlopen(f'https://localhost:{port}/_stcore/health',context=ssl._create_unverified_context(),timeout=1):
                    break
            except (OSError,ValueError):
                if process.poll() is not None:
                    raise RuntimeError('LOCAL_STREAMLIT_FAILED')
                time.sleep(.1)
        else:
            raise TimeoutError('LOCAL_STREAMLIT_START')
        yield f'https://localhost:{port}',token,protocol,identity,observe
    finally:
        process.terminate()
        process.wait(timeout=15)
        log.close()


def test_real_https_ui_full_export(tmp_path):
    from scripts.study_operator.i7_browser import run
    with local_app(tmp_path) as (url,token,protocol,identity,observe):
        result=run(url,token,protocol,identity,observe,local_test=True)
    assert result['status']=='BROWSER_EXPORT_VERIFIED'
    assert result['attempts']==8 and result['instruments']==2
    assert result['websocket_seen'] and len(result['confirmed_inputs_sha256'])>5
    target=Path(os.environ.get('CLOUDRAG_BROWSER_RECEIPT_DIR',str(tmp_path)))
    target.mkdir(parents=True,exist_ok=True)
    (target/'browser-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


@pytest.mark.parametrize('fault',['unregistered','incomplete','identity'])
def test_real_https_ui_rejects_backend_fault(tmp_path,fault):
    from scripts.study_operator.i7_browser import run
    with local_app(tmp_path) as (url,token,protocol,identity,observe):
        triggered=[]
        def corrupted():
            result=observe()
            if fault=='identity':
                triggered.append(fault)
                result['identity']=dict(identity,boot_id='other')
            if fault=='unregistered' and result['data'] and result['data']['attempts']:
                triggered.append(fault)
                result['data']['attempts'][-1]['question']='not registered'
            if fault=='incomplete' and 'export_base64' in result:
                triggered.append(fault)
                data=json.loads(base64.b64decode(result['export_base64']))
                data['attempts'].pop()
                raw=json.dumps(data).encode()
                result.update(export_base64=base64.b64encode(raw).decode(),export_sha256=hashlib.sha256(raw).hexdigest())
            return result
        expected={'unregistered':'APP_VALUE_NOT_REGISTERED',
                  'incomplete':'BROWSER_EXPORT_INCOMPLETE_OR_CHANGED',
                  'identity':'BROWSER_BOOT_IDENTITY_DIFFERS'}[fault]
        with pytest.raises(ValueError,match=expected):
            run(url,token,protocol,identity,corrupted,local_test=True)
        assert triggered
