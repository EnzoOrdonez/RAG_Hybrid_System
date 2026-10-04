"""Host-only storage agent. Fixed directories and buckets, never a credential proxy."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import re
import socketserver
import urllib.request

from scripts.study_operator.gcs import Storage
from scripts.study_operator.policy import participant_code


def checked_file(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
        raise ValueError('Invalid managed path')
    return path


def backup(settings, request, storage):
    sid = request.get('session_id')
    if not isinstance(sid, str) or not re.fullmatch('[a-f0-9]{32}', sid):
        raise ValueError('Invalid session id')
    if request.get('bucket') != settings['sessions_bucket'] or request.get('prefix') != settings['prefix']:
        raise ValueError('Backup destination differs from host policy')
    root = Path(settings['sessions_root'])
    export = checked_file(root, sid + '/full_session.json')
    data = export.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if request.get('sha256') != digest:
        raise ValueError('Export changed')
    payload = json.loads(data)
    code = participant_code(payload.get('participant_id', payload.get('assignment', {}).get('participant_id')))
    if payload['session_id'] != sid or payload['purpose'] != settings['purpose'] or payload['stage'] not in ('complete', 'abandoned'):
        raise ValueError('Export not eligible')
    if settings['purpose'] != 'study' and int(code[1:]) < 900:
        raise ValueError('Synthetic purpose requires synthetic participant code')
    manifest_path = checked_file(root, sid + '/export_manifest.json')
    manifest = manifest_path.read_bytes()
    if json.loads(manifest)['files'] != {'full_session.json': digest}:
        raise ValueError('Export manifest mismatch')
    prefix = settings['prefix'] + '/' + code + '/' + sid
    receipts = {name: storage.put(prefix + '/' + name, content)
                for name, content in [('full_session.json', data), ('export_manifest.json', manifest)]}
    private_inventory = Path(settings['private_inventory_root'])
    private_inventory.mkdir(parents=True, exist_ok=True)
    inventory = dict(session_id=sid, participant_id=code, purpose=settings['purpose'],
                     paths=[str(checked_file(root, sid + '/' + name)) for name in
                            ['study_checkpoint.json', 'full_session.json', 'export_manifest.json', 'backup_state.json']],
                     objects=receipts)
    target = private_inventory / (sid + '.json')
    temporary = private_inventory / (sid + '.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(inventory, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    if os.name == 'posix':
        fd = os.open(private_inventory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return dict(status='complete', source=str(root / sid), destination='gs://' + settings['sessions_bucket'] + '/' + prefix,
                sha256=digest, objects=receipts)


def access_token():
    request = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',
                                     headers={'Metadata-Flavor': 'Google'})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)['access_token']


def serve(settings):
    storage = Storage(settings['sessions_bucket'], access_token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if self.path != '/backup' or not 0 < size <= 4096:
                    raise ValueError('Invalid operation')
                result = backup(settings, json.loads(self.rfile.read(size)), storage)
                status = 200
            except Exception:
                result = {'status': 'pending', 'error': 'BACKUP_NOT_VERIFIED'}
                status = 503
            content = json.dumps(result).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            pass

    with Server(settings['socket'], Handler) as server:
        os.chmod(settings['socket'], 0o660)
        server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', required=True)
    arguments = parser.parse_args()
    serve(json.loads(Path(arguments.settings).read_text(encoding='utf-8')))
