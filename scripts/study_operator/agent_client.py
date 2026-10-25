"""Private Unix backup transport; the application never receives a cloud token."""
import hashlib
import http.client
import json
import re
import socket


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost', timeout=120)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def backup_request(socket_path, session_dir, bucket, prefix):
    session_id = session_dir.name
    if not re.fullmatch('[a-f0-9]{32}', session_id):
        raise ValueError('Invalid managed session id')
    content = (session_dir / 'full_session.json').read_bytes()
    manifest = (session_dir / 'export_manifest.json').read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    body = dict(session_id=session_id, bucket=bucket, prefix=prefix, sha256=digest)
    connection = UnixConnection(socket_path)
    try:
        connection.request('POST', '/backup', json.dumps(body).encode(), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError('Mandatory host backup not verified')
        result = json.loads(response.read(32769))
    finally:
        connection.close()
    base = 'gs://' + bucket + '/' + prefix + '/'
    destination = result.get('destination', '')
    relative = destination.removeprefix(base)
    if (result.get('status') != 'complete' or result.get('sha256') != digest
            or not destination.startswith(base) or not re.fullmatch('P[0-9]{2,6}/' + session_id, relative)
            or set(result.get('objects', {})) != {'full_session.json', 'export_manifest.json'}):
        raise ValueError('Host backup receipt differs')
    for name, data in [('full_session.json', content), ('export_manifest.json', manifest)]:
        row = result['objects'][name]
        if (row.get('object') != destination.removeprefix('gs://' + bucket + '/') + '/' + name
                or not re.fullmatch('[1-9][0-9]*', str(row.get('generation', '')))
                or row.get('sha256') != hashlib.sha256(data).hexdigest()):
            raise ValueError('Generation-bound backup receipt differs')
    return result
