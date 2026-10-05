import ast
import base64
from pathlib import Path
import urllib.request

import pytest

from scripts.study_operator.startup import host_key_publication, write_startup


def public_key(algorithm='ssh-ed25519'):
    wire = len(algorithm).to_bytes(4, 'big') + algorithm.encode() + (32).to_bytes(4, 'big') + b'\x01'*32
    return algorithm+' '+base64.b64encode(wire).decode()+' synthetic-public-key\n'


def test_startup_publishes_only_valid_public_keys_to_instance_metadata(tmp_path, monkeypatch):
    (tmp_path/'ssh_host_ed25519_key.pub').write_text(public_key())
    # A private file is a sentinel: it must never be read or sent.
    private=b'SYNTHETIC_PRIVATE_FILE_MUST_NOT_BE_READ'
    (tmp_path/'ssh_host_ed25519_key').write_bytes(private)
    requests=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def read(self):return b''
    def urlopen(request, timeout):
        requests.append(request)
        assert timeout==5
        return Response()
    monkeypatch.setattr(urllib.request,'urlopen',urlopen)
    source=host_key_publication(str(tmp_path))
    ast.parse(source)
    exec(compile(source,'fixture-publication','exec'),{'Path':Path})
    assert len(requests)==1
    request=requests[0]
    assert request.full_url=='http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/hostkeys/ssh-ed25519'
    assert request.get_method()=='PUT' and request.get_header('Metadata-flavor')=='Google'
    assert request.data==public_key().split()[1].encode()
    assert private not in request.data


@pytest.mark.parametrize('content',['ssh-rsa '+public_key().split()[1], 'ssh-ed25519 invalid-base64', ''])
def test_invalid_public_file_cannot_publish_a_trust_pin(tmp_path, monkeypatch, content):
    (tmp_path/'ssh_host_ed25519_key.pub').write_text(content)
    requests=[]
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:requests.append(a))
    with pytest.raises(ValueError):
        exec(compile(host_key_publication(str(tmp_path)),'fixture-publication','exec'),{'Path':Path})
    assert requests==[]


def test_missing_public_keys_fails_startup_before_host_launch(tmp_path):
    with pytest.raises(ValueError,match='NO_PUBLIC_HOST_KEYS'):
        exec(compile(host_key_publication(str(tmp_path)),'fixture-publication','exec'),{'Path':Path})


def test_publication_runs_before_launch_config_and_host_start(tmp_path):
    script=write_startup(tmp_path/'startup.sh',{'fixture':True}).read_text()
    python=script.split("python3 - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    ast.parse(python)
    assert script.index('guest-attributes/hostkeys/')<script.index('p=root/"launch-config.json"')<script.index('subprocess.Popen')
