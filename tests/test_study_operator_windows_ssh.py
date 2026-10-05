import base64
import json
import os
import subprocess

import pytest

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.policy import OperatorError
from scripts.study_operator.windows_ssh import api_host_key_flags, sdk_argv, windows_argv


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows SDK/PuTTY stdin and native argv contract')


def api_keys():
    wire = (11).to_bytes(4, 'big') + b'ssh-ed25519' + (32).to_bytes(4, 'big') + b'\x01'*32
    return {'queryValue': {'items': [{'namespace':'hostkeys','key':'ssh-ed25519',
                                     'value':base64.b64encode(wire).decode()}]}}


def rpc_args():
    return ['compute', 'ssh', 'fixture', '--zone=us-central1-b']


def test_sdk_nested_iap_display_recovers_exact_arguments(tmp_path):
    executable = tmp_path/'SDK path/plink.exe'
    expected = [str(executable), '-T', '-i', 'C:/owned/key.ppk', '-proxycmd',
        '"C:/SDK path/python.exe" "-S" "C:/SDK path/gcloud.py" compute start-iap-tunnel vm %port --listen-on-stdin',
        '-batch', 'user@compute.123', 'sudo', '-n', 'python3', '-B', '-']
    display = ' '.join('"'+arg+'"' if ' ' in arg else arg for arg in expected)
    assert windows_argv(display) != expected
    assert sdk_argv(display.encode(), executable) == expected


@pytest.mark.parametrize('text', ['', 'one\ntwo', 'one\rtwo'])
def test_invalid_sdk_display_fails_closed(text):
    with pytest.raises(ValueError):
        sdk_argv(text.encode(), 'C:/fixture/plink.exe')


def test_rpc_private_stdin_and_output_bypass_sdk_mediation(tmp_path):
    secret = b'SYNTHETIC_PRIVATE_RPC_SENTINEL'
    executable = tmp_path/'SDK path/sdk/plink.exe'
    sdk = executable.parent.parent/'gcloud.cmd'
    fingerprint = api_host_key_flags(api_keys())[1].removeprefix('--ssh-flag=')
    command = subprocess.list2cmdline([str(executable), '-batch', '-hostkey', fingerprint, 'host', 'fixed-command']).encode()
    calls = []

    def invoke(argv, **kwargs):
        calls.append((argv, kwargs))
        if 'get-guest-attributes' in argv:
            assert kwargs['input'] is None
            return subprocess.CompletedProcess(argv, 0, json.dumps(api_keys()).encode(), b'')
        if '--command=true' in argv:
            assert kwargs['input'] is None and '--ssh-flag=-batch' in argv
            assert '--ssh-flag=-hostkey' in argv and '--ssh-flag='+fingerprint in argv
            return subprocess.CompletedProcess(argv, 0, b'', b'')
        if '--dry-run' in argv:
            assert kwargs['input'] is None
            return subprocess.CompletedProcess(argv, 0, command, b'')
        assert argv[0] == str(executable) and kwargs['input'] == secret
        assert kwargs['env']['CLOUDSDK_SSH_PUTTY_FORCE_CONNECT'] == 'False'
        return subprocess.CompletedProcess(argv, 0, secret, b'')

    cloud = Cloud(str(sdk), 'pure-loop-474323-a8', tmp_path/'runs', invoke=invoke)
    assert cloud.command(rpc_args(), input_data=secret,
        private_output=True, json_output=False) == secret
    assert len(calls) == 4
    assert secret.decode() not in ''.join(path.read_text() for path in cloud.root.iterdir())


def test_changed_sdk_executable_never_runs_remote_child(tmp_path):
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        if 'get-guest-attributes' in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(api_keys()).encode(), b'')
        if '--command=true' in argv:
            return subprocess.CompletedProcess(argv, 0, b'', b'')
        return subprocess.CompletedProcess(argv, 0, b'other.exe -batch host', b'')

    cloud = Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke)
    with pytest.raises(OperatorError, match='clave del host'):
        cloud.command(rpc_args(), input_data=b'private', json_output=False)
    assert len(calls) == 3


def test_authentication_failure_never_sends_private_bytes(tmp_path):
    calls = []

    def invoke(argv, **kwargs):
        calls.append((argv, kwargs))
        if 'get-guest-attributes' in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(api_keys()).encode(), b'')
        assert '--command=true' in argv and kwargs['input'] is None
        return subprocess.CompletedProcess(argv, 1, b'', b'key rejected')

    cloud = Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke)
    with pytest.raises(OperatorError, match='clave del host'):
        cloud.command(rpc_args(), input_data=b'private', json_output=False)
    assert len(calls) == 2


def test_missing_or_discarded_public_host_key_never_sends_private_bytes(tmp_path):
    executable = tmp_path/'sdk/plink.exe'
    sdk = executable.parent.parent/'gcloud.cmd'
    calls=[]
    def invoke(argv, **options):
        calls.append(argv)
        assert options['input'] is None
        if 'get-guest-attributes' in argv:
            return subprocess.CompletedProcess(argv,0,json.dumps(api_keys()).encode(),b'')
        if '--command=true' in argv:
            return subprocess.CompletedProcess(argv,0,b'',b'')
        return subprocess.CompletedProcess(argv,0,subprocess.list2cmdline([str(executable),'-batch','host']).encode(),b'')
    cloud=Cloud(str(sdk),'pure-loop-474323-a8',tmp_path/'runs',invoke=invoke)
    with pytest.raises(OperatorError,match='clave del host'):
        cloud.command(rpc_args(),input_data=b'PRIVATE_NEVER_SENT',json_output=False)
    assert len(calls)==3
