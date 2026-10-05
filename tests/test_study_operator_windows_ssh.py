import os
import subprocess

import pytest

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.policy import OperatorError
from scripts.study_operator.windows_ssh import sdk_argv, windows_argv


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows SDK/PuTTY stdin and native argv contract')


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
    command = subprocess.list2cmdline([str(executable), '-batch', 'host', 'fixed-command']).encode()
    calls = []

    def invoke(argv, **kwargs):
        calls.append((argv, kwargs))
        if '--command=true' in argv:
            assert kwargs['input'] is None and '--ssh-flag=-batch' in argv
            return subprocess.CompletedProcess(argv, 0, b'', b'')
        if '--dry-run' in argv:
            assert kwargs['input'] is None
            return subprocess.CompletedProcess(argv, 0, command, b'')
        assert argv[0] == str(executable) and kwargs['input'] == secret
        assert kwargs['env']['CLOUDSDK_SSH_PUTTY_FORCE_CONNECT'] == 'False'
        return subprocess.CompletedProcess(argv, 0, secret, b'')

    cloud = Cloud(str(sdk), 'pure-loop-474323-a8', tmp_path/'runs', invoke=invoke)
    assert cloud.command(['compute', 'ssh', 'fixture'], input_data=secret,
        private_output=True, json_output=False) == secret
    assert len(calls) == 3
    assert secret.decode() not in ''.join(path.read_text() for path in cloud.root.iterdir())


def test_changed_sdk_executable_never_runs_remote_child(tmp_path):
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        if '--command=true' in argv:
            return subprocess.CompletedProcess(argv, 0, b'', b'')
        return subprocess.CompletedProcess(argv, 0, b'other.exe -batch host', b'')

    cloud = Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke)
    with pytest.raises(OperatorError, match='clave del host'):
        cloud.command(['compute', 'ssh', 'fixture'], input_data=b'private', json_output=False)
    assert len(calls) == 2


def test_authentication_failure_never_sends_private_bytes(tmp_path):
    calls = []

    def invoke(argv, **kwargs):
        calls.append((argv, kwargs))
        assert '--command=true' in argv and kwargs['input'] is None
        return subprocess.CompletedProcess(argv, 1, b'', b'key rejected')

    cloud = Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke)
    with pytest.raises(OperatorError, match='clave del host'):
        cloud.command(['compute', 'ssh', 'fixture'], input_data=b'private', json_output=False)
    assert len(calls) == 1
