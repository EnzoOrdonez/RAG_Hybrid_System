import base64
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.policy import OperatorError, ReadyPending
from scripts.study_operator.windows_ssh import api_host_key_flags, native_argv, sdk_argv, windows_argv


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows SDK/PuTTY stdin and native argv contract')


@pytest.fixture(autouse=True)
def owned_client_files(tmp_path, monkeypatch):
    import scripts.study_operator.windows_ssh as module
    ssh, key = tmp_path/'ssh.exe', tmp_path/'synthetic-key'
    ssh.write_bytes(b'')
    key.write_bytes(b'SYNTHETIC_NOT_A_PRIVATE_KEY')
    original = module.native_argv
    monkeypatch.setattr(module, 'native_argv', lambda *args, **kwargs:
                        original(*args, **kwargs, ssh=ssh, key=key))
    return ssh, key


def api_keys():
    wire = (11).to_bytes(4, 'big') + b'ssh-ed25519' + (32).to_bytes(4, 'big') + b'\x01'*32
    return [{'namespace':'hostkeys','key':'ssh-ed25519',
             'value':base64.b64encode(wire).decode()}]


def rpc_args():
    return ['compute', 'ssh', 'cloudrag-fixture', '--zone=us-central1-b']


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
    proxy = 'gcloud compute start-iap-tunnel cloudrag-fixture 22 --zone=us-central1-b --project=pure-loop-474323-a8'
    command = subprocess.list2cmdline([str(executable), '-T', '-proxycmd', proxy,
        '-batch', '-hostkey', fingerprint, 'user@compute.123', 'sudo', '-n', 'python3', '-B', '-']).encode()
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
        if argv[1:4] == ['compute', 'instances', 'describe']:
            assert kwargs['input'] is None
            return subprocess.CompletedProcess(argv, 0, json.dumps(dict(
                name='cloudrag-fixture', id='123', zone='zones/us-central1-b')).encode(), b'')
        assert Path(argv[0]).name == 'ssh.exe' and kwargs['input'] == secret
        assert 'StrictHostKeyChecking=yes' in argv and 'UpdateHostKeys=no' in argv
        assert kwargs['env']['CLOUDSDK_SSH_PUTTY_FORCE_CONNECT'] == 'False'
        return subprocess.CompletedProcess(argv, 0, secret, b'')

    cloud = Cloud(str(sdk), 'pure-loop-474323-a8', tmp_path/'runs', invoke=invoke)
    assert cloud.command(rpc_args(), input_data=secret,
        private_output=True, json_output=False) == secret
    assert len(calls) == 5
    assert json.loads((cloud.root/'0001-receipt.json').read_bytes())['transport'] == 'WINDOWS_OPENSSH_IAP_PINNED'
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


def test_cli_json_contract_mismatch_is_readable_and_never_reaches_ssh(tmp_path):
    calls=[]
    def invoke(argv, **options):
        calls.append(argv)
        assert 'get-guest-attributes' in argv and options['input'] is None
        return subprocess.CompletedProcess(argv,0,b'{"unexpected":"REST_WRAPPER"}',b'')
    cloud=Cloud('gcloud-fixture','pure-loop-474323-a8',tmp_path,invoke=invoke)
    with pytest.raises(OperatorError,match='clave del host'):
        cloud.command(rpc_args(),input_data=b'PRIVATE_NEVER_SENT',json_output=False)
    assert len(calls)==1


def test_boot_keys_not_published_yet_waits_without_private_ssh(tmp_path):
    calls = []
    def invoke(argv, **options):
        calls.append(argv)
        assert 'get-guest-attributes' in argv and options['input'] is None
        return subprocess.CompletedProcess(argv, 1, b'',
            b"HTTPError 404: The resource 'hostkeys/' of type 'Guest Attribute' was not found.")
    cloud = Cloud('fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke)
    with pytest.raises(ReadyPending, match='claves'):
        cloud.command(rpc_args(), input_data=b'PRIVATE_NEVER_SENT', json_output=False)
    assert len(calls) == 1
    assert json.loads((tmp_path/'0001-receipt.json').read_bytes())['reason'] == 'PUBLIC_HOST_KEYS_PENDING'


def native_fixture(tmp_path):
    sdk = tmp_path/'SDK path/bin/gcloud.cmd'
    pins = api_host_key_flags(api_keys())[1].removeprefix('--ssh-flag=')
    proxy = 'gcloud start-iap-tunnel cloudrag-fixture 22 --zone=us-central1-b --project=pure-loop-474323-a8'
    argv = [str(sdk.parent/'sdk/plink.exe'), '-T', '-proxycmd', proxy, '-batch',
            '-hostkey', pins, 'user@compute.123', 'sudo', '-n', 'python3', '-B', '-']
    target = dict(name='cloudrag-fixture', id='123', zone='us-central1-b')
    return sdk, argv, target


@pytest.mark.parametrize('change', ['id', 'zone', 'key', 'command', 'alias', 'executable'])
def test_native_target_and_pins_fail_closed(tmp_path, owned_client_files, change):
    sdk, argv, target = native_fixture(tmp_path)
    rows = api_keys()
    if change == 'id':
        target['id'] = '124'
    elif change == 'zone':
        target['zone'] = 'europe-west1-b'
    elif change == 'key':
        rows[0]['value'] = base64.b64encode(b'wrong').decode()
    elif change == 'command':
        argv[-5] = 'other'
    elif change == 'alias':
        argv[argv.index('user@compute.123')] = 'user@compute.124'
    else:
        argv[0] = 'other.exe'
    with pytest.raises(ValueError):
        native_argv(argv, rows, sdk=sdk, target=target, known_hosts=tmp_path/'pins',
                    ssh=owned_client_files[0], key=owned_client_files[1])
    assert not (tmp_path/'pins').exists()


def test_native_pins_are_public_and_config_is_isolated(tmp_path, owned_client_files):
    sdk, argv, target = native_fixture(tmp_path)
    actual = native_argv(argv, api_keys(), sdk=sdk, target=target, known_hosts=tmp_path/'pins',
                         ssh=owned_client_files[0], key=owned_client_files[1])
    assert actual[1:4] == ['-F', 'NUL', '-T']
    assert actual[-6:] == ['user@compute.123', 'sudo', '-n', 'python3', '-B', '-']
    assert 'StrictHostKeyChecking=yes' in actual and 'GlobalKnownHostsFile=NUL' in actual
    assert 'PasswordAuthentication=no' in actual and 'CheckHostIP=no' in actual
    assert (tmp_path/'pins').read_text().startswith('compute.123 ssh-ed25519 ')
    with pytest.raises(FileExistsError):
        native_argv(argv, api_keys(), sdk=sdk, target=target, known_hosts=tmp_path/'pins',
                    ssh=owned_client_files[0], key=owned_client_files[1])


def test_native_timeout_kills_only_owned_child_tree_and_preserves_private_ram(monkeypatch):
    import scripts.study_operator.windows_ssh as module
    sentinel = b'PRIVATE_NATIVE_TIMEOUT_SENTINEL'
    killed = []

    class Child:
        pid = 1234
        calls = 0

        def communicate(self, **options):
            self.calls += 1
            if self.calls == 1:
                assert options['input'] == sentinel
                raise subprocess.TimeoutExpired('ssh', 1)
            return sentinel, b''

    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: Child())
    monkeypatch.setattr(module.subprocess, 'run', lambda argv, **options: killed.append(argv))
    with pytest.raises(subprocess.TimeoutExpired) as error:
        module.native_run(['ssh.exe'], input=sentinel, capture_output=True, timeout=1, env={})
    assert error.value.stdout == sentinel
    assert killed == [['taskkill', '/PID', '1234', '/T', '/F']]
