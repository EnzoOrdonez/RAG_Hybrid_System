from pathlib import Path
import json
import subprocess

import pytest

from scripts.study_operator.cloud_client import Cloud, checked_vm, no_other_gpu, readiness
from scripts.study_operator.policy import OperatorError, ReadyPending


def test_secret_output_and_private_download_never_written(tmp_path):
    sentinel = b'SYNTHETIC_PRIVATE_SENTINEL'
    cloud = Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path,
                  invoke=lambda *a, **k: subprocess.CompletedProcess([], 0, sentinel, b''))
    assert cloud.owner_token() == sentinel.decode()
    assert sentinel.decode() not in ''.join(p.read_text() for p in tmp_path.iterdir())
    assert not list(tmp_path.glob('*.stdout'))


def test_owner_compute_metadata_is_private_by_default(tmp_path):
    content = dict(metadata={'ssh-keys': 'PERSON_EMAIL_CANARY@example.invalid'},
                   unneeded='CLIENT_IP_AND_UA_CANARY')
    cloud = Cloud('fixture', 'pure-loop-474323-a8', tmp_path,
        invoke=lambda *a, **k: subprocess.CompletedProcess([], 0, json.dumps(content).encode(), b''))
    assert cloud.command(['compute', 'instances', 'list']) == content
    assert not list(tmp_path.glob('*.stdout'))
    assert 'CANARY' not in ''.join(p.read_text() for p in tmp_path.iterdir())


@pytest.mark.parametrize('private_suffix', [b'', b' identity PERSON_EMAIL_CANARY@example.invalid'])
def test_capacity_code_preserved_but_personal_footer_not_written(tmp_path, private_suffix):
    error = b'ZONE_RESOURCE_POOL_EXHAUSTED: zone us-central1-a lacks available resources'+private_suffix
    cloud = Cloud('fixture', 'pure-loop-474323-a8', tmp_path,
        invoke=lambda *a, **k: subprocess.CompletedProcess([], 1, b'', error))
    with pytest.raises(OperatorError, match='ZONE_RESOURCE_POOL_EXHAUSTED'):
        cloud.command(['compute', 'instances', 'start', 'fixture', '--zone=us-central1-a'])
    receipt = json.loads((tmp_path/'0001-receipt.json').read_bytes())
    assert receipt['cloud_error_code'] == 'ZONE_RESOURCE_POOL_EXHAUSTED'
    assert ('capacity_error_text' in receipt) == (not private_suffix)
    assert 'PERSON_EMAIL_CANARY' not in json.dumps(receipt)


def test_timeout_and_malformed_json_return_actionable_errors(tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('synthetic tool', 1)

    cloud = Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path / 'timeout', invoke=timeout)
    with pytest.raises(OperatorError, match='verifica status'):
        cloud.command(['compute', 'instances', 'list'])
    cloud = Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path / 'json',
                  invoke=lambda *a, **k: subprocess.CompletedProcess([], 0, b'not JSON', b''))
    with pytest.raises(OperatorError, match='contrato JSON'):
        cloud.command(['compute', 'instances', 'list'])


def test_native_protection_or_other_gpu_fails_before_start():
    vm = dict(name='fixture', id='1', zone='zones/us-central1-a', machineType='machineTypes/g2-standard-4',
              deletionProtection=True, disks=[{'autoDelete': False}],
              scheduling={'instanceTerminationAction': 'STOP', 'maxRunDuration': {'seconds': '10800'}})
    assert checked_vm(vm, name='fixture', instance_id='1', zone='us-central1-a') == vm
    with pytest.raises(OperatorError, match='protecciones'):
        checked_vm(dict(vm, deletionProtection=False), name='fixture', instance_id='1', zone='us-central1-a')
    with pytest.raises(OperatorError, match='otra VM'):
        no_other_gpu([dict(id='2', status='RUNNING', guestAccelerators=[{}])], selected_id='1')
    assert no_other_gpu([dict(id='2', status='TERMINATED', guestAccelerators=[{}])], selected_id='1') is None


def test_missing_or_invalid_ready_has_message_instead_of_stopiteration():
    for value in (None, {}, {'status': 'STARTING'}):
        with pytest.raises(OperatorError, match='no repitas start'):
            readiness(value, image_id='fixture-image', url='https://fixture.invalid', boot_id='fixture-boot')
    with pytest.raises(OperatorError, match='admisión'):
        readiness({'status': 'READY'}, image_id='fixture-image', url='https://fixture.invalid', boot_id='fixture-boot')


def test_every_resource_command_keeps_project_json_and_no_file_logs(tmp_path):
    observed = []

    def invoke(argv, **kwargs):
        observed.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b'[]', b'')

    cloud = Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke)
    assert cloud.command(['compute', 'instances', 'list']) == []
    argv, options = observed[0]
    assert '--project=pure-loop-474323-a8' in argv and '--format=json' in argv
    assert options['env']['CLOUDSDK_CORE_DISABLE_FILE_LOGGING'] == '1'
    assert options['env']['CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED'] == 'False'
    assert len(list(Path(tmp_path).glob('*-receipt.json'))) == 1


def test_new_client_preserves_prior_command_receipts(tmp_path):
    def invoke(*args, **kwargs):
        return subprocess.CompletedProcess([], 0, b'[]', b'')

    Cloud('fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke).command(['compute', 'instances', 'list'])
    original = (tmp_path / '0001-receipt.json').read_bytes()
    Cloud('fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke).command(['compute', 'instances', 'list'])
    assert (tmp_path / '0001-receipt.json').read_bytes() == original
    assert (tmp_path / '0002-receipt.json').exists()


def test_boot_guest_keys_404_is_pending_but_other_404_stays_failure(tmp_path):
    error = b"HTTPError 404: The resource 'hostkeys/' of type 'Guest Attribute' was not found."
    cloud = Cloud('fixture', 'pure-loop-474323-a8', tmp_path,
        invoke=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, b'', error))
    with pytest.raises(ReadyPending, match='claves'):
        cloud.command(['compute', 'instances', 'get-guest-attributes', 'fixture',
            '--zone=us-central1-c', '--query-path=hostkeys/'])
    with pytest.raises(OperatorError, match='rechaz'):
        cloud.command(['compute', 'instances', 'describe', 'fixture', '--zone=us-central1-c'])


@pytest.mark.parametrize('private', [False, True])
def test_timeout_partial_output_is_preserved_only_for_public_commands(tmp_path, private):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('fixture', 1, output=b'PUBLIC_OR_PRIVATE_SENTINEL',
                                        stderr=b'TRANSPORT_FAILURE_DETAIL')

    cloud = Cloud('fixture', 'pure-loop-474323-a8', tmp_path, invoke=timeout)
    with pytest.raises(OperatorError, match='verifica status'):
        cloud.command(['compute', 'ssh', 'fixture'], private_output=private, json_output=False)
    receipt = json.loads((tmp_path/'0001-receipt.json').read_bytes())
    assert receipt['exit_code'] == 124
    assert receipt['partial_output_policy'] == ('PRIVATE_NOT_PERSISTED' if private else 'PUBLIC_PRESERVED')
    if private:
        assert not list(tmp_path.glob('*.stdout')) and not list(tmp_path.glob('*.stderr'))
        assert 'PUBLIC_OR_PRIVATE_SENTINEL' not in ''.join(p.read_text() for p in tmp_path.iterdir())
    else:
        assert (tmp_path/'0001.stdout').read_bytes() == b'PUBLIC_OR_PRIVATE_SENTINEL'
        assert (tmp_path/'0001.stderr').read_bytes() == b'TRANSPORT_FAILURE_DETAIL'
