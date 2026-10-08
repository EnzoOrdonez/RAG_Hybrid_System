import json
import hashlib
import subprocess

import pytest

from scripts.study_operator.read_only_http_probe import NATIVE, URL, probe, validate_paths


def sdk_fixture(tmp_path):
    sdk = tmp_path/'sdk'
    for relative in ('platform/bundledpython/python.exe', 'lib/googlecloudsdk/core/requests.py'):
        path = sdk/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    return sdk


def test_probe_is_read_only_and_credentials_only_reach_private_stdin(tmp_path):
    sdk = sdk_fixture(tmp_path)
    token = 'PRIVATE_FIXTURE_SENTINEL'
    def invoke(argv, **options):
        assert token not in str(argv)
        settings = json.loads(options['input'])
        assert settings['token'] == token and settings['url'] == URL
        assert 'setIamPolicy' not in NATIVE and '/instances' not in NATIVE
        assert options['timeout'] == 120
        assert options['env']['CLOUDSDK_CORE_DISABLE_FILE_LOGGING'] == '1'
        rows = [dict(repetition=r, padding_bytes=n, server_responded=True, http_status=400)
                for r in (1, 2) for n in (0, 106294)]
        return subprocess.CompletedProcess(argv, 0, json.dumps(dict(status='READ_ONLY_SDK_SIZE_PROBE', rows=rows,
            no_iam_or_compute_mutations=True, token_and_policy_not_persisted=True)).encode(), b'')
    result = probe(sdk, token, 106294, invoke=invoke)
    assert token not in json.dumps(result)
    assert not any(token in p.read_text() for p in sdk.rglob('*') if p.is_file())


@pytest.mark.parametrize('size', [-1, True, 99999, 200001])
def test_probe_rejects_out_of_scope_input_before_network(tmp_path, size):
    with pytest.raises(ValueError):
        probe(sdk_fixture(tmp_path), 'fixture', size, invoke=lambda *a, **kw: pytest.fail('network'))


def test_probe_private_failure_is_not_written_or_exposed(tmp_path):
    def invoke(argv, **options):
        return subprocess.CompletedProcess(argv, 1, b'', b'PRIVATE_TOKEN_PROXY_POLICY_SENTINEL')
    with pytest.raises(ValueError, match='^READ_ONLY_SDK_PROBE_EXECUTION_FAILED$'):
        probe(sdk_fixture(tmp_path), 'fixture', 106294, invoke=invoke)


def test_dry_import_does_not_require_credentials(tmp_path):
    def invoke(argv, **options):
        settings = json.loads(options['input'])
        assert settings['token'] == '' and settings['dry_run'] is True
        return subprocess.CompletedProcess(argv, 0, json.dumps(dict(status='SDK_TRANSPORT_IMPORT_PASS', no_network=True)).encode(), b'')
    assert probe(sdk_fixture(tmp_path), '', 106294, invoke=invoke, dry_run=True)['no_network'] is True


@pytest.mark.parametrize('defect', [None, 'foreign_output', 'existing_output', 'foreign_script', 'tampered_script', 'foreign_package'])
def test_paths_bound_to_owned_intent_and_exclusive_evidence(tmp_path, defect):
    package, owner = tmp_path/'package', tmp_path/'owner'
    package.mkdir()
    (owner/'runs'/'run').mkdir(parents=True)
    script = owner/'runs'/'run'/'primary-startup.sh'
    script.write_bytes(b'synthetic')
    (owner/'installation.json').write_text(json.dumps(dict(audit_run=str(package))))
    (owner/'active.json').write_text(json.dumps(dict(primary_creation_intent=dict(startup_sha256=hashlib.sha256(script.read_bytes()).hexdigest()))))
    output = package/'receipt.json'
    if defect == 'foreign_output':
        output = tmp_path/'foreign.json'
    elif defect == 'existing_output':
        output.write_text('previous evidence')
    elif defect == 'foreign_script':
        script = tmp_path/'primary-startup.sh'
        script.write_bytes(b'synthetic')
    elif defect == 'tampered_script':
        script.write_bytes(b'tampered')
    elif defect == 'foreign_package':
        package = tmp_path/'foreign_package'
    if defect:
        with pytest.raises(ValueError):
            validate_paths(package, owner, script, output)
    else:
        assert validate_paths(package, owner, script, output)[1:] == (script.resolve(), output.resolve())
