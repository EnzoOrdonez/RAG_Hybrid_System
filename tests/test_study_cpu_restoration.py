import hashlib
import json

import pytest

from scripts.study_operator.cpu_restoration import Controller, create_arguments, startup_text, verified_closure


def test_no_paid_effect_before_verified_inherited_closure(tmp_path):
    calls = []

    class Cloud:
        def command(self, args, **kwargs):
            calls.append(args)

    with pytest.raises(FileNotFoundError):
        Controller(tmp_path, Cloud()).prepare({}, {})
    assert not calls


def test_cpu_create_has_no_gpu_public_ip_or_credentials_and_native_stop():
    argv = create_arguments('owned', 'cloned', 'script', 'marker', dict(network='study', subnet='study-central'))
    assert '--machine-type=e2-standard-2' in argv
    assert '--no-address' in argv and '--no-service-account' in argv and '--no-scopes' in argv
    assert '--max-run-duration=2h' in argv and '--instance-termination-action=STOP' in argv
    assert '--disk=name=cloned,boot=yes,auto-delete=no' in argv and '--deletion-protection' in argv
    assert not any('accelerator' in arg for arg in argv)
    text = startup_text()
    assert 'hostkeys/' in text and "['docker','stop',*ids]" in text
    assert 'shutdown' in text and '+110' in text
    assert 'host_runtime' not in text and 'serve' not in text


def test_creation_intents_are_idempotent_but_cannot_adopt_foreign_resource(tmp_path):
    (tmp_path/'STATE.json').write_text(json.dumps(dict(resources=[])))
    controller = Controller(tmp_path, None)
    controller.intent('vm', 'owned', 'marker')
    controller.intent('vm', 'owned', 'marker')
    assert len(json.loads((tmp_path/'STATE.json').read_bytes())['resource_intents']) == 1
    with pytest.raises(ValueError, match='differs'):
        controller.intent('vm', 'owned', 'different')
    with pytest.raises(ValueError, match='ownership'):
        controller.observed_resource('vm', dict(name='owned', id='1', description='foreign'), 'marker')


def test_seal_requires_complete_external_census_and_pinned_manifest(tmp_path):
    old = tmp_path/'old'
    old.mkdir()
    manifest = old/'MANIFEST_SHA256.jsonl'
    manifest.write_bytes(b'fixture inventory')
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    proof = dict(status='PASS', errors=[], missing=[], added=[], links=[], checked_files=1, expected_files=1,
                 manifest_sha256=digest, audited_directory_modified=False)
    external = tmp_path/'proof.json'
    external.write_text(json.dumps(proof))
    receipt = dict(status='SEALED_AND_EXTERNALLY_VERIFIED', verifier_exit_code=0,
                   verification_receipt=str(external), root=str(old))
    (tmp_path/'iteration4-recovered-seal02.json').write_text(json.dumps(receipt))
    (tmp_path/'legacy-seal-retry-pin.json').write_text(json.dumps(dict(files=1, manifest_sha256=digest)))
    assert verified_closure(tmp_path) == receipt
    proof['checked_files'] = 0
    external.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='externally'):
        verified_closure(tmp_path)
    proof['checked_files'] = 1
    external.write_text(json.dumps(proof))
    manifest.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        verified_closure(tmp_path)
