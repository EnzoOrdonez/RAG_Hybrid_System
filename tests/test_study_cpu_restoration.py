import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_operator.cpu_restoration import Controller, create_arguments, record_stop, startup_text, verified_closure


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


def test_retry_never_overwrites_previous_stop_receipt(tmp_path):
    first = record_stop(tmp_path, 'owned', dict(id='1', attempt='failed-first-trial'))
    original = first.read_bytes()
    second = record_stop(tmp_path, 'owned', dict(id='1', attempt='diagnostic-second-trial'))
    assert first != second and first.read_bytes() == original


def test_stopped_owned_clone_is_resumed_once_without_an_extra_disk(monkeypatch, tmp_path):
    from scripts.study_operator import cpu_restoration

    monkeypatch.setattr(cpu_restoration, 'verified_closure', lambda root: True)
    monkeypatch.setattr(cpu_restoration, 'quote_archive', lambda *args:
        dict(usd_per_hour='.067', catalog_receipt_sha256='a'*64))
    state = dict(status='ACTIVE', resources=[], independent_closure_verified=True, cloud_cutoff_usd=90,
        closure_reserved_utc=(datetime.now(timezone.utc)+timedelta(hours=3)).isoformat(),
        cost=dict(estimated_spend_usd=12, reserved_retention_and_closure_usd=4))
    (tmp_path/'STATE.json').write_text(json.dumps(state))
    name = 'cloudrag-i5-restore-bootstrap01'
    marker = 'CloudRAG-I5-restore-'+tmp_path.name+'-bootstrap01'
    disk = dict(name=name+'-boot', id='disk', description=marker, creationTimestamp='2026-10-07T00:00:00Z',
        sourceSnapshotId='snapshot', selfLink='disks/owned')
    vm = dict(name=name, id='vm', description=marker, creationTimestamp='2026-10-07T00:00:00Z',
        machineType='machines/e2-standard-2', status='TERMINATED', deletionProtection=True,
        disks=[dict(autoDelete=False, source=disk['selfLink'])], networkInterfaces=[{}],
        scheduling=dict(instanceTerminationAction='STOP', maxRunDuration={'seconds': '7200'}))
    rule = dict(name=name+'-iap', id='rule', description=marker, creationTimestamp='2026-10-07T00:00:00Z',
        sourceRanges=['35.235.240.0/20'], targetTags=[name+'-iap'], allowed=[dict(IPProtocol='tcp', ports=['22'])])
    calls = []

    class Cloud:
        def command(self, args, **kwargs):
            calls.append(args)
            assert 'create' not in args
            if args[1] == 'snapshots':
                return dict(id='snapshot', status='READY')
            if args[1] == 'disks':
                return [disk] if args[2] == 'list' else disk
            if args[1] == 'firewall-rules':
                return [rule] if args[2] == 'list' else rule
            if args[2] == 'start':
                vm['status'] = 'RUNNING'
                return None
            return [dict(vm)] if args[2] == 'list' else dict(vm)

    controller = Controller(tmp_path, Cloud())
    inventory = dict(resources={'snapshots': [dict(name='cloudrag-i4-user-recovery-20261006', id='snapshot')]})
    assert controller.prepare(inventory, dict(network='study', subnet='study'))['vm']['status'] == 'RUNNING'
    controller.prepare(inventory, dict(network='study', subnet='study'))
    assert len([call for call in calls if call[:3] == ['compute', 'instances', 'start']]) == 1
    saved = json.loads((tmp_path/'STATE.json').read_bytes())
    assert len(saved['open_exposures']) == 2
