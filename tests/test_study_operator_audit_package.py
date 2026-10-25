import json

import pytest

from scripts.study_operator.audit_package import census, diagnose_existing, inventory, seal, verify_existing
from scripts.study_operator.run_control import validate_command


def test_clock_anomaly_keeps_original_and_never_invents_placement():
    bad = dict(started_utc='2026-10-05T06:33:46.718820+00:00',
               ended_utc='2026-10-05T06:33:46.273535+00:00', duration_s=.9759943997487426, phase=4)
    good = dict(start='2026-10-05T06:34:00+00:00', end='2026-10-05T06:34:10+00:00', phase=4)
    result = census([('bad', bad), ('valid', good), ('overlap', good)])
    assert result['phases']['4']['lower_bound_seconds'] == 10
    assert result['global_union']['lower_bound_seconds'] == 10
    assert result['clock_anomalies'][0]['original'] is bad
    assert result['clock_anomalies'][0]['duration_not_locatable'] == bad['duration_s']
    assert not result['reconstructed_timestamps']


def test_naive_and_missing_timestamps_are_not_valid_measurement_time():
    result = census([('naive', dict(start='2026-10-05T00:00:00', end='2026-10-05T00:00:01')),
                     ('missing', {})])
    assert not result['rows'] and len(result['clock_anomalies']) == 2


def test_complete_inventory_includes_empty_unicode_and_nested_files(tmp_path):
    root = tmp_path / 'package'
    root.mkdir()
    (root / 'vacío.txt').write_text('', encoding='utf-8')
    (root / 'nested').mkdir()
    (root / 'nested' / 'evidence.json').write_bytes(b'{"status":"FAIL"}')
    staging = tmp_path / 'pending'
    result = inventory(root, staging)
    rows = [json.loads(line) for line in staging.read_text(encoding='utf-8').splitlines()]
    assert result['files'] == 2 and result['bytes'] == 17
    assert {r['path'] for r in rows} == {'vacío.txt', 'nested/evidence.json'}
    assert all(len(r['sha256']) == 64 for r in rows)
    assert not (root / 'MANIFEST_SHA256.jsonl').exists()


def test_existing_manifest_and_inside_staging_refused(tmp_path):
    with pytest.raises(ValueError):
        inventory(tmp_path, tmp_path / 'inside')
    (tmp_path / 'MANIFEST_SHA256.jsonl').write_text('')
    with pytest.raises(ValueError):
        inventory(tmp_path, tmp_path.parent / 'outside')


def test_junction_reparse_guard_without_admin(monkeypatch, tmp_path):
    (tmp_path / 'evidence').write_text('kept')
    monkeypatch.setattr('scripts.study_operator.audit_package.linked', lambda p: p.name == 'evidence')
    with pytest.raises(ValueError, match='reparse'):
        inventory(tmp_path, tmp_path.parent / 'pending')


@pytest.mark.parametrize('args', [['python', 'invite', 'P01'], ['python', '--token=secret']])
def test_private_token_display_never_captured(args):
    with pytest.raises(ValueError, match='invitation'):
        validate_command(dict(name='private', argv=args))


def test_runner_requires_bounded_command_and_safe_label():
    with pytest.raises(ValueError):
        validate_command(dict(name='../old', argv=['python']))
    with pytest.raises(ValueError):
        validate_command(dict(name='new', argv=['python'], timeout=0))
    assert validate_command(dict(name='finite', argv=['python'], timeout=30)) == ['python']


def prepared_retry(tmp_path, source):
    root = tmp_path/'old'
    root.mkdir()
    (root/'data').write_text('unchanged')
    staging = tmp_path/'staging'
    result = inventory(root, staging)
    staging.rename(root/'MANIFEST_SHA256.jsonl')
    verifier = tmp_path/'external.py'
    verifier.write_text(source)
    failure = tmp_path/'failed.stderr'
    failure.write_text("'--expected-manifest-sha256', '"+result['manifest_sha256']+"'")
    pin = tmp_path/'pin.json'
    diagnose_existing(root, failure, verifier, pin)
    return root, verifier, pin


def test_read_only_retry_uses_pin_and_unchanged_external_verifier(tmp_path):
    root, verifier, pin = prepared_retry(tmp_path,
        "import sys,json\nfrom pathlib import Path\nPath(sys.argv[sys.argv.index('--output')+1]).write_text(json.dumps({'status':'PASS'}))\n")
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = verify_existing(root, tmp_path/'receipt.json', verifier, pin, seconds=10)
    assert result['status'] == 'SEALED_AND_EXTERNALLY_VERIFIED'
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}
    verifier.write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        verify_existing(root, tmp_path/'receipt02.json', verifier, pin)


def test_retry_timeout_gets_immutable_failure_receipt_without_root_writes(tmp_path):
    root, verifier, pin = prepared_retry(tmp_path, 'import time\ntime.sleep(10)\n')
    result = verify_existing(root, tmp_path/'receipt.json', verifier, pin, seconds=.1)
    assert result['status'] == 'TIMEOUT_PRESERVED'
    assert json.loads((tmp_path/'receipt.json').read_bytes())['audited_directory_modified'] is False
    with pytest.raises(ValueError, match='New external'):
        verify_existing(root, tmp_path/'receipt.json', verifier, pin, seconds=.1)


def test_existing_inventory_cannot_be_repinned_from_live_content(tmp_path):
    root, verifier, pin = prepared_retry(tmp_path, '')
    (root/'MANIFEST_SHA256.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        verify_existing(root, tmp_path/'receipt.json', verifier, pin)


def test_seal_and_readonly_retry_use_actual_standalone_verifier_cli(tmp_path):
    from pathlib import Path
    from scripts.study_operator import package_verify
    root = tmp_path/'own'
    root.mkdir()
    (root/'preserved-failure.txt').write_text('FAIL is evidence', encoding='utf-8')
    verifier = tmp_path/'external-verifier.py'
    verifier.write_bytes(Path(package_verify.__file__).read_bytes())
    receipt = tmp_path/'external-seal.json'
    result = seal(root, receipt, verifier, seconds=10)
    assert result['status'] == 'SEALED_AND_EXTERNALLY_VERIFIED'
    assert json.loads(receipt.with_suffix('.verification.json').read_bytes())['checked_files'] == 1
    pin = tmp_path/'pin.json'
    pin.write_text(json.dumps(dict(root=str(root.resolve()),
        manifest_sha256=result['manifest_sha256'], verifier_source_sha256=result['verifier_source_sha256'])))
    before = {p.name:p.read_bytes() for p in root.iterdir()}
    assert verify_existing(root, tmp_path/'retry.json', verifier, pin, seconds=10)['status'] == 'SEALED_AND_EXTERNALLY_VERIFIED'
    assert {p.name:p.read_bytes() for p in root.iterdir()} == before
