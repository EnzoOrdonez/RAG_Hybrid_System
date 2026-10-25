import json

import pytest

from scripts.study_operator.audit_package import census
from scripts.study_operator.interactive_receipts import import_receipts


def fixture(tmp_path):
    (tmp_path/'STATE.json').write_text(json.dumps(dict(status='ACTIVE', agent='Codex', model='declared-model')))
    (tmp_path/'COMMANDS.log').write_text('')
    directory = tmp_path/'interactive-receipts'
    directory.mkdir()
    row = dict(agent='Codex', model='declared-model', phase=3, tool='exec_command', input=dict(cmd='Get-Date'),
               duration_s=1.25, exit_code=0, observed_after_utc='2026-10-07 19:42:51 UTC', start_timestamp_not_reconstructed=True)
    path = directory/'receipt.json'
    path.write_text(json.dumps(row))
    return path, row


def test_import_once_preserves_observation_without_inventing_interval(tmp_path):
    path, original = fixture(tmp_path)
    before = path.read_bytes()
    assert import_receipts(tmp_path)['imported'] == 1
    value = json.loads((tmp_path/'COMMANDS.log').read_text())
    assert value['duration_s'] == original['duration_s'] and 'started_utc' not in value and 'ended_utc' not in value
    assert census([('interactive', value)])['global_union']['lower_bound_seconds'] == 0
    log = (tmp_path/'COMMANDS.log').read_bytes()
    assert import_receipts(tmp_path)['imported'] == 0
    assert (tmp_path/'COMMANDS.log').read_bytes() == log and path.read_bytes() == before


def test_changed_previously_imported_receipt_is_rejected(tmp_path):
    path, row = fixture(tmp_path)
    import_receipts(tmp_path)
    row['duration_s'] = 12
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='changed'):
        import_receipts(tmp_path)


@pytest.mark.parametrize('field,value', [('duration_s', float('nan')), ('duration_s', True),
    ('phase', 7), ('agent', 'Other'), ('start_timestamp_not_reconstructed', False)])
def test_bad_receipt_does_not_partially_change_log(tmp_path, field, value):
    path, row = fixture(tmp_path)
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        import_receipts(tmp_path)
    assert (tmp_path/'COMMANDS.log').read_text() == ''


def test_sealed_package_is_read_only(tmp_path):
    fixture(tmp_path)
    (tmp_path/'MANIFEST_SHA256.jsonl').write_text('sealed')
    with pytest.raises(ValueError):
        import_receipts(tmp_path)
    assert (tmp_path/'COMMANDS.log').read_text() == ''
