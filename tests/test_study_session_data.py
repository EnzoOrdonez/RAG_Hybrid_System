import json

import pytest

from scripts.study_operator.policy import OperatorError
from scripts.study_operator.session_data import clean, inventory, export_by_code


def fixture(tmp_path):
    root = tmp_path / 'sessions'
    root.mkdir()
    (root / '_i4_root.json').write_text(json.dumps(dict(schema_version=1, purpose='smoke')))
    rows = {}
    for index, code in enumerate(('P999', 'P998'), 1):
        sid = str(index) * 32
        folder = root / sid
        folder.mkdir()
        payload = dict(assignment={'participant_id': code}, session_id=sid, purpose='smoke')
        (folder / 'study_checkpoint.json').write_text(json.dumps(payload))
        (folder / 'full_session.json').write_text(json.dumps(payload))
        rows[str(index)] = dict(participant_id=code, session_id=sid)
    (root / '_admissions.json').write_text(json.dumps(dict(invitations=rows, active='1' * 32)))
    return root


def test_withdraw_preserves_other_code_and_requires_every_local_download(tmp_path):
    root = fixture(tmp_path)
    plan = inventory(root, code='P999')
    before = (root / ('2' * 32) / 'study_checkpoint.json').read_bytes()
    with pytest.raises(OperatorError, match='Faltan'):
        clean(plan, [])
    assert (root / ('1' * 32) / 'study_checkpoint.json').exists()
    result = clean(plan, plan['files'])
    assert result['empty']
    assert (root / ('2' * 32) / 'study_checkpoint.json').read_bytes() == before
    assert json.loads((root / '_admissions.json').read_text())['invitations'] == {'2': plan['invitations']['1'] | {'participant_id': 'P998', 'session_id': '2' * 32}}


def test_purge_removes_preparation_pending_and_private_inventory_copies(tmp_path):
    root = fixture(tmp_path)
    (root / ('1' * 32) / '.pending-synthetic').write_text('synthetic incomplete write')
    prep = root / '_preparation'
    prep.mkdir()
    (prep / 'event.json').write_text(json.dumps({'scope': '1' * 32, 'kind': 'synthetic'}))
    private = tmp_path / 'private'
    private.mkdir()
    (private / ('1' * 32 + '.json')).write_text('synthetic path inventory')
    plan = inventory(root, private_inventory=private)
    assert len(plan['files']) == 7
    assert clean(plan, plan['files'])['empty']
    assert not inventory(root, private_inventory=private)['session_codes']
    assert not list(private.iterdir()) and not list(prep.iterdir())


def test_unknown_files_changed_inventory_and_study_root_fail_closed(tmp_path):
    root = fixture(tmp_path)
    plan = inventory(root, code='P999')
    (root / ('1' * 32) / 'unregistered.log').write_text('unregistered data')
    with pytest.raises(OperatorError, match='desconocido'):
        clean(plan, plan['files'])
    (root / '_i4_root.json').write_text(json.dumps(dict(schema_version=1, purpose='study')))
    with pytest.raises(OperatorError, match='solo autoriza'):
        inventory(root)


def test_export_excludes_direct_identifiers_and_keeps_free_text_in_private_review():
    session = dict(assignment={'participant_id': 'P01'}, purpose='study', session_id='SESSION_SENTINEL',
        created_at='TIMESTAMP_SENTINEL', ip='IP_SENTINEL', attempts=[dict(condition='hybrid', analysis_role='free_query',
        question='PERSONAL_SENTINEL', answer='PRIVATE_SENTINEL', decline_class='answered', elapsed_ms=123)],
        instruments=[], comparative={'C1': 'A', 'C2': 'B', 'C3': 'A', 'C4': 'OPEN_SENTINEL'},
        blinding={'choice': 'A', 'reason': 'REASON_SENTINEL'})
    result = export_by_code([session])
    assert 'SENTINEL' not in json.dumps(result['pseudonymous_by_code'])
    assert 'SENTINEL' not in json.dumps(result['publication_candidate'])
    assert 'PERSONAL_SENTINEL' in json.dumps(result['private_manual_review'])
    assert result['publication_candidate']['aggregate'][0]['response_class'] == 'answered'
    assert result['automatic_publication_allowed'] is False


def test_cleanup_resume_after_checkpoint_and_export_already_unlinked(tmp_path, monkeypatch):
    from scripts.study_operator import session_data

    root = fixture(tmp_path)
    plan = inventory(root)
    atomic = session_data.atomic_json

    def interrupted(path, value):
        if path.name.startswith('_cleanup-') and len(value['removed']) == 2:
            raise ConnectionError('synthetic crash after unlink, before progress receipt')
        return atomic(path, value)

    monkeypatch.setattr(session_data, 'atomic_json', interrupted)
    with pytest.raises(ConnectionError):
        clean(plan, plan['files'])
    assert not list((root / ('1' * 32)).iterdir())
    monkeypatch.setattr(session_data, 'atomic_json', atomic)
    assert clean(plan, plan['files'])['empty']
    assert not inventory(root)['session_codes']
    assert clean(plan, plan['files'])['replayed']
