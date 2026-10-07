import json

import pytest

from scripts.study_operator.evidence import add, verify


def test_claim_receipts_idempotence_and_tamper(tmp_path):
    (tmp_path/'STATE.json').write_text(json.dumps(dict(agent='Fixture agent', model='Fixture model')))
    (tmp_path/'test.stdout').write_text('synthetic command output')
    (tmp_path/'test-receipt.json').write_text(json.dumps(dict(command=['test', 'exact'], exit_code=0)))
    spec = dict(key='test', statement='Synthetic component passed', certainty='VERIFICADO',
                evidence=['test.stdout'], command_receipts=['test-receipt.json'])
    rows = add(tmp_path, [spec])
    assert rows[0]['agent'] == 'Fixture agent' and rows[0]['model'] == 'Fixture model'
    assert add(tmp_path, [spec]) == rows
    assert verify(tmp_path)['claims'] == 1
    with pytest.raises(ValueError, match='relabeled'):
        add(tmp_path, [dict(spec, certainty='DECLARADO')])
    (tmp_path/'test.stdout').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        verify(tmp_path)


def test_mutable_state_and_unsupported_verification_rejected(tmp_path):
    (tmp_path/'STATE.json').write_text(json.dumps(dict(agent='Fixture agent', model='Fixture model')))
    (tmp_path/'test.stdout').write_text('synthetic')
    spec = dict(key='test', statement='component', certainty='VERIFICADO', evidence=['STATE.json'])
    with pytest.raises(ValueError, match='immutable'):
        add(tmp_path, [spec])
    with pytest.raises(ValueError, match='exact command'):
        add(tmp_path, [dict(spec, evidence=['test.stdout'])])
