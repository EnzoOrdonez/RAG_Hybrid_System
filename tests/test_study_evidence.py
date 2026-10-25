import json

import pytest

from scripts.study_operator.evidence import add, retract, verify


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


def test_correction_preserves_original_claim_and_marks_it_unusable(tmp_path):
    (tmp_path/'STATE.json').write_text(json.dumps(dict(agent='Fixture agent',model='Fixture model')))
    (tmp_path/'facts.json').write_text('synthetic observed facts')
    spec = dict(key='old',statement='Wrong interpretation',certainty='DECLARADO',evidence=['facts.json'])
    rows = add(tmp_path,[spec,dict(spec,key='new',statement='Corrected interpretation')])
    original = (tmp_path/'claims.json').read_bytes()
    result = retract(tmp_path,rows[0]['id'],rows[1]['id'],'Read the preserved census correctly')
    assert (tmp_path/'claims.json').read_bytes() == original
    assert retract(tmp_path,rows[0]['id'],rows[1]['id'],result['reason']) == result
    checked = verify(tmp_path)
    assert checked['claims'] == 2 and checked['current_claims'] == 1
    assert checked['retracted_claims'] == [rows[0]['id']]
    assert 'RETRACTADA: no usar como resultado' in (tmp_path/'CLAIMS_LEDGER.md').read_text(encoding='utf-8')
    add(tmp_path,[dict(spec,key='third',statement='Another observation')])
    assert 'RETRACTADA' in (tmp_path/'CLAIMS_LEDGER.md').read_text(encoding='utf-8')
    with pytest.raises(ValueError,match='cannot be rewritten'):
        retract(tmp_path,rows[0]['id'],rows[1]['id'],'Silently replace the explanation')


@pytest.mark.parametrize('old,new',[('I5-V999','I5-V002'),('I5-V002','I5-V001'),('I5-V001','I5-V001')])
def test_invalid_correction_never_changes_claims_or_ledger(tmp_path,old,new):
    (tmp_path/'STATE.json').write_text(json.dumps(dict(agent='Fixture agent',model='Fixture model')))
    (tmp_path/'facts.json').write_text('synthetic')
    spec = dict(key='one',statement='One',certainty='DECLARADO',evidence=['facts.json'])
    add(tmp_path,[spec,dict(spec,key='two',statement='Two')])
    ledger = (tmp_path/'CLAIMS_LEDGER.md').read_bytes()
    with pytest.raises(ValueError,match='later existing'):
        retract(tmp_path,old,new,'Wrong reference')
    assert not (tmp_path/'claim_corrections.json').exists()
    assert (tmp_path/'CLAIMS_LEDGER.md').read_bytes() == ledger
