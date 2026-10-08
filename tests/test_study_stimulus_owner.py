import json
from types import SimpleNamespace

import pytest

from scripts.study_operator.policy import OperatorError
from scripts.study_operator.stimulus_host_evidence import sha
from scripts.study_operator.stimulus_owner import run


def operator(tmp_path):
    boot = '12345678-aaaa-bbbb-cccc-123456789abc'
    proof = dict(inventory=dict(observed=dict(boot_id=boot)),rows=[dict(raw_text='private synthetic answer')],boot_index=5)
    calls = []
    def bridge(request,**kwargs):
        calls.append((request,kwargs))
        if request['operation'] == 'stimulus-evidence':
            return dict(proof=proof,proof_sha256=sha(proof))
        return dict(status='OWNED')
    value = SimpleNamespace(root=tmp_path,state=dict(purpose='technical'),
        observed=lambda: dict(status='RUNNING'),preflight=lambda: calls.append('preflight'),bridge=bridge)
    return value,proof,calls


def test_owner_download_is_hash_verified_before_ack_and_never_prints_content(tmp_path,capsys):
    value,proof,calls = operator(tmp_path)
    result = run(value,'stimulus-collect')
    assert json.loads(open(result['path'],encoding='utf-8').read()) == proof
    assert calls[-1][0] == dict(operation='stimulus-ack',proof_sha256=sha(proof))
    assert all(item[1]['private'] for item in calls)
    assert 'private synthetic answer' not in json.dumps(result) and capsys.readouterr().out == ''
    assert result['acceptance_not_inferred'] is True
    assert run(value,'stimulus-collect')['proof_sha256'] == result['proof_sha256']


def test_existing_different_evidence_is_not_overwritten_or_acknowledged(tmp_path):
    value,proof,calls = operator(tmp_path)
    root = tmp_path/'private'/'stimulus'/'coded-P999'
    root.mkdir(parents=True)
    path = root/(proof['inventory']['observed']['boot_id']+'.json')
    path.write_text('{}')
    with pytest.raises(OperatorError,match='distinta'):
        run(value,'stimulus-collect')
    assert path.read_text() == '{}' and len(calls) == 1


@pytest.mark.parametrize('defect',['purpose','stopped','hash','traversal'])
def test_owner_refuses_invalid_or_out_of_scope_collection(tmp_path,defect):
    value,proof,calls = operator(tmp_path)
    if defect == 'purpose':
        value.state['purpose'] = 'study'
    elif defect == 'stopped':
        value.observed = lambda: dict(status='TERMINATED')
    elif defect == 'hash':
        value.bridge = lambda *args,**kwargs: dict(proof=proof,proof_sha256='f'*64)
    else:
        proof['inventory']['observed']['boot_id'] = '../outside'
    with pytest.raises(OperatorError):
        run(value,'stimulus-collect')
    assert not (tmp_path/'private').exists()


def test_start_requires_preflight_and_sends_only_calendar_index(tmp_path):
    value,_,calls = operator(tmp_path)
    run(value,'stimulus-start',boot_index=5)
    assert calls == ['preflight',(dict(operation='stimulus-start',boot_index=5),dict(private=True))]


def test_cli_parses_only_fixed_stimulus_operations():
    from scripts.study_operator.cli import parser

    assert parser().parse_args(['stimulus-start','--boot-index','12']).boot_index == 12
    with pytest.raises(SystemExit):
        parser().parse_args(['stimulus-start','--boot-index','13'])
    assert parser().parse_args(['stimulus-collect']).operation == 'stimulus-collect'
