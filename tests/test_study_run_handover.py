import json

from scripts.study_operator.run_control import Recorder


def test_checkpoint_preserves_paid_state_and_describes_active_work(tmp_path):
    state = dict(status='ACTIVE', resources=[dict(type='vm', id='123')],
        cost=dict(estimated_spend_usd=12), open_exposures={'test': {'maximum_usd': 1}},
        independent_safety={'task': 'own-stop'}, deadline_utc='2026-10-10T02:00:00Z')
    (tmp_path/'STATE.json').write_text(json.dumps(state))
    (tmp_path/'own-active.json').write_text(json.dumps(dict(status='RUNNING', pid=123)))
    (tmp_path/'own-task-receipt.json').write_text(json.dumps(dict(task='CloudRAG-I5-own')))
    Recorder(tmp_path, tmp_path, agent='Fixture', model='Fixture', phase=1).checkpoint('Verify before replay')
    observed = json.loads((tmp_path/'STATE.json').read_bytes())
    assert observed['resources'] == state['resources'] and observed['cost'] == state['cost']
    handover = (tmp_path/'HANDOVER.md').read_text()
    payload = json.loads(handover.split('```json\n')[1].split('\n```')[0])
    assert payload['resources'] == state['resources'] and payload['open_exposures'] == state['open_exposures']
    assert payload['active_jobs'] == [dict(status='RUNNING', pid=123)]
    assert payload['independent_safety'] == state['independent_safety']
    assert payload['next_action'] == 'Verify before replay' and payload['tasks'] == ['CloudRAG-I5-own']
