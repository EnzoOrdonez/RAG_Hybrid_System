from datetime import datetime, timezone
import json

import pytest

from scripts.study_operator.cloud_safety import admission, close, targets


def state():
    return dict(status='ACTIVE', closure_reserved_utc='2026-10-09T23:01:39Z', cloud_cutoff_usd=90,
                cost=dict(estimated_spend_usd=12, reserved_retention_and_closure_usd=4),
                independent_closure_verified=True,
                resources=[dict(type='vm', id='123', name='original', zone='us-central1-a')])


def test_exposure_cutoff_and_deadline_are_hard():
    value = state()
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    assert admission(value, 1, now=now)['projected_with_exposure_usd'] == 17
    with pytest.raises(ValueError, match='cutoff'):
        admission(value, 74, now=now)
    with pytest.raises(ValueError, match='closure'):
        admission(dict(value, independent_closure_verified=False), 1, now=now)
    with pytest.raises(ValueError, match='closed'):
        admission(value, 1, now=datetime(2026, 10, 10, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match='Finite'):
        admission(value, float('nan'), now=now)
    with pytest.raises(ValueError, match='Finite'):
        admission(value, True, now=now)


def test_stop_targets_owned_ids_and_confirmed_intents_only():
    value = state()
    original = dict(id='123', name='original', zone='zones/us-central1-a')
    unknown = dict(id='321', name='unowned', zone='zones/us-central1-a')
    assert targets(value, [original, unknown]) == [original]
    with pytest.raises(ValueError, match='changed'):
        targets(value, [dict(original, name='tampered')])
    intent = dict(type='vm', name='cloudrag-i5-cpu', zone='us-central1-a', ownership_marker='CloudRAG-I5-own')
    value['resource_intents'] = [intent]
    created = dict(id='456', name=intent['name'], zone='zones/us-central1-a', description=intent['ownership_marker'])
    assert targets(value, [created]) == [created]
    with pytest.raises(ValueError, match='ownership'):
        targets(value, [dict(created, description='unowned')])


def test_independent_stop_closes_admission_and_preserves_only_technical_projection(tmp_path):
    (tmp_path/'STATE.json').write_text(json.dumps(state()))
    original = dict(id='123', name='original', zone='zones/us-central1-a',
                    status='RUNNING', metadata={'secret': 'PRIVATE_CANARY_NOT_FOR_EVIDENCE'})
    calls = []

    class Cloud:
        def command(self, argv, **kwargs):
            calls.append(argv)
            if argv[-1] == 'list':
                assert kwargs.get('private_output') is True
                return [dict(original, status='TERMINATED' if len(calls) > 1 else 'RUNNING')]
            assert argv == ['compute', 'instances', 'stop', 'original', '--zone=us-central1-a']
            assert kwargs['timeout'] == 600

    result = close(tmp_path, Cloud())
    assert result['status'] == 'OWN_VMS_TERMINATED_VERIFIED' and len(calls) == 3
    assert json.loads((tmp_path/'STATE.json').read_bytes())['status'] == 'CLOSING'
    assert 'PRIVATE_CANARY' not in json.dumps(result)
    assert 'PRIVATE_CANARY' not in next(tmp_path.glob('safety-close-*.json')).read_text()


def test_empty_census_does_not_prove_closure(tmp_path):
    (tmp_path/'STATE.json').write_text(json.dumps(state()))

    class Cloud:
        def command(self, argv, **kwargs):
            return []

    with pytest.raises(ValueError, match='empty census'):
        close(tmp_path, Cloud())
    assert not list(tmp_path.glob('safety-close-*.json'))


def test_disposed_vm_is_not_expected_alive_but_reappearance_is_rejected(tmp_path):
    value = state()
    value['resources'].append(dict(type='vm', id='456', name='cloudrag-i4-test',
        zone='us-central1-b', disposed=True, absence_verified=True))
    original = dict(id='123', name='original', zone='zones/us-central1-a', status='TERMINATED')
    (tmp_path/'STATE.json').write_text(json.dumps(value))

    class Cloud:
        def command(self, argv, **kwargs):
            assert argv == ['compute', 'instances', 'list']
            return [original]

    assert close(tmp_path, Cloud())['status'] == 'OWN_VMS_TERMINATED_VERIFIED'
    for identity in ('456', '789'):
        with pytest.raises(ValueError, match='reappeared'):
            targets(value, [dict(id=identity, name='cloudrag-i4-test', zone='zones/us-central1-b')])
