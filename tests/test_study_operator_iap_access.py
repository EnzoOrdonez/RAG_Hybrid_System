from datetime import datetime, timezone
import json

import pytest

from scripts.study_operator import iap_access
from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_lifecycle import installation


def fixture(tmp_path):
    operator, cloud = installation(tmp_path)
    operator.config.update(network='study-network', iap_name='cloudrag-i5-owner-fixture-iap')
    operator.now = lambda: datetime(2026, 10, 7, tzinfo=timezone.utc)
    rows, calls = [], []
    original = cloud.command

    def command(args, **options):
        calls.append(args)
        if args[:3] == ['compute', 'firewall-rules', 'list']:
            return list(rows)
        if args[:3] == ['compute', 'firewall-rules', 'create']:
            saved = json.loads(operator.state_path.read_bytes())
            marker = saved['iap_creation_intent']['ownership_marker']
            assert '--description='+marker in args
            assert '--source-ranges=35.235.240.0/20' in args and '--rules=tcp:22' in args
            assert '--target-tags=cloudrag-i3-managed' in args
            rows.append(dict(name=args[3], id='777', description=marker, network='networks/study-network',
                direction='INGRESS', sourceRanges=[iap_access.SOURCE], targetTags=[iap_access.TAG],
                allowed=[dict(IPProtocol='tcp', ports=['22'])], creationTimestamp=operator.now().isoformat()))
            return
        if args[:3] == ['compute', 'firewall-rules', 'delete']:
            rows.clear()
            return
        return original(args, **options)

    cloud.command = command
    return operator, rows, calls


def test_exact_private_iap_creation_is_idempotent_and_receipted(tmp_path):
    operator, rows, calls = fixture(tmp_path)
    assert iap_access.prepare(operator)['id'] == '777'
    assert iap_access.prepare(operator)['id'] == '777'
    assert len([args for args in calls if args[2] == 'create']) == 1
    assert operator.state['audit_resources'][0]['id'] == rows[0]['id']
    assert iap_access.release(operator)['status'] == 'OWN_IAP_RULE_ABSENT_VERIFIED'
    assert not rows and operator.state['audit_resources'][0]['absence_verified']
    assert iap_access.release(operator)['status'] == 'OWN_IAP_RULE_ABSENT_VERIFIED'


def test_successful_create_lost_response_reconciles_without_second_create(tmp_path):
    operator, _, calls = fixture(tmp_path)
    command = operator.cloud.command
    def lose(args, **options):
        result = command(args, **options)
        if args[:3] == ['compute', 'firewall-rules', 'create']:
            raise OperatorError('LOST_RESPONSE')
        return result
    operator.cloud.command = lose
    with pytest.raises(OperatorError, match='LOST_RESPONSE'):
        iap_access.prepare(operator)
    operator.cloud.command = command
    assert iap_access.prepare(operator)['id'] == '777'
    assert len([args for args in calls if args[2] == 'create']) == 1


@pytest.mark.parametrize('change', ['description', 'id', 'sourceRanges', 'allowed', 'targetTags', 'disabled', 'sourceServiceAccounts'])
def test_foreign_or_broad_rule_is_never_adopted_or_deleted(tmp_path, change):
    operator, rows, calls = fixture(tmp_path)
    iap_access.prepare(operator)
    rows[0][change] = {'description': 'foreign', 'id': '778', 'sourceRanges': ['0.0.0.0/0'],
        'allowed': [dict(IPProtocol='tcp', ports=['22', '8501'])], 'targetTags': [],
        'disabled': True, 'sourceServiceAccounts': ['foreign']}[change]
    with pytest.raises(OperatorError):
        iap_access.prepare(operator)
    with pytest.raises(OperatorError):
        iap_access.release(operator)
    assert not any(args[2] == 'delete' for args in calls)


def test_release_blocks_while_gpu_runs_and_checks_actual_absence(tmp_path):
    operator, rows, calls = fixture(tmp_path)
    iap_access.prepare(operator)
    operator.cloud.vm.update(status='RUNNING', guestAccelerators=[dict(acceleratorType='nvidia-l4')])
    with pytest.raises(OperatorError):
        iap_access.release(operator)
    assert not any(args[2] == 'delete' for args in calls)
    operator.cloud.vm['status'] = 'TERMINATED'
    command = operator.cloud.command
    operator.cloud.command = lambda args, **kw: None if args[2] == 'delete' else command(args, **kw)
    with pytest.raises(OperatorError, match='presente'):
        iap_access.release(operator)
    assert rows and operator.state['iap_rule_id'] == '777'
    assert not operator.state['audit_resources'][0].get('disposed')


def test_active_run_does_not_reuse_disposed_name(tmp_path):
    operator, _, calls = fixture(tmp_path)
    iap_access.prepare(operator)
    iap_access.release(operator)
    operator.config['audit_run'] = 'fixture-active'
    # Isolate the name reuse guard from global admission, tested separately.
    from unittest.mock import patch
    with patch.object(iap_access, 'admit'), pytest.raises(OperatorError, match='retirada'):
        iap_access.prepare(operator)
    assert len([args for args in calls if args[2] == 'create']) == 1
