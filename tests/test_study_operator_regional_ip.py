from datetime import datetime, timezone
import json

import pytest

from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_lifecycle import installation


def fixture(tmp_path):
    operator, cloud = installation(tmp_path)
    operator.config.update(zone='us-west1-a', ip_name='cloudrag-i5-west1-fixture')
    operator.now = lambda: datetime(2026, 10, 7, tzinfo=timezone.utc)
    addresses = []
    original = cloud.command

    def command(args, **options):
        if args[:3] == ['compute', 'addresses', 'list']:
            return addresses
        if args[:3] == ['compute', 'addresses', 'create']:
            assert '--region=us-west1' in args
            marker = next(a.split('=', 1)[1] for a in args if a.startswith('--description='))
            addresses.append(dict(name=args[3], id='777', region='regions/us-west1', addressType='EXTERNAL',
                description=marker, address='203.0.113.9', creationTimestamp='2026-10-07T00:00:00+00:00'))
            return
        if args[:3] == ['compute', 'addresses', 'describe']:
            assert '--region=us-west1' in args
            return addresses[0]
        if args[:3] == ['compute', 'addresses', 'delete']:
            assert '--region=us-west1' in args
            addresses.clear()
            return
        return original(args, **options)

    cloud.command = command
    return operator, cloud, addresses


def test_new_region_ip_never_attaches_to_original_central_vm(tmp_path):
    operator, cloud, addresses = fixture(tmp_path)
    result = operator.ip_reserve()
    assert result['association_pending_bootstrap'] and result['idle_usd_day'] == .24
    assert operator.state['reserved_address_region'] == 'us-west1'
    saved = json.loads((operator.root/'active.json').read_bytes())
    assert saved['reserved_address_id'] == addresses[0]['id']
    assert saved['reserved_address_region'] == 'us-west1'
    assert not any(a[:3] == ['compute', 'instances', 'add-access-config'] for a, _ in cloud.calls)
    assert operator.ip_reserve()['address_id'] == '777'


def test_pending_bootstrap_ip_releases_in_actual_new_region(tmp_path):
    operator, _, addresses = fixture(tmp_path)
    operator.ip_reserve()
    assert operator.ip_release()['status'] == 'STATIC_IP_RELEASED_VERIFIED'
    assert not addresses and 'reserved_address_region' not in operator.state
    assert 'static_ip' not in operator.config
    assert operator.state['audit_resources'][0]['disposed']
    assert operator.state['audit_resources'][0]['absence_verified']


def test_region_cannot_be_changed_while_old_ip_is_reserved(tmp_path):
    operator, _, _ = fixture(tmp_path)
    operator.ip_reserve()
    operator.config['zone'] = 'us-east1-b'
    with pytest.raises(OperatorError, match='región'):
        operator.ip_reserve()
