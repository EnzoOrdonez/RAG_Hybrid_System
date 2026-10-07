import copy
import json

import pytest

from scripts.study_operator.bootstrap_region import configure, from_file
from scripts.study_operator.policy import OperatorError
from tests.test_study_operator_bootstrap import fixture as bootstrap_fixture


def fixture(tmp_path):
    operator, cloud, _, vms, _, _ = bootstrap_fixture(tmp_path)
    operator.config.update(purpose='technical', ip_name='cloudrag-i5-static-us-central1-fixture',
        configuration_provenance={})
    operator.config.pop('static_ip')
    operator.config.pop('hostname')
    operator.state.clear()
    candidate = copy.deepcopy(operator.config)
    candidate.update(zone='us-west1-a', subnet='west1-study', ip_name='cloudrag-i5-static-us-west1-fixture',
        period_id='NEW_CANDIDATE_PERIOD_NOT_ADOPTED')
    candidate['official_rates']['compute_usd_h'] = .8
    quote = dict(machine='g2-standard-4', region='us-west1', usd_per_hour='.8')
    original = cloud.command
    flags = dict(old_ip_present=False, subnet_flow_logs=False)
    def command(args, **options):
        if args[:3] == ['compute', 'addresses', 'list']:
            return [dict(id='333')] if flags['old_ip_present'] else []
        if args[:4] == ['compute', 'networks', 'subnets', 'describe']:
            return dict(network='networks/study-net', region='regions/us-west1', privateIpGoogleAccess=True,
                enableFlowLogs=flags['subnet_flow_logs'])
        return original(args, **options)
    cloud.command = command
    return operator, candidate, quote, vms, flags


def test_regional_configuration_keeps_identity_period_and_prior_costs(tmp_path):
    operator, candidate, quote, _, _ = fixture(tmp_path)
    original = copy.deepcopy(operator.config)
    operator.state.update(cost={'estimated_usd': 1, 'margin_usd': 2, 'reservations': {'disk-retention-prior': 3}},
        primary_creation_intent=dict(name='cloudrag-i5-prior', disk_name='cloudrag-i5-prior-boot',
            zone='us-central1-b', capacity_error_code='ZONE_RESOURCE_POOL_EXHAUSTED', ownership_marker='CloudRAG-I5-prior'))
    costs = copy.deepcopy(operator.state['cost'])
    assert configure(operator, candidate, quote=quote)['status'] == 'BOOTSTRAP_REGION_CONFIGURED_NOT_MEASURED'
    assert operator.config['period_id'] == original['period_id']
    assert operator.config['primary_vm'] == original['primary_vm']
    assert operator.config['image_id'] == original['image_id']
    assert operator.config['official_rates']['compute_usd_h'] == .8
    assert operator.state['cost'] == costs and len(operator.state['primary_capacity_attempts']) == 1
    assert not operator.state.get('primary_creation_intent')
    saved = json.loads(operator.state_path.read_bytes())
    assert saved['bootstrap_region_history'][0]['previous_ip_absent']
    assert configure(operator, candidate, quote=quote)['status'] == 'BOOTSTRAP_REGION_ALREADY_CONFIGURED'


@pytest.mark.parametrize('defect', ['completed', 'study', 'reserved_ip', 'live_ip', 'uncertain_create',
    'prior_vm', 'gpu_running', 'image', 'protocol', 'network', 'quote', 'flow_logs', 'ip_name'])
def test_relocation_rejects_unsafe_or_different_environment_without_mutations(tmp_path, defect):
    operator, candidate, quote, vms, flags = fixture(tmp_path)
    if defect == 'completed':
        operator.state['primary_bootstrap_complete'] = True
    elif defect == 'study':
        operator.config['purpose'] = 'study'
    elif defect == 'reserved_ip':
        operator.state['reserved_address_id'] = '333'
    elif defect == 'live_ip':
        flags['old_ip_present'] = True
    elif defect in {'uncertain_create', 'prior_vm'}:
        operator.state['primary_creation_intent'] = dict(name='cloudrag-i5-prior', zone='us-central1-b')
        if defect == 'prior_vm':
            operator.state['primary_creation_intent']['capacity_error_code'] = 'ZONE_RESOURCE_POOL_EXHAUSTED'
            vms.append(dict(name='cloudrag-i5-prior', id='998', zone='zones/us-central1-b', status='TERMINATED'))
    elif defect == 'gpu_running':
        vms.append(dict(name='other', id='998', zone='zones/us-central1-c', status='RUNNING', machineType='machines/g2-standard-4'))
    elif defect in {'image', 'protocol', 'network'}:
        candidate[{'image': 'image_id', 'protocol': 'fingerprint', 'network': 'network'}[defect]] = 'CHANGED'
    elif defect == 'quote':
        quote['usd_per_hour'] = '.9'
    elif defect == 'flow_logs':
        flags['subnet_flow_logs'] = True
    else:
        candidate['ip_name'] = 'foreign'
    before_config, before_state = copy.deepcopy(operator.config), copy.deepcopy(operator.state)
    with pytest.raises(OperatorError):
        configure(operator, candidate, quote=quote)
    assert operator.config == before_config and operator.state == before_state
    assert not operator.state_path.exists()


def test_sealed_run_is_not_reopened_by_configuration_file(tmp_path):
    operator, candidate, _, _, _ = fixture(tmp_path)
    path = tmp_path/'candidate.json'
    path.write_text(json.dumps(candidate))
    with pytest.raises(OperatorError, match='sellado'):
        from_file(operator, path)
