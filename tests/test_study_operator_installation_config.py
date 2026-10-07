import copy

import pytest

from scripts.study_operator.installation_config import derive


def fixture(tmp_path):
    base = dict(network='study-net', sessions_bucket='sessions', technical_bucket='evidence',
        static_ip='203.0.113.1', hostname='old.sslip.io', prepared_snapshot={'id': 'old'},
        period_ids={'study': 'old-period'}, period_id='old-period',
        official_rates=dict(compute_usd_h=.7, associated_ip_usd_h=.005, unused_ip_usd_h=.01),
        host_code='/srv/cloudrag/iteration5/code-final02', host_infrastructure_commit='old')
    image = 'sha256:'+'a'*64
    build = dict(status='TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED', image_id=image, commit='b'*40)
    proof = dict(status='CPU_RESTORATION_VERIFIED', synthetic=False, image_id=image,
        source_snapshot_id='900', all_expected_files_verified=True, image_config_verified=True,
        model_manifest_and_blobs_verified=True, source=dict(files=19), artifacts=dict(files=79),
        runtime_user_pair=dict(status='PAIRED_RUNTIME_USER_SUPPORTED'))
    return base, dict(build=build, restoration=proof, snapshot=dict(name='final', id='900', status='READY'),
        costs=dict(not_invoice=True, cost=dict(as_of_utc='2026-10-07T00:00:00Z', estimated_spend_usd=13,
            reserved_retention_and_closure_usd=24, current_idle_upper_usd_day=.429)),
        quote=dict(machine='g2-standard-4', region='us-west1', usd_per_hour='.706832255'),
        original=dict(name='original', id='123', zone='zones/us-central1-a', status='TERMINATED'),
        subnet=dict(name='west1-study', zone='us-west1-a', privateIpGoogleAccess=True, network='networks/study-net'),
        inputs=dict(root='/srv/cloudrag/iteration5/inputs/'+'c'*64,
                    files=[dict(name='service-preregistration.md', sha256='d'*64)]),
        fingerprint='e'*64, proof_path=tmp_path/'proof.json', proof_sha256='f'*64,
        operator_commit='9'*40, package=tmp_path/'iteration5-run-fixture', python=tmp_path/'existing-python')


def test_derive_separates_image_owner_cost_and_protocol_without_stale_ip(tmp_path):
    base, inputs = fixture(tmp_path)
    old = copy.deepcopy(base)
    result = derive(base, **inputs)
    assert base == old
    assert result['commit'] == result['host_infrastructure_commit'] == inputs['build']['commit']
    assert result['operator_commit'] == inputs['operator_commit']
    assert result['zone'] == 'us-west1-a' and result['primary_vm']['zone'] == 'us-central1-a'
    assert result['cost']['estimated_usd'] == 13 and result['cost']['margin_usd'] == 24
    assert result['cost']['retention_usd_day'] == .429
    assert result['official_rates']['compute_usd_h'] == .706832255
    assert result['purpose'] == 'technical' and len(result['period_id']) == 32
    assert all(key not in result for key in ('static_ip', 'hostname', 'prepared_snapshot', 'period_ids'))
    assert result['ip_name'].startswith('cloudrag-i5-static-us-west1-')
    assert result['preregistration_sha256'] == 'd'*64
    assert result['final_snapshot']['restoration_proof_sha256'] == 'f'*64
    assert result['qualification_status'].endswith('REAL_GPU_ACCEPTANCE_PENDING')


@pytest.mark.parametrize('tamper', ['image', 'snapshot', 'synthetic', 'artifact_count', 'runtime_pair',
                                  'quote_region', 'flow_logs', 'private_access', 'running_original', 'cost'])
def test_derive_rejects_incomplete_or_mixed_evidence(tmp_path, tamper):
    base, inputs = fixture(tmp_path)
    if tamper == 'image':
        inputs['build']['image_id'] = 'sha256:'+'0'*64
    elif tamper == 'snapshot':
        inputs['snapshot']['id'] = '901'
    elif tamper == 'synthetic':
        inputs['restoration']['synthetic'] = True
    elif tamper == 'artifact_count':
        inputs['restoration']['artifacts']['files'] = 78
    elif tamper == 'runtime_pair':
        inputs['restoration']['runtime_user_pair']['status'] = 'NOT_MEASURED'
    elif tamper == 'quote_region':
        inputs['quote']['region'] = 'us-east1'
    elif tamper == 'flow_logs':
        inputs['subnet']['enableFlowLogs'] = True
    elif tamper == 'private_access':
        inputs['subnet']['privateIpGoogleAccess'] = False
    elif tamper == 'running_original':
        inputs['original']['status'] = 'RUNNING'
    else:
        inputs['costs']['cost']['estimated_spend_usd'] = 90
    with pytest.raises(ValueError):
        derive(base, **inputs)
