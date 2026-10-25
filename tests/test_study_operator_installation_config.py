import copy
import json

import pytest

from scripts.study_operator.installation_config import derive, qualified_inputs


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
        costs=dict(not_invoice=True, regional_transfer_upper_basis=dict(rate_usd_gib=.02),
                   cost=dict(as_of_utc='2026-10-07T00:00:00Z', estimated_spend_usd=13,
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


def test_derive_accepts_reconciled_cost_schema_without_changing_image_identity(tmp_path):
    base, inputs = fixture(tmp_path)
    costs = inputs['costs']
    costs.pop('not_invoice')
    costs['cost']['not_invoice'] = True
    costs.pop('regional_transfer_upper_basis')
    costs['transfer_margin_basis'] = dict(quote=dict(usd_per_usage_unit='0.02'))
    result = derive(base, **inputs)
    assert result['official_rates']['snapshot_transfer_na_usd_gib'] == .02
    assert result['image_id'] == inputs['build']['image_id']


@pytest.mark.parametrize('rate', ['NaN', 'Infinity', '-0.01'])
def test_unquoted_reconciled_transfer_rate_is_rejected(tmp_path, rate):
    base, inputs = fixture(tmp_path)
    inputs['costs'].pop('regional_transfer_upper_basis')
    inputs['costs']['transfer_margin_basis'] = dict(quote=dict(usd_per_usage_unit=rate))
    with pytest.raises(ValueError):
        derive(base, **inputs)


def qualified_fixture(root):
    _, values = fixture(root)
    rows = {
        'cpu-restoration-final03r-proof.json': values['restoration'],
        'linux-build-final03-download-proof.json': values['build'],
        'final-snapshot-final03-receipt.json': dict(snapshot=values['snapshot'],
            image_id=values['build']['image_id'], commit=values['build']['commit']),
        'cost-reconciliation249.json': values['costs'],
        'STATE.json': dict(resources=[dict(type='snapshot', id='900', name='final', disposed=False)])}
    for name, value in rows.items():
        (root/name).write_text(json.dumps(value), encoding='utf-8')


def test_selected_final3_inputs_are_pinned_without_reading_deleted_final2(tmp_path):
    qualified_fixture(tmp_path)
    proof, build, creation, costs, pins = qualified_inputs(tmp_path, 'final03', 'final03r',
                                                        'cost-reconciliation249.json')
    assert proof['image_id'] == build['image_id'] == creation['image_id']
    assert costs['cost']['estimated_spend_usd'] == 13
    assert len(pins) == 4 and all(len(value) == 64 for value in pins.values())
    assert all('final02' not in key for key in pins)


@pytest.mark.parametrize('defect', ['disposed', 'mixed_image', 'snapshot', 'missing_selected'])
def test_selected_qualification_never_falls_back_or_adopts_mixed_identity(tmp_path, defect):
    qualified_fixture(tmp_path)
    if defect == 'missing_selected':
        (tmp_path/'cpu-restoration-final03r-proof.json').unlink()
        with pytest.raises(FileNotFoundError):
            qualified_inputs(tmp_path, 'final03', 'final03r', 'cost-reconciliation249.json')
        return
    name = 'STATE.json' if defect == 'disposed' else 'final-snapshot-final03-receipt.json'
    value = json.loads((tmp_path/name).read_bytes())
    if defect == 'disposed':
        value['resources'][0]['disposed'] = True
    elif defect == 'mixed_image':
        value['image_id'] = 'another-image'
    else:
        value['snapshot']['id'] = '901'
    (tmp_path/name).write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        qualified_inputs(tmp_path, 'final03', 'final03r', 'cost-reconciliation249.json')


@pytest.mark.parametrize('build,restore,cost', [('../final03','final03r','cost-reconciliation249.json'),
    ('final03','../final03r','cost-reconciliation249.json'),
    ('final03','final03r','../cost-reconciliation249.json')])
def test_qualification_input_paths_are_confined_to_receipts(tmp_path, build, restore, cost):
    with pytest.raises(ValueError, match='Safe'):
        qualified_inputs(tmp_path, build, restore, cost)
