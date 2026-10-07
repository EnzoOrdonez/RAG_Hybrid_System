"""Derive operator5 inputs from verified receipts, never from hand-entered hashes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

from scripts.study_operator.cloud_client import Cloud, checked_vm, no_other_gpu
from scripts.study_operator.pricing import quote_archive
from scripts.study_operator.region_scope import region
from scripts.study_operator.run_control import require_limited
from scripts.study_operator.startup_inputs import assemble
from src.ui.components.study_protocol import verify_draw


def derive(base, *, build, restoration, snapshot, costs, quote, original, subnet, inputs,
           fingerprint, proof_path, proof_sha256, operator_commit, package, python):
    zone = subnet['zone']
    selected_region = region(zone)
    if (build['status'] != 'TECHNICAL_BUILD_PASS_DOWNLOADED_VERIFIED'
            or restoration['status'] != 'CPU_RESTORATION_VERIFIED' or restoration['synthetic'] is not False
            or restoration['image_id'] != build['image_id'] or restoration['source_snapshot_id'] != str(snapshot['id'])
            or snapshot['status'] != 'READY' or not restoration['all_expected_files_verified']
            or not restoration['image_config_verified'] or not restoration['model_manifest_and_blobs_verified']
            or restoration['source']['files'] != 19 or restoration['artifacts']['files'] != 79
            or restoration['runtime_user_pair']['status'] != 'PAIRED_RUNTIME_USER_SUPPORTED'
            or quote['machine'] != 'g2-standard-4' or quote['region'] != selected_region
            or original['status'] != 'TERMINATED' or not subnet['privateIpGoogleAccess']
            or subnet.get('enableFlowLogs', False) or subnet['network'].split('/')[-1] != base['network']):
        raise ValueError('Final image, CPU qualification, cost quote or live infrastructure differs')
    if costs['not_invoice'] is not True or not 0 <= costs['cost']['estimated_spend_usd'] < 90:
        raise ValueError('Conservative cost receipt required')
    result = copy.deepcopy(base)
    for key in ('static_ip', 'hostname', 'prepared_snapshot', 'period_ids', 'period_id',
                'start_requested_utc', 'native_deadline_utc', 'configuration_provenance'):
        result.pop(key, None)
    result.update(zone=zone, subnet=subnet['name'], commit=build['commit'], image_id=build['image_id'],
        operator_commit=operator_commit, host_infrastructure_commit=build['commit'], purpose='technical',
        period_id=uuid.uuid4().hex, python=str(python), audit_run=str(package),
        primary_vm=dict(name=original['name'], id=str(original['id']), zone=original['zone'].split('/')[-1]),
        instance_id=str(original['id']), ip_name='cloudrag-i5-static-'+selected_region+'-'+Path(package).name.removeprefix('iteration5-run-').lower(),
        iap_name='cloudrag-i5-owner-'+Path(package).name.removeprefix('iteration5-run-').lower()+'-iap',
        bootstrap_inputs=inputs, fingerprint=fingerprint, reviewed_config=inputs['root']+'/reviewed',
        artifact_manifest=inputs['root']+'/deployment-artifacts.json',
        preregistration_file=inputs['root']+'/service-preregistration.md',
        preregistration_sha256=next(row['sha256'] for row in inputs['files'] if row['name'] == 'service-preregistration.md'),
        final_snapshot=dict(name=snapshot['name'], id=str(snapshot['id']),
            restoration_proof=str(proof_path), restoration_proof_sha256=proof_sha256),
        qualification_status='CPU_QUALIFIED_FINAL_IMAGE_REAL_GPU_ACCEPTANCE_PENDING')
    current = costs['cost']
    result['cost'] = dict(as_of_utc=current['as_of_utc'], estimated_usd=current['estimated_spend_usd'],
        margin_usd=current['reserved_retention_and_closure_usd'], retention_usd_day=current['current_idle_upper_usd_day'])
    result['cost_provenance'] = 'Conservative own receipt; spend separate from retention/closure and transfer upper margins. Not invoice.'
    result['official_rates']['compute_usd_h'] = float(quote['usd_per_hour'])
    result['official_rates']['persistent_disk_gib_usd_h'] = .1/730
    result['official_rates']['snapshot_gib_usd_h'] = .05/730
    result['official_rates']['snapshot_transfer_na_usd_gib'] = costs['regional_transfer_upper_basis']['rate_usd_gib']
    result['configuration_provenance'] = dict(cpu_restoration_verified=True, live_original_terminated=True,
        official_quote=quote, service_not_changed=True, rag_not_changed=True, ethics_record_not_created=True,
        old_operators_not_modified=True, fresh_gpu_identity_pending=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    for key in ('package', 'repo', 'base', 'zone', 'output'):
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args(argv)
    require_limited()
    package, repo = Path(args.package), Path(args.repo)
    base = json.loads(Path(args.base).read_bytes())
    proof_path = package/'cpu-restoration-final02r-proof.json'
    proof_bytes = proof_path.read_bytes()
    proof = json.loads(proof_bytes)
    build = json.loads((package/'linux-build-final02-download-proof.json').read_bytes())
    creation = json.loads((package/'final-snapshot-final02-receipt.json').read_bytes())
    # The immutable creation receipt names the snapshot; current status is read
    # separately. A declaration in STATE cannot substitute for the CPU proof.
    snapshot_name = next(row['name'] for row in json.loads((package/'STATE.json').read_bytes())['resources']
                         if row['type'] == 'snapshot' and str(row['id']) == proof['source_snapshot_id'] and not row.get('disposed'))
    if (proof['source_snapshot_id'] != str(creation['snapshot']['id'])
            or snapshot_name != creation['snapshot']['name'] or build['image_id'] != creation['image_id']
            or build['commit'] != creation['commit']):
        raise ValueError('CPU proof lacks the final snapshot creation receipt')
    quote = quote_archive(package/'official-compute-skus', 'g2-standard-4', region(args.zone))
    cloud = Cloud(base['gcloud'], base['project'], package/'installation-config-live')
    original = cloud.command(['compute', 'instances', 'describe', base['primary_vm']['name'],
                              '--zone='+base['primary_vm']['zone']])
    checked_vm(original, name=base['primary_vm']['name'], instance_id=base['primary_vm']['id'], zone=base['primary_vm']['zone'])
    no_other_gpu(cloud.command(['compute', 'instances', 'list']), selected_id='none')
    snapshot = cloud.command(['compute', 'snapshots', 'describe', snapshot_name])
    subnets = cloud.command(['compute', 'networks', 'subnets', 'list'])
    selected = [row for row in subnets if row['network'].split('/')[-1] == base['network']
                and row['region'].split('/')[-1] == region(args.zone)]
    if len(selected) != 1:
        raise ValueError('Expected one reviewed study subnet in selected region')
    subnet = dict(selected[0], zone=args.zone)
    reviewed = package/'study-config-reviewed-v2'
    protocol = verify_draw(reviewed)
    if protocol['config']['schema_version'] != 2:
        raise ValueError('Only reviewed UEQ-S protocol2 can be deployed')
    preregistration = repo/'docs/STUDY_ITERATION5_ACCEPTANCE_PREREGISTRATION.md'
    inputs = assemble(reviewed, 'C:/CloudRAG/autonomous-run-20261002T203006Z/deployment-artifacts.json', preregistration)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], timeout=30).decode().strip()
    result = derive(base, build=build, restoration=proof, snapshot=snapshot,
        costs=json.loads((package/'cost-reconciliation85.json').read_bytes()), quote=quote,
        original=original, subnet=subnet, inputs=inputs, fingerprint=protocol['fingerprint'],
        proof_path=proof_path, proof_sha256=hashlib.sha256(proof_bytes).hexdigest(),
        operator_commit=commit, package=package, python=repo/'.venv-app/Scripts/python.exe')
    result['configuration_provenance']['receipts_sha256'] = {
        name: hashlib.sha256((package/name).read_bytes()).hexdigest() for name in
        ('cpu-restoration-final02r-proof.json', 'linux-build-final02-download-proof.json',
         'final-snapshot-final02-receipt.json', 'cost-reconciliation85.json')}
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(dict(status='DERIVED_OPERATOR5_CONFIG_GPU_ACCEPTANCE_PENDING',
        image_id=result['image_id'], image_commit=result['commit'], operator_commit=commit,
        fingerprint=result['fingerprint'], region=region(args.zone), final_snapshot_id=str(snapshot['id']),
        config_sha256=hashlib.sha256(Path(args.output).read_bytes()).hexdigest(), token_or_ethics_not_created=True)))


if __name__ == '__main__':
    main()
