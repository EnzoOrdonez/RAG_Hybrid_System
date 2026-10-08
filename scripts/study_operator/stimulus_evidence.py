"""Verify complete prospective boot evidence; synthetic observations never accept."""
import hashlib
import json
import math
import re

from scripts.study_operator.stimulus_calendar import V2_VERSION, analyze, calendar
from src.evaluation.decline_classifier import CLASSIFIER_VERSION, classify_response

OPTIONS = dict(temperature=0, num_predict=1024, seed=42, num_ctx=4096)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def software_projection(inventory):
    """Exclude boot/VM/GPU UUID and location only; retain effective numerical controls."""
    image, service, observed = inventory['image'], inventory['service'], inventory['observed']
    gpu = [part.strip() for part in observed['gpu'].split(',')]
    if (len(gpu) != 4 or gpu[1] != 'NVIDIA L4'
            or image['image_id'] != image['container_image_id']
            or not re.fullmatch(r'sha256:[a-f0-9]{64}', image['image_id'])
            or not re.fullmatch(r'[a-f0-9]{40}', inventory['source']['commit'])
            or service['mode'] != 'fresh_runner' or service['policy']['generation_options'] != OPTIONS
            or observed['device'] != '1'):
        raise ValueError('Final L4 image, source and fresh-runner policy required')
    result = {key: inventory[key] for key in ('source', 'recipes', 'dependencies', 'execution_environment',
              'locks', 'vendor', 'ollama', 'artifacts')}
    result.update(image_id=image['image_id'], protocol_fingerprint=inventory['protocol']['fingerprint'],
        service=service, hardware=dict(machine_type='g2-standard-4', gpu_name=gpu[1], driver=gpu[2], memory_total=gpu[3]),
        platform=observed['platform'], prer_sha256=observed['preregistration'])
    return result


def verify_boot(proof, config):
    index = proof['boot_index']
    slots = calendar(config).get(index)
    if (type(index) is not int or slots is None or proof.get('status') != 'COMPLETE'
            or proof.get('mode') not in {'LIVE', 'SYNTHETIC'} or len(proof['rows']) != len(slots)):
        raise ValueError('Complete registered boot with explicit execution mode required')
    inventory = proof['inventory']
    boot = inventory['observed']['boot_id']
    if (not boot or proof['initial_service_state'] != dict(mode='fresh_runner', phase='STARTING', sequence=1, boot_id=boot)
            or inventory['observed']['machine-type'].split('/')[-1] != 'g2-standard-4'
            or proof['identity_verified_live'] is not (proof['mode'] == 'LIVE')):
        raise ValueError('Cold initial service and verified execution identity required')
    software = digest(software_projection(inventory))
    rows, previous_pids = [], set()
    for position, (row, slot) in enumerate(zip(proof['rows'], slots, strict=True), 1):
        if (row['boot_index'] != index or row['index'] != position or row['slot'] != slot
                or row['boot_id'] != boot or row['software_sha256'] != software
                or row.get('synthetic') is not (proof['mode'] == 'SYNTHETIC')
                or row['response_class_version'] != V2_VERSION or CLASSIFIER_VERSION != V2_VERSION
                or row['response_class'] != classify_response(row['answer'])
                or row.get('generation_options') != OPTIONS or not re.fullmatch('[a-f0-9]{64}', row['request_sha256'])
                or type(row['elapsed_s']) not in (int, float) or not math.isfinite(row['elapsed_s'])
                or not 0 < row['elapsed_s'] <= 600 or row.get('valid') is not True or row['status'] != 'success'):
            raise ValueError('Invalid, mixed, altered or unobserved stimulus call')
        state = row['service_after']
        pids = state.get('runner_pids', [])
        if (state['boot_id'] != boot or state['mode'] != 'fresh_runner' or state['phase'] != 'RESIDENT'
                or state['sequence'] != 1+4*position or not pids or len(set(pids)) != len(pids)
                or any(type(pid) is not int or pid <= 0 for pid in pids) or previous_pids.intersection(pids)):
            raise ValueError('Runner renewal or exclusive cold history not demonstrated')
        previous_pids = set(pids)
        rows.append(row)
    return dict(boot_index=index, boot_id=boot, mode=proof['mode'], software_sha256=software, rows=rows)


def accept(proofs, config):
    if len(proofs) != 12:
        raise ValueError('Twelve complete registered boots required')
    verified = [verify_boot(proof, config) for proof in proofs]
    if {row['boot_index'] for row in verified} != set(range(1, 13)):
        raise ValueError('Duplicate or missing boot index')
    # Mode comes from each boot and each observation, never an analyzer flag.
    synthetic = any(proof['mode'] != 'LIVE' for proof in verified)
    result = analyze([row for proof in verified for row in proof['rows']], config, synthetic=synthetic)
    result.update(evidence_bound=True, runner_resets_verified=True, synthetic_cannot_grant_acceptance=True,
        context_identity_not_inferred=True, software_projection_excludes=['physical_gpu_uuid', 'vm_id', 'boot_id', 'location'])
    return result
