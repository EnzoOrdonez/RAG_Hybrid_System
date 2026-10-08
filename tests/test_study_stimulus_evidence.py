import copy

import pytest

from scripts.study_operator.stimulus_calendar import TASKS, V2_VERSION, calendar
from scripts.study_operator.stimulus_evidence import OPTIONS, accept, digest, software_projection, verify_boot
from src.evaluation.decline_classifier import classify_response

CONFIG = dict(tasks=dict(T1=list(TASKS[:3]), T2=list(TASKS[3:])), labels=dict(A='hybrid', B='no_rag'))


def fixture():
    proofs = []
    for boot, slots in calendar(CONFIG).items():
        inventory = dict(source=dict(commit='a'*40, tree='b'*40), recipes={}, dependencies=[], execution_environment={},
            locks={}, vendor={}, ollama={}, artifacts={}, protocol=dict(fingerprint='c'*64),
            image=dict(image_id='sha256:'+'d'*64, container_image_id='sha256:'+'d'*64),
            service=dict(mode='fresh_runner', policy=dict(generation_options=OPTIONS)),
            observed=dict(boot_id='fixture-boot-'+str(boot), gpu='GPU-physical-'+str(boot)+', NVIDIA L4, fixture-driver, 23034 MiB',
                          device='1', platform='fixture-platform', preregistration='e'*64,
                          **{'machine-type': 'fixture/g2-standard-4'}))
        software = digest(software_projection(inventory))
        rows = []
        for index, slot in enumerate(slots, 1):
            answer = ('This is a synthetic answer describing a fictional service with useful technical detail. '*5
                      +str(slot.get('query_id', 'free'))+slot['condition'])
            rows.append(dict(boot_index=boot, index=index, slot=slot, boot_id=inventory['observed']['boot_id'],
                software_sha256=software, synthetic=True, status='success', valid=True, response_class_version=V2_VERSION,
                raw_text=answer, answer=answer, response_class=classify_response(answer), citations=[],
                generation_options=OPTIONS, request_sha256='f'*64, elapsed_s=0.1,
                service_after=dict(boot_id=inventory['observed']['boot_id'], mode='fresh_runner', phase='RESIDENT',
                                   sequence=1+4*index, runner_pids=[100+index])))
        proofs.append(dict(boot_index=boot, mode='SYNTHETIC', status='COMPLETE', inventory=inventory, rows=rows,
            identity_verified_live=False, initial_service_state=dict(mode='fresh_runner', phase='STARTING',
                sequence=1, boot_id=inventory['observed']['boot_id'])))
    return proofs


def test_complete_144_call_dry_evidence_never_accepts():
    result = accept(fixture(), CONFIG)
    assert result['status'] == 'SYNTHETIC_NOT_ACCEPTANCE'
    assert result['calls'] == 144 and result['accepted_combinations'] == 12


@pytest.mark.parametrize('defect', ['missing_boot', 'duplicate_boot', 'warm_first', 'unobserved_reset',
    'reused_pid', 'old_class', 'modified_options', 'timeout', 'synthetic_relabel', 'software_mix'])
def test_invalid_census_or_execution_evidence_rejected(defect):
    proofs = fixture()
    row = proofs[0]['rows'][1]
    if defect == 'missing_boot':
        proofs.pop()
    elif defect == 'duplicate_boot':
        proofs[-1] = copy.deepcopy(proofs[0])
    elif defect == 'warm_first':
        proofs[0]['initial_service_state']['sequence'] = 5
    elif defect == 'unobserved_reset':
        row['service_after']['sequence'] += 4
    elif defect == 'reused_pid':
        row['service_after']['runner_pids'] = proofs[0]['rows'][0]['service_after']['runner_pids']
    elif defect == 'old_class':
        row['response_class_version'] = 'old'
    elif defect == 'modified_options':
        row['generation_options'] = dict(OPTIONS, num_predict=512)
    elif defect == 'timeout':
        row['elapsed_s'] = 600.1
    elif defect == 'synthetic_relabel':
        proofs[0]['mode'] = 'LIVE'
        proofs[0]['identity_verified_live'] = True
    else:
        proofs[-1]['inventory']['execution_environment']['ONEDNN_MAX_CPU_ISA'] = 'changed'
    with pytest.raises(ValueError):
        accept(proofs, CONFIG)


def test_boot_verification_does_not_modify_evidence():
    proof = fixture()[0]
    before = copy.deepcopy(proof)
    verify_boot(proof, CONFIG)
    assert proof == before


def test_physical_gpu_uuid_is_excluded_but_driver_and_numerical_controls_remain():
    inventory = fixture()[0]['inventory']
    other = copy.deepcopy(inventory)
    other['observed']['gpu'] = 'GPU-other, NVIDIA L4, fixture-driver, 23034 MiB'
    assert software_projection(inventory) == software_projection(other)
    other['observed']['gpu'] = 'GPU-other, NVIDIA L4, changed-driver, 23034 MiB'
    assert software_projection(inventory) != software_projection(other)
