import copy

import pytest

from scripts.study_operator.stimulus_evidence import digest, software_projection, verify_boot
from scripts.study_operator.stimulus_host_evidence import admission_receipt, sha


def live_proof():
    from test_study_stimulus_evidence import fixture
    from test_study_cold_admission import rows

    proof = fixture()[4]
    proof.update(mode='LIVE',identity_verified_live=True)
    inventory = proof['inventory']
    inventory['ollama']['digest'] = 'a'*64
    row = proof['rows'][0]
    row.update(synthetic=False,software_sha256=digest(software_projection(inventory)))
    samples = rows()
    for sample in samples:
        sample['owned_pids'] = [100]
        sample['service_state']['boot_id'] = inventory['observed']['boot_id']
    last = copy.deepcopy(samples[-1])
    last['monotonic_s'] = 163
    last['service_state'].update(row['service_after'],request_id='b'*32,written_monotonic_s=162)
    last['ollama_ps_api']['models'] = [dict(digest='a'*64,context_length=4096,expires_at='2100-01-01T00:00:00+00:00')]
    host = dict(schema_version=1,mode='LIVE',status='COMPLETE',boot_id=inventory['observed']['boot_id'],boot_index=5,
        app_image_id=inventory['image']['image_id'],source_commit=inventory['source']['commit'],
        own_container_id='a'*64,ollama_container_id='b'*64,isolation_verified=True,metadata_unreachable=True,
        native_limit_s=7200,preparation_started=100,admission_started=100,admission_ended=160,
        admission_samples=samples,samples=samples+[last],ended=163,
        calls=[dict(index=1,started=161,ended=162,row_sha256=sha(row))])
    proof.update(host_evidence=host,host_admission_sha256=sha(admission_receipt(host)))
    return proof


def test_live_proof_is_bound_to_actual_host_admission_without_modifying_it():
    from test_study_stimulus_evidence import CONFIG

    proof = live_proof()
    before = copy.deepcopy(proof)
    assert verify_boot(proof,CONFIG)['mode'] == 'LIVE'
    assert proof == before


@pytest.mark.parametrize('defect',['missing','gap','warm','admission_sha','foreign_gpu','named_ollama_gpu',
    'named_ollama_cpu','deadline','native','image','source','row','call_time','no_call','no_isolation'])
def test_live_host_admission_and_measurement_defects_reject(defect):
    from test_study_stimulus_evidence import CONFIG

    proof = live_proof()
    host = proof['host_evidence']
    if defect == 'missing':
        del proof['host_evidence']
    elif defect == 'gap':
        host['samples'] = host['samples'][:1]+host['samples'][5:]
    elif defect == 'warm':
        host['admission_samples'][0]['service_state']['sequence'] = 5
    elif defect == 'admission_sha':
        proof['host_admission_sha256'] = 'f'*64
    elif defect in {'foreign_gpu','named_ollama_gpu','named_ollama_cpu'}:
        for row in host['samples']:
            row['processes'] = [dict(pid=999,name='ollama',cpu_percent=20 if defect.endswith('cpu') else 0)]
            row['gpu_pids'] = [] if defect.endswith('cpu') else [999]
    elif defect == 'deadline':
        host['ended'] = 8000
    elif defect == 'native':
        host['native_limit_s'] = 9000
    elif defect == 'image':
        host['app_image_id'] = 'sha256:'+'f'*64
    elif defect == 'source':
        host['source_commit'] = 'f'*40
    elif defect == 'row':
        host['calls'][0]['row_sha256'] = 'f'*64
    elif defect == 'call_time':
        host['calls'][0]['started'] = 90
    elif defect == 'no_call':
        host['calls'] = []
    else:
        host['isolation_verified'] = False
    with pytest.raises(ValueError):
        verify_boot(proof,CONFIG)


def test_older_runner_pid_cannot_reappear_after_an_intervening_runner():
    from test_study_stimulus_evidence import CONFIG, fixture

    proof = fixture()[0]
    proof['rows'][2]['service_after']['runner_pids'] = proof['rows'][0]['service_after']['runner_pids']
    with pytest.raises(ValueError,match='renewal'):
        verify_boot(proof,CONFIG)
