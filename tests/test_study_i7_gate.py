import pytest

from scripts.study_operator.i7_gate import GateRun, calendar, summarize

TASKS = ('q001', 'q068', 'q180', 'q016', 'q070', 'q172')


def test_pilot_is_separate_and_only_full_aggregate_decides():
    run = GateRun(TASKS, {'boot_id': 'boot', 'image_id': 'image'})
    slots = calendar(TASKS)
    assert len(slots) == 140
    assert [s['query_id'] for s in slots[:20]] == [s['query_id'] for s in slots[20:40]]
    for index, slot in enumerate(slots, 1):
        run.record(index, dict(status='success', valid=True, elapsed_s=20.,
                              **slot), started=index*25., ended=index*25.+21)
        assert run.decision()['decision'] == ('GO' if index == 140 else 'INCOMPLETE')
    result = run.decision()
    assert result['pilot_count'] == 20 and result['gate_count'] == 120
    assert result['systems']['hybrid']['n'] == 60
    assert result['systems']['no_rag']['p95'] == 20


def test_p95_is_linear_and_slow_rows_are_not_excluded():
    rows = [{'condition': 'hybrid', 'status': 'success', 'valid': True, 'elapsed_s': float(i)}
            for i in range(1, 61)]
    assert summarize(rows)['hybrid']['p95'] == pytest.approx(57.05)


@pytest.mark.parametrize('change', [{'valid': False}, {'status': 'failed'}, {'elapsed_s': float('nan')},
                                  {'identity': {'boot_id': 'other'}}, {'condition': 'wrong'}])
def test_bad_attempt_is_retained_and_cannot_resume(change):
    run = GateRun(TASKS, {'boot_id': 'boot', 'image_id': 'image'})
    row = dict(calendar(TASKS)[0], status='success', valid=True, elapsed_s=2.)
    row.update(change)
    with pytest.raises(ValueError):
        run.record(1, row, started=0, ended=3)
    assert run.rows == [row]
    assert run.decision()['decision'] == 'INCOMPLETE_TERMINAL'
    with pytest.raises(ValueError):
        run.record(2, row, started=4, ended=7)


def test_window_deadline_and_no_go_even_with_zero_errors():
    run = GateRun(TASKS, {'boot_id': 'boot', 'image_id': 'image'})
    for index, slot in enumerate(calendar(TASKS), 1):
        run.record(index, dict(slot, status='success', valid=True, elapsed_s=61.),
                   started=index*65., ended=index*65.+62)
    assert run.decision()['decision'] == 'NO-GO'


def test_schedule_matches_existing_preregistered_windows():
    from scripts.run_study_gate import schedule
    slots = calendar(TASKS)[20:]
    assert [{k: v for k, v in row.items() if k != 'phase'} for row in slots] == schedule(1, TASKS)+schedule(2, TASKS)


def test_driver_uses_call_deadlines_and_requires_final_receipt():
    from scripts.study_operator.i7_gate import collect
    class Host:
        def __init__(self):
            self.t = 0
            self.samples = []
            self.root = None
        def clock(self):
            return self.t
        def poll(self, **kwargs):
            self.samples.append(self.t)
    host = Host()
    class Channel:
        def send(self, request):
            self.request = request
        def receive(self, timeout):
            assert 0 < timeout <= 600
            host.t += 21
            if self.request == {'operation': 'complete'}:
                return {'status': 'GATE_COMPLETE', 'rows_sha256': 'wrong'}
            index = self.request['index']
            return {'status': 'OBSERVATION', 'row': dict(calendar(TASKS)[index-1], index=index,
                status='success', valid=True, elapsed_s=20.)}
    with pytest.raises(ValueError, match='FINAL'):
        collect(Channel(), host, GateRun(TASKS, {'boot_id':'boot','image_id':'image'}))


def test_gate_native_launch_is_bounded_and_not_legacy_cli():
    from scripts.study_operator.i7_gate_host import launch_command
    command = launch_command({'boot_id': 'a'*32, 'config': {'host_code':'/srv/cloudrag/i7/code'}})
    assert '--property=RuntimeMaxSec=9000' in command
    assert 'scripts.study_operator.i7_gate_host' in command
    assert '--property=ExecStopPost=/usr/sbin/shutdown -h now' in command
    assert 'scripts.run_study_gate' not in command


def test_call_and_window_limits_are_independent():
    run = GateRun(TASKS, {'boot_id':'boot','image_id':'image'})
    with pytest.raises(ValueError):
        run.record(1, dict(calendar(TASKS)[0],status='success',valid=True,elapsed_s=601.),
                   started=0,ended=601)
    run = GateRun(TASKS, {'boot_id':'boot','image_id':'image'})
    slots = calendar(TASKS)
    for index in range(1, 22):
        run.record(index,dict(slots[index-1],status='success',valid=True,elapsed_s=1.),
                   started=index*2., ended=index*2.+1)
    with pytest.raises(ValueError):
        run.record(22,dict(slots[21],status='success',valid=True,elapsed_s=1.),
                   started=7300,ended=7301)


def full_host_proof():
    import copy

    from test_study_stimulus_host_evidence import live_proof

    from scripts.study_operator.stimulus_evidence import OPTIONS, software_projection
    from scripts.study_operator.i7_gate_host import gate_admission_receipt
    from scripts.study_operator.stimulus_host_evidence import sha
    source=live_proof()
    host=copy.deepcopy(source['host_evidence'])
    host['samples']=[]
    resident=copy.deepcopy(source['host_evidence']['samples'][-1])
    resident['owned_pids']=[990]
    resident['service_state'].update(sequence=9,runner_pids=[990],written_monotonic_s=99)
    for instant in range(100,161,5):
        sample=copy.deepcopy(resident)
        sample['monotonic_s']=instant
        host['samples'].append(sample)
    host['admission_samples']=host['samples'].copy()
    inventory=source['inventory']
    software=sha(software_projection(inventory))
    identity={'boot_id':host['boot_id'],'image_id':host['app_image_id'],'software_sha256':software}
    events=[dict(kind=kind,instance='synthetic',scope='i7-gate-'+host['boot_id'],**extra) for kind,extra in [
        ('preparation_started',{}),('warmup_response',{'system':'hybrid'}),
        ('pipeline_prepared',{'system':'hybrid'}),('warmup_response',{'system':'no_rag'}),
        ('pipeline_prepared',{'system':'no_rag'}),('preparation_ready',{
            'systems':['hybrid','no_rag'],'status':'ready','identity':{'model_digest':'a'*64}})]]
    host['preparation_sha256']=sha(events)
    host['window_admissions']=[]
    run=GateRun(TASKS,identity)
    offset=0
    for index,slot in enumerate(calendar(TASKS),1):
        if index in (21,81):
            previous=copy.deepcopy(host['samples'][-1])
            rows=[]
            for instant in range(5,66,5):
                sample=copy.deepcopy(previous)
                sample['monotonic_s']+=instant
                rows.append(sample)
            host['samples'].extend(rows)
            host['window_admissions'].append({'index':index,'samples':rows,'inventory_sha256':sha(inventory)})
            offset+=65
        start=161+(index-1)*3+offset
        sample=copy.deepcopy(source['host_evidence']['samples'][-1])
        sample['monotonic_s']=start+2
        sample['owned_pids']=[1000+index]
        sample['service_state'].update(sequence=9+4*index,runner_pids=[1000+index],written_monotonic_s=start+1)
        session_id='synthetic'+str(index)
        backup={'status':'complete','sha256':'a'*64,'objects':{name:{
            'object':'periods/synthetic/'+session_id+'/'+name,'generation':'1','sha256':'a'*64}
            for name in ('full_session.json','export_manifest.json')}}
        row=dict(slot,index=index,status='success',valid=True,elapsed_s=1.,identity=identity,
                 session_id=session_id,backup=backup,
                 service_after=copy.deepcopy(sample['service_state']),generation_options=OPTIONS,
                 request_sha256='a'*64,raw_text='synthetic',answer='synthetic',citations=[])
        run.record(index,row,started=start,ended=start+1)
        host['samples'].append(sample)
    host.update(calls=run.calls,ended=host['samples'][-1]['monotonic_s'],native_limit_s=9000)
    return {'inventory': inventory,'software_sha256': software,'tasks': TASKS,'rows': run.rows,
                'preparation_events':events,
                'host_evidence': host,'host_admission_sha256': sha(gate_admission_receipt(host)),'result': run.decision()}


def test_complete_host_proof_is_recomputed_and_row_policy_cannot_be_rehashed_away():
    from scripts.study_operator.i7_gate_host import verify
    from scripts.study_operator.stimulus_host_evidence import sha
    proof=full_host_proof()
    assert verify(proof)['decision']=='GO'
    proof['rows'][0]['generation_options']={'temperature':1}
    proof['host_evidence']['calls'][0]['row_sha256']=sha(proof['rows'][0])
    with pytest.raises(ValueError):
        verify(proof)
