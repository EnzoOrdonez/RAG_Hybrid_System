from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from scripts.study_operator import stimulus_guest as guest
from scripts.study_operator.stimulus_host import HostCollection, launch_command
from scripts.study_operator.stimulus_host_evidence import sha


def setup(tmp_path,monkeypatch):
    from test_study_operator_deployment import config

    boot = '12345678-aaaa-bbbb-cccc-123456789abc'
    root = tmp_path/'boot'
    (root/'sockets').mkdir(parents=True)
    (root/'sockets'/'service-state.json').write_text(json.dumps(dict(schema_version=1,mode='fresh_runner',
        phase='STARTING',sequence=1,boot_id=boot,request_id=None,deadline_monotonic_s=None)))
    now = datetime(2026,10,8,tzinfo=timezone.utc)
    cfg = config()
    cfg.update(purpose='technical',host_code='/srv/cloudrag/iteration5/code')
    active = dict(boot_id=boot,boot_root=str(root),session_root=str(tmp_path/'sessions'),config=cfg,
        app_container='owned-app',ollama_container='owned-ollama',
        guest_deadline_utc=(now+timedelta(minutes=130)).isoformat(),
        native_deadline_utc=(now+timedelta(minutes=140)).isoformat())
    monkeypatch.setattr(guest.os,'access',lambda *args: True)
    monkeypatch.setattr(guest,'recovery_counts',lambda *args: dict(session_count=0,invitation_count=0))
    calls = []
    def execute(argv,**kwargs):
        calls.append(argv)
        return {}
    kwargs = dict(execute=execute,controller=lambda *args: dict(session_count=0,invitation_count=0,active_session=False),
                  maintenance=lambda: calls.append('maintenance'),now=lambda: now)
    return active,calls,kwargs


def test_fixed_unit_is_validated_before_launch_and_cannot_repeat(tmp_path,monkeypatch):
    active,calls,kwargs = setup(tmp_path,monkeypatch)
    result = guest.dispatch_stimulus(active,dict(operation='stimulus-start',boot_index=5),**kwargs)
    assert result['native_limit_s'] == 7200 and result['replay_allowed'] is False
    assert calls[0][:2] == ['systemd-analyze','verify'] and calls[1] == 'maintenance'
    command = calls[2]
    assert command == launch_command(active,5)
    assert '--property=StandardOutput=null' in command and '--property=RuntimeMaxSec=7200' in command
    assert '--property=ExecStopPost=/usr/sbin/shutdown -h now' in command
    assert '--working-directory=/srv/cloudrag/iteration5/code' in command
    with pytest.raises(ValueError,match='ALREADY_REGISTERED'):
        guest.dispatch_stimulus(active,dict(operation='stimulus-start',boot_index=5),**kwargs)
    assert len(calls) == 3


@pytest.mark.parametrize('defect',['study','warm','margin','index','options','binary','data'])
def test_unsafe_job_is_rejected_before_any_host_effect(tmp_path,monkeypatch,defect):
    active,calls,kwargs = setup(tmp_path,monkeypatch)
    request = dict(operation='stimulus-start',boot_index=5)
    if defect == 'study':
        active['config']['purpose'] = 'study'
    elif defect == 'warm':
        path = tmp_path/'boot'/'sockets'/'service-state.json'
        state = json.loads(path.read_bytes())
        state['sequence'] = 5
        path.write_text(json.dumps(state))
    elif defect == 'margin':
        active['guest_deadline_utc'] = (kwargs['now']()+timedelta(minutes=124)).isoformat()
    elif defect == 'index':
        request['boot_index'] = True
    elif defect == 'options':
        request['num_predict'] = 512
    elif defect == 'binary':
        monkeypatch.setattr(guest.os,'access',lambda *args: False)
    else:
        kwargs['controller'] = lambda *args: dict(session_count=1,invitation_count=0,active_session=True)
    with pytest.raises(ValueError):
        guest.dispatch_stimulus(active,request,**kwargs)
    assert calls == [] and not (tmp_path/'boot'/'stimulus').exists()


def test_failed_syntax_or_lost_launch_remains_terminal(tmp_path,monkeypatch):
    active,calls,kwargs = setup(tmp_path,monkeypatch)
    def failure(argv,**settings):
        calls.append(argv)
        raise ValueError('tool failure')
    kwargs['execute'] = failure
    with pytest.raises(ValueError):
        guest.dispatch_stimulus(active,dict(operation='stimulus-start',boot_index=5),**kwargs)
    with pytest.raises(ValueError,match='ALREADY_REGISTERED'):
        guest.dispatch_stimulus(active,dict(operation='stimulus-start',boot_index=5),**kwargs)
    assert len(calls) == 1


def test_status_is_technical_and_evidence_uses_private_owner_transport(tmp_path,monkeypatch):
    active,_,kwargs = setup(tmp_path,monkeypatch)
    root = tmp_path/'boot'/'stimulus'
    (root/'coded-P999').mkdir(parents=True)
    proof = dict(synthetic=True,raw_text='synthetic coded content')
    (root/'coded-P999'/'complete.json').write_text(json.dumps(proof))
    (root/'result.json').write_text(json.dumps(dict(status='BOOT_COMPLETE_UNANALYZED',proof_sha256=sha(proof))))
    status = guest.dispatch_stimulus(active,dict(operation='stimulus-status'),**kwargs)
    assert proof['raw_text'] not in json.dumps(status)
    assert guest.dispatch_stimulus(active,dict(operation='stimulus-evidence'),**kwargs)['proof'] == proof
    acknowledged = guest.dispatch_stimulus(active,dict(operation='stimulus-ack',proof_sha256=sha(proof)),**kwargs)
    assert acknowledged['status'] == 'STIMULUS_DOWNLOAD_ACKNOWLEDGED'
    with pytest.raises(ValueError,match='HASH_DIFFERS'):
        guest.dispatch_stimulus(active,dict(operation='stimulus-ack',proof_sha256='f'*64),**kwargs)
    (root/'coded-P999'/'complete.json').write_text('{}')
    with pytest.raises(ValueError,match='CHANGED'):
        guest.dispatch_stimulus(active,dict(operation='stimulus-evidence'),**kwargs)


def test_poll_uses_exact_host_owned_pids_and_detects_other_container(tmp_path,monkeypatch):
    active,_,_ = setup(tmp_path,monkeypatch)
    from test_study_cold_admission import rows
    row = rows()[0]
    row['service_state']['boot_id'] = active['boot_id']
    containers = {active['ollama_container']:dict(Id='b'*64,State=dict(Running=True),Config=dict(Image=active['config']['ollama_image']))}
    def invoke(argv,**kwargs):
        if argv[:2] == ['docker','inspect']:
            result = json.dumps([containers[argv[2]]]).encode()
        elif argv[:2] == ['docker','top']:
            result = b'PID\n100\n'
        else:
            result = b'100\n'
        return SimpleNamespace(returncode=0,stdout=result)
    root = tmp_path/'job'
    (root/'telemetry').mkdir(parents=True)
    collector = HostCollection(active,5,root,invoke=invoke,sampler=lambda: row,clock=lambda:100)
    containers[collector.name] = dict(Id='a'*64,Image=active['config']['image_id'],State=dict(Running=True))
    collector.host.update(own_container_id='a'*64,ollama_container_id='b'*64)
    collector.poll(force=True)
    assert row['gpu_pids'] == [100] and 100 in row['owned_pids']
    containers[collector.name]['Id'] = 'c'*64
    with pytest.raises(ValueError,match='IDENTITY_CHANGED'):
        collector.poll(force=True)


def test_complete_host_flow_admits_before_first_query_and_preserves_private_proof(tmp_path,monkeypatch):
    import copy
    from test_study_cold_admission import rows
    from test_study_stimulus_evidence import CONFIG
    from test_study_stimulus_host_evidence import live_proof
    from scripts.study_operator import stimulus_host as module
    from scripts.study_operator.stimulus_evidence import verify_boot

    active,_,_ = setup(tmp_path,monkeypatch)
    proof = live_proof()
    active['boot_id'] = proof['inventory']['observed']['boot_id']
    # Product UUID/name validation remains tested separately by owned_names.
    monkeypatch.setattr(module,'owned_names',lambda active,index: ('owned-stimulus','owned-unit'))
    active['config'].update(commit=proof['inventory']['source']['commit'],image_id=proof['inventory']['image']['image_id'],model_digest='a'*64)
    proof.pop('host_evidence')
    proof.pop('host_admission_sha256')
    moment = [100.0]
    generated = [False]
    stopped = []
    process = SimpleNamespace(returncode=None)
    process.poll = lambda: process.returncode
    process.terminate = lambda: setattr(process,'returncode',0)
    process.wait = lambda **kwargs: process.returncode
    def clock():
        moment[0] += .01
        return moment[0]
    def sleep(seconds):
        moment[0] += seconds
    def invoke(argv,**kwargs):
        if argv[:2] == ['docker','inspect']:
            observed = dict(Id=('b' if argv[2]=='owned-ollama' else 'a')*64,
                Image=active['config']['image_id'],State=dict(Running=argv[2]!='owned-app'),
                Config=dict(Image=active['config']['ollama_image']))
            content = json.dumps([observed]).encode()
        elif argv[:2] == ['docker','top']:
            content = b'PID\n100\n101\n'
        elif argv[:2] == ['docker','stop']:
            stopped.append(argv[4])
            process.returncode = 0
            content = b''
        else:
            content = b'101\n' if generated[0] else b''
        return SimpleNamespace(returncode=0,stdout=content)
    def sampler():
        row = copy.deepcopy(rows()[0])
        row['monotonic_s'] = clock()
        row['service_state']['boot_id'] = active['boot_id']
        if generated[0]:
            row['service_state'].update(proof['rows'][0]['service_after'],request_id='b'*32,
                                        written_monotonic_s=moment[0]-.01)
            row['ollama_ps_api']['models'] = [dict(digest='a'*64,context_length=4096,expires_at='2100-01-01T00:00:00+00:00')]
        return row
    class Channel:
        def __init__(self,process,stop,poll):
            self.stop,self.poll = stop,poll
            self.sent = []
            self.responses = 0
        def send(self,request):
            self.sent.append(request)
            if 'index' in request:
                assert self.sent[0]['operation'] == 'admit'
                generated[0] = True
        def receive(self,seconds):
            self.poll()
            self.responses += 1
            if self.responses == 1:
                return dict(status='PREPARED_NOT_ADMITTED',inventory=proof['inventory'],slots=1)
            if self.responses == 2:
                return dict(status='HOST_ADMISSION_ACKNOWLEDGED',receipt_sha256=self.sent[0]['receipt_sha256'])
            if self.responses == 3:
                moment[0] += 1
                return dict(status='OBSERVATION',row=proof['rows'][0])
            proof['host_admission_sha256'] = self.sent[0]['receipt_sha256']
            return dict(status='COMPLETE_NOT_ACCEPTANCE',proof=proof)
        def close(self):
            self.stop()
    monkeypatch.setattr(module,'PrivatePipe',Channel)
    monkeypatch.setattr(module,'assert_isolation',lambda *args: {})
    monkeypatch.setattr(module,'metadata_unreachable',lambda *args,**kwargs: dict(metadata_unreachable=True))
    (tmp_path/'boot'/'maintenance.json').write_text('{}')
    root = tmp_path/'boot'/'stimulus'
    root.mkdir()
    launched = []
    def launch(argv,**kwargs):
        launched.append(argv)
        return process
    collector = HostCollection(active,5,root,invoke=invoke,launch=launch,sampler=sampler,clock=clock,sleep=sleep)
    result = collector.run()
    assert verify_boot(result,CONFIG)['mode'] == 'LIVE'  # Fixture verifies binding, no real acceptance.
    assert json.loads((root/'result.json').read_bytes())['acceptance_not_inferred'] is True
    assert json.loads((root/'coded-P999'/'complete.json').read_bytes()) == result
    assert '/web/stimulus.sock' in launched[0] and '--rm=false' in launched[0]
    assert stopped == ['owned-stimulus'] and not (tmp_path/'boot'/'stop-request.json').exists()
