from datetime import datetime, timezone
import json

import pytest

from scripts.study_operator.lifecycle import Operator
from scripts.study_operator.policy import OperatorError, ReadyPending


class Cloud:
    project = 'pure-loop-474323-a8'

    def __init__(self,root,vm):
        self.root,self.vm,self.calls = root,vm,[]

    def command(self,args,**options):
        self.calls.append((args,options))
        if args[:3] == ['compute','instances','describe']:
            return self.vm
        if args[:3] == ['compute','instances','list']:
            return [self.vm]
        if args[:3] == ['compute','addresses','list']:
            return []
        if args[:3] == ['compute','instances','start']:
            self.vm['status'] = 'RUNNING'
        if args[:3] == ['compute','instances','stop']:
            self.vm['status'] = 'TERMINATED'
        return None


def installation(tmp_path):
    root = tmp_path/'operator'
    root.mkdir()
    value = dict(schema_version=1, project='pure-loop-474323-a8', zone='us-central1-a', machine_type='g2-standard-4',
        sessions_bucket='cloudrag-study-i4-103950017681-20261004',technical_bucket='cloudrag-study-103950017681-20261002',
        purpose='smoke',period_id='a'*32,image_id='sha256:'+'b'*64,commit='c'*40,model_digest='d'*64,
        static_ip='203.0.113.8',hostname='203.0.113.8.sslip.io',ollama_image='ollama@sha256:'+'e'*64,
        caddy_image='caddy@sha256:'+'f'*64,asset_root='/srv/cloudrag/assets',ollama_models='/srv/cloudrag/models',
        host_code='/srv/cloudrag/iteration4/code',primary_vm=dict(name='original',id='123',zone='us-central1-a'),
        official_rates=dict(compute_usd_h=.706832276),ip_name='owned-address',
        cost=dict(estimated_usd=8,margin_usd=3))
    (root/'installation.json').write_text(json.dumps(value))
    vm = dict(name='original',id='123',zone='projects/p/zones/us-central1-a',machineType='zones/z/machineTypes/g2-standard-4',
        deletionProtection=True,disks=[dict(autoDelete=False)],status='TERMINATED',
        scheduling=dict(instanceTerminationAction='STOP',maxRunDuration=dict(seconds='10800')),
        networkInterfaces=[dict(name='nic0',accessConfigs=[dict(name='External NAT',natIP='203.0.113.8')])])
    cloud = Cloud(tmp_path/'receipts',vm)
    cloud.root.mkdir()
    return Operator(root,cloud,now=lambda:datetime(2026,10,5,tzinfo=timezone.utc)),cloud


def test_missing_installation_has_human_action(tmp_path):
    with pytest.raises(OperatorError,match='installation.json'):
        Operator(tmp_path,None)


def test_stopped_diagnostics_downloads_only_exact_technical_generation(tmp_path,monkeypatch):
    import base64
    import hashlib
    from scripts.study_operator.bootstrap_failure import failure_summary
    from scripts.study_operator.gcs import Storage

    operator,cloud = installation(tmp_path)
    cloud.owner_token = lambda:'fixture-token-only'
    boot = tmp_path/'11111111-1111-4111-8111-111111111111'
    boot.mkdir()
    (boot/'failure.json').write_text('{"reason":"HOST_COMMAND_FAILED","error_type":"ValueError"}')
    summary = failure_summary(boot,instance_id='123',image_id=operator.config['image_id'],commit=operator.config['commit'])
    content = json.dumps(summary).encode()
    name = 'iteration4/failed-boots/123/'+boot.name+'.json'
    row = dict(name=name,generation='42',timeCreated='2026-10-05T00:00:00Z',size=len(content),
        md5Hash=base64.b64encode(hashlib.md5(content).digest()).decode())
    monkeypatch.setattr(Storage,'objects',lambda self,prefix:[row] if prefix=='iteration4/failed-boots/123/' else [])
    reads=[]
    def read(self,name,generation):
        reads.append((name,generation))
        return content
    monkeypatch.setattr(Storage,'read',read)
    result=operator.diagnostics()
    assert reads==[(name,'42')]
    assert result['provenance']['object_sha256']==hashlib.sha256(content).hexdigest()
    assert result['session_content_excluded'] and result['failure']['reason']=='HOST_COMMAND_FAILED'
    assert not any(args[:3]==['compute','instances','start'] for args,_ in cloud.calls)
    row['md5Hash']='tampered'
    with pytest.raises(OperatorError,match='alterado'):
        operator.diagnostics()


def test_stopped_diagnostics_without_archive_has_disk_recovery_action(tmp_path,monkeypatch):
    from scripts.study_operator.gcs import Storage
    operator,cloud=installation(tmp_path)
    cloud.owner_token=lambda:'fixture-token-only'
    monkeypatch.setattr(Storage,'objects',lambda self,prefix:[])
    with pytest.raises(OperatorError,match='recuperación técnica del disco'):
        operator.diagnostics()


def test_start_idempotent_scope_budget_and_terminal_metadata(tmp_path):
    operator,cloud = installation(tmp_path)
    assert operator.start('smoke')['status'] == 'STARTED_SUPERVISED'
    script = (cloud.root/'startup.sh').read_text()
    assert 'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL' in script
    assert 'host_runtime' in script and '/srv/cloudrag/iteration5' in script
    assert operator.start('smoke')['status'] == 'ALREADY_RUNNING'
    assert sum(args[:3] == ['compute','instances','start'] for args,_ in cloud.calls) == 1
    assert operator.state['cost']['reservations']


def test_ethics_and_budget_reject_before_paid_actions(tmp_path):
    operator,cloud = installation(tmp_path)
    with pytest.raises(OperatorError,match='ética'):
        operator.start('study')
    assert not cloud.calls
    operator.config['cost']['estimated_usd'] = 89
    with pytest.raises(OperatorError,match='USD90'):
        operator.start('smoke')
    assert not any(args[:3] == ['compute','instances','start'] for args,_ in cloud.calls)


def test_start_rejects_other_gpu_and_changed_identity(tmp_path):
    operator,cloud = installation(tmp_path)
    original = cloud.command
    def command(args,**options):
        if args[:3] == ['compute','instances','list']:
            return [dict(id='other',guestAccelerators=[{}],status='RUNNING')]
        return original(args,**options)
    cloud.command = command
    with pytest.raises(OperatorError,match='otra VM'):
        operator.start('smoke')
    cloud.vm['id'] = 'changed'
    with pytest.raises(OperatorError,match='VM, disco'):
        operator.start('smoke')


def test_stop_idempotent_and_no_invitation_in_receipts(tmp_path,monkeypatch):
    operator,cloud = installation(tmp_path)
    assert operator.stop()['status'] == 'TERMINATED_VERIFIED'
    assert not any(args[:3] == ['compute','instances','stop'] for args,_ in cloud.calls)
    operator.preflight = lambda:dict(status='READY_VERIFIED')
    cloud.owner_token = lambda:'synthetic-fixture-no-network'
    monkeypatch.setattr('scripts.study_operator.gcs.Storage',lambda *args,**options:
                        type('Empty',(),{'objects':lambda self,prefix:[]})())
    operator.state['purpose'] = 'smoke'
    requests = []
    operator.bridge = lambda request,**options: requests.append((request,options)) or {}
    token = operator.invite('P999',cell=1,profile='without_experience')
    assert len(token) >= 32 and 'token_sha256' in requests[0][0]
    assert token not in json.dumps(requests) and requests[0][1]['private']
    assert operator.state['failover_data_reconciled'] is False


def test_archived_code_cannot_be_reinvited_from_empty_disk(tmp_path,monkeypatch):
    operator,cloud = installation(tmp_path)
    operator.state['purpose'] = 'smoke'
    operator.preflight = lambda:dict(status='READY_VERIFIED')
    cloud.owner_token = lambda:'synthetic-fixture-no-network'
    monkeypatch.setattr('scripts.study_operator.gcs.Storage',lambda *args,**options:
                        type('Backed',(),{'objects':lambda self,prefix:[{'name':prefix+'full_session.json'}]})())
    operator.bridge = lambda *args,**options:pytest.fail('No new invitation hash may reach the guest')
    with pytest.raises(OperatorError,match='ya tiene una sesión'):
        operator.invite('P999',cell=1,profile='without_experience')


def test_invitation_missing_synthetic_assignment_rejected_before_cloud_preflight(tmp_path):
    operator,cloud = installation(tmp_path)
    operator.state['purpose'] = 'smoke'
    with pytest.raises(OperatorError,match='invitación sintética'):
        operator.invite('P999')
    assert not cloud.calls


def test_purpose_change_gets_separate_persistent_period(tmp_path):
    operator,_ = installation(tmp_path)
    operator.start('technical')
    assert operator.config['period_id'] != 'a'*32
    assert operator.config['period_ids']['technical'] == operator.config['period_id']
    assert operator.state['period_origin']['status'] == 'NEW_UNINVITED_PERIOD'
    assert operator.state['failover_data_reconciled']


def test_compute_and_static_ip_reservations_are_not_charged_twice(tmp_path):
    operator,_ = installation(tmp_path)
    operator.state['cost'] = dict(estimated_usd=0,margin_usd=0,reservations={'ip-own':.72})
    operator.start('smoke')
    operator.bridge = lambda *args,**options:dict(failover_data_reconciled=True)
    operator.now = lambda:datetime(2026,10,5,1,tzinfo=timezone.utc)
    operator.stop()
    assert operator.state['cost']['estimated_usd'] == pytest.approx(.706832276)
    assert operator.state['cost']['reservations'] == {'ip-own':.72}
    assert operator.state['cost']['margin_usd'] == .25


def test_elapsed_retention_blocks_new_paid_effect_before_budget_overrun(tmp_path):
    operator,cloud = installation(tmp_path)
    operator.config['cost'].update(estimated_usd=89,margin_usd=0,
        as_of_utc='2026-10-01T00:00:00+00:00',retention_usd_day=.3287664)
    with pytest.raises(OperatorError,match='USD90'):
        operator.start('smoke')
    assert not any(args[:3] == ['compute','instances','start'] for args,_ in cloud.calls)


def test_ip_creation_response_loss_remains_recoverable_and_rejects_foreign_address(tmp_path):
    operator,cloud = installation(tmp_path)
    live = []
    original = cloud.command
    def command(args,**options):
        if args[:3] == ['compute','addresses','list']:
            return live
        if args[:3] == ['compute','addresses','create']:
            assert '--region=us-central1' in args
            assert not any(arg.startswith('--ip-version=') for arg in args)
            marker = next(arg.removeprefix('--description=') for arg in args if arg.startswith('--description='))
            live.append(dict(name='owned-address',id='42',region='regions/us-central1',addressType='EXTERNAL',
                address='203.0.113.8',description=marker,creationTimestamp='2026-10-05T00:00:00+00:00'))
            raise OperatorError('response lost')
        if args[:3] == ['compute','addresses','describe']:
            return live[0]
        return original(args,**options)
    cloud.command = command
    with pytest.raises(OperatorError,match='response lost'):
        operator.ip_reserve()
    persisted = json.loads(operator.state_path.read_text())
    assert persisted['ip_creation_intent']['ownership_marker'] == live[0]['description']
    assert operator.ip_reserve()['address_id'] == '42'
    assert 'ip_creation_intent' not in operator.state
    live[0]['id'] = '43'
    with pytest.raises(OperatorError,match='creación propio'):
        operator.ip_reserve()


def test_rejected_regional_ip_create_preserves_one_intent_and_budget_reservation(tmp_path):
    operator,cloud = installation(tmp_path)
    original = cloud.command
    def command(args,**options):
        if args[:3] == ['compute','addresses','list']:
            return []
        if args[:3] == ['compute','addresses','create']:
            assert '--region=us-central1' in args
            assert not any(arg.startswith('--ip-version=') for arg in args)
            raise OperatorError('creation rejected')
        return original(args,**options)
    cloud.command = command
    with pytest.raises(OperatorError,match='creation rejected'):
        operator.ip_reserve()
    intent = dict(operator.state['ip_creation_intent'])
    reservations = dict(operator.state['cost']['reservations'])
    operator.now = lambda:datetime(2026,10,5,1,tzinfo=timezone.utc)
    with pytest.raises(OperatorError,match='creation rejected'):
        operator.ip_reserve()
    assert operator.state['ip_creation_intent'] == intent
    assert operator.state['cost']['reservations'] == reservations


def test_never_associated_ip_is_charged_at_unused_rate(tmp_path):
    operator,cloud = installation(tmp_path)
    address = dict(name='owned-address',id='42',address='203.0.113.8',region='regions/us-central1')
    original = cloud.command
    live = [address]
    def command(args,**options):
        if args[:3] == ['compute','addresses','list']:
            return live
        if args[:3] == ['compute','addresses','delete']:
            live.clear()
        return original(args,**options)
    cloud.command = command
    operator.state.update(reserved_address_id='42',ip_reserved_utc='2026-10-05T00:00:00+00:00',
        cost=dict(estimated_usd=0,margin_usd=0,reservations={'ip-own':.72}))
    operator.now = lambda:datetime(2026,10,5,1,tzinfo=timezone.utc)
    operator.ip_release()
    assert operator.state['cost']['estimated_usd'] == pytest.approx(.01)


def test_ip_delete_ack_without_api_absence_never_settles(tmp_path):
    operator,cloud = installation(tmp_path)
    original = cloud.command
    address = dict(name='owned-address',id='42',address='203.0.113.8',region='regions/us-central1')
    cloud.command = lambda args,**options: [address] if args[:3] == ['compute','addresses','list'] else original(args,**options)
    operator.state.update(reserved_address_id='42',ip_reserved_utc='2026-10-05T00:00:00+00:00',
        cost=dict(estimated_usd=0,margin_usd=0,reservations={'ip-own':.72}))
    operator.now = lambda:datetime(2026,10,5,1,tzinfo=timezone.utc)
    with pytest.raises(OperatorError,match='sigue en la API'):
        operator.ip_release()
    assert operator.state['reserved_address_id'] == '42'
    assert operator.state['cost']['estimated_usd'] == 0


def test_tls_preparation_and_capacity_fail_closed_before_start(tmp_path):
    operator,cloud = installation(tmp_path)
    with pytest.raises(OperatorError,match='3 días'):
        operator.tls_prepare('2026-10-07')
    with pytest.raises(OperatorError,match='Fecha inválida'):
        operator.tls_prepare('unscheduled')
    with pytest.raises(OperatorError,match='otra región'):
        operator.failover('us-east1-b')
    with pytest.raises(OperatorError,match='instantánea preparada'):
        operator.failover('us-central1-b')
    assert not cloud.calls


def test_tls_waits_only_for_explicit_bootstrap_and_stops_on_fatal_error(tmp_path,monkeypatch):
    operator, _ = installation(tmp_path)
    calls = []
    operator.start = lambda purpose: calls.append('start')
    operator.stop = lambda: calls.append('stop')
    operator.sleep = lambda seconds: calls.append('wait')
    operator.maintenance = lambda: calls.append('maintenance')
    operator.bridge = lambda request:dict(status='ALL_I4_PERIODS_EMPTY')
    operator.state['ready'] = dict(tls=dict(certificate_sha256='d'*64))
    monkeypatch.setattr('scripts.study_operator.prepared_snapshot.prepare',lambda *args:dict(status='READY'))
    results = iter([ReadyPending('booting'), {'status': 'READY_VERIFIED'}])
    def preflight():
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result
    operator.preflight = preflight
    assert operator.tls_prepare('2026-10-08')['status'] == 'READY_VERIFIED'
    assert calls == ['start', 'wait', 'maintenance', 'stop']
    calls.clear()
    operator.preflight = lambda: (_ for _ in ()).throw(OperatorError('IAM rejected'))
    with pytest.raises(OperatorError, match='IAM rejected'):
        operator.tls_prepare('2026-10-08')
    assert calls == ['start', 'stop']


def test_ip_release_idempotent_and_no_tls_idle_charge(tmp_path):
    operator,cloud = installation(tmp_path)
    assert operator.ip_release()['status'] == 'ALREADY_RELEASED'
    assert not any(args[:3] == ['compute','addresses','delete'] for args,_ in cloud.calls)


def test_foreign_region_ip_rejected_before_vm_detach_or_delete(tmp_path):
    operator,cloud = installation(tmp_path)
    original = cloud.command
    address = dict(name='owned-address',id='42',region='regions/us-east1',address='203.0.113.8')
    cloud.command = lambda args,**options: [address] if args[:3] == ['compute','addresses','list'] else original(args,**options)
    operator.state['reserved_address_id'] = '42'
    with pytest.raises(OperatorError,match='otra región'):
        operator.ip_release()
    assert not cloud.calls  # Even VM describe/detach is forbidden after the mismatch.


def test_regional_ip_reservation_uses_selected_us_region(tmp_path):
    operator,cloud = installation(tmp_path)
    operator.config['primary_vm']['zone'] = 'us-west4-a'
    operator.config['zone'] = 'us-west4-a'
    cloud.vm['zone'] = 'zones/us-west4-a'
    original = cloud.command
    addresses = []
    def command(args,**options):
        cloud.calls.append((args,options))
        if args[:3] == ['compute','addresses','list']:
            return addresses
        if args[:3] == ['compute','addresses','create']:
            assert '--region=us-west4' in args
            marker = next(arg.split('=',1)[1] for arg in args if arg.startswith('--description='))
            addresses.append(dict(name='owned-address',id='42',region='regions/us-west4',
                addressType='EXTERNAL',address='203.0.113.8',description=marker,
                creationTimestamp='2026-10-05T00:00:00+00:00'))
            return None
        if args[:3] == ['compute','addresses','describe']:
            assert '--region=us-west4' in args
            return addresses[0]
        return original(args,**options)
    cloud.command = command
    assert operator.ip_reserve()['address_id'] == '42'
    assert operator.state['ip_associated_utc']
    assert not any('--region=us-central1' in args for args,_ in cloud.calls)
    assert operator.failover('us-west4-a')['status'] == 'ALREADY_SELECTED'
    with pytest.raises(OperatorError,match='instantánea preparada'):
        operator.failover('us-west4-c')
    with pytest.raises(OperatorError,match='otra región'):
        operator.failover('us-central1-b')


def test_absent_owned_ip_settles_once_and_invalidates_live_tls_state(tmp_path):
    operator,cloud = installation(tmp_path)
    operator.state.update(reserved_address_id='42',
        ip_reserved_utc='2026-10-03T00:00:00+00:00',
        ip_associated_utc='2026-10-03T00:00:00+00:00',
        ready={'status':'READY_VERIFIED'},ready_verified=True,snapshots=[{'id':'preserved-fixture'}],
        cost=dict(estimated_usd=0,margin_usd=0,reservations={'ip-own':.72,'other':.25}))
    operator.config['prepared_snapshot']={'id':'preserved-fixture'}
    assert operator.ip_release()['status']=='ALREADY_RELEASED'
    assert 'reserved_address_id' not in operator.state
    assert 'ip_reserved_utc' not in operator.state and 'ip_associated_utc' not in operator.state
    assert 'static_ip' not in operator.config and 'hostname' not in operator.config
    assert 'prepared_snapshot' not in operator.config and 'ready' not in operator.state
    assert operator.state['ready_verified'] is False
    assert operator.state['snapshots']==[{'id':'preserved-fixture'}]
    assert operator.state['cost']['estimated_usd']==pytest.approx(.48)
    assert operator.state['cost']['reservations']=={'other':.25}
    receipt=json.loads((cloud.root/'ip-release-settlement.json').read_text())
    assert receipt['estimation_method']=='UNUSED_RATE_UPPER_UNTIL_OBSERVED_RELEASE'
    assert receipt['address_id']=='42' and receipt['snapshot_resources_preserved']
    reloaded=Operator(operator.root,cloud,now=operator.now)
    assert reloaded.ip_release()['status']=='ALREADY_RELEASED'
    assert reloaded.state['cost']['estimated_usd']==pytest.approx(.48)
    assert reloaded.state['ready_verified'] is False
    assert not any(args[:3]==['compute','instances','describe'] for args,_ in cloud.calls)


def test_lost_ip_delete_response_can_reconcile_then_reserve_new_owned_id(tmp_path):
    operator,cloud = installation(tmp_path)
    live=[dict(name='owned-address',id='42',address='203.0.113.8',region='regions/us-central1')]
    original=cloud.command
    def command(args,**options):
        cloud.calls.append((args,options))
        if args[:3]==['compute','addresses','list']:
            return live
        if args[:3]==['compute','addresses','delete']:
            live.clear()
            raise OperatorError('response lost')
        if args[:3]==['compute','addresses','create']:
            marker=next(arg.removeprefix('--description=') for arg in args if arg.startswith('--description='))
            live.append(dict(name='owned-address',id='43',region='regions/us-central1',addressType='EXTERNAL',
                address='203.0.113.9',description=marker,creationTimestamp='2026-10-05T00:00:00+00:00'))
            return None
        if args[:3]==['compute','addresses','describe']:
            return live[0]
        return original(args,**options)
    cloud.command=command
    operator.state.update(reserved_address_id='42',ip_reserved_utc='2026-10-04T00:00:00+00:00',
        cost=dict(estimated_usd=0,margin_usd=0,reservations={'ip-old':.72}))
    with pytest.raises(OperatorError,match='response lost'):
        operator.ip_release()
    assert operator.state['reserved_address_id']=='42'
    assert operator.ip_release()['status']=='ALREADY_RELEASED'
    assert operator.state['cost']['estimated_usd']==pytest.approx(.24)
    assert operator.ip_reserve()['address_id']=='43'
    assert operator.config['hostname']=='203.0.113.9.sslip.io'
    assert sum(args[:3]==['compute','addresses','delete'] for args,_ in cloud.calls)==1


def test_absent_owned_ip_without_reservation_time_blocks_new_paid_effect(tmp_path):
    operator,cloud = installation(tmp_path)
    operator.state['reserved_address_id']='42'
    with pytest.raises(OperatorError,match='fecha.*IP'):
        operator.ip_reserve()
    assert not any(args[:3]==['compute','addresses','create'] for args,_ in cloud.calls)
    assert operator.state['reserved_address_id']=='42'


def test_unknown_ip_detachment_uses_unused_rate_upper_before_paid_effect(tmp_path):
    operator,cloud = installation(tmp_path)
    operator.config['cost']=dict(estimated_usd=88.8,margin_usd=0)
    operator.state.update(reserved_address_id='42',ip_reserved_utc='2026-09-30T00:00:00+00:00',
        ip_associated_utc='2026-09-30T00:00:00+00:00',
        cost=dict(estimated_usd=0,margin_usd=0,reservations={'ip-old':.72}))
    with pytest.raises(OperatorError,match='USD90'):
        operator.reserve_cost('fixture-paid-effect',.1)
    assert not cloud.calls
