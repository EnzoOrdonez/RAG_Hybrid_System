from datetime import datetime, timezone
import json

import pytest

from scripts.study_operator.lifecycle import Operator
from scripts.study_operator.policy import OperatorError


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


def test_start_idempotent_scope_budget_and_terminal_metadata(tmp_path):
    operator,cloud = installation(tmp_path)
    assert operator.start('smoke')['status'] == 'STARTED_SUPERVISED'
    script = (cloud.root/'startup.sh').read_text()
    assert 'stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL' in script
    assert 'host_runtime' in script and '/srv/cloudrag/iteration4' in script
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


def test_stop_idempotent_and_no_invitation_in_receipts(tmp_path):
    operator,cloud = installation(tmp_path)
    assert operator.stop()['status'] == 'TERMINATED_VERIFIED'
    assert not any(args[:3] == ['compute','instances','stop'] for args,_ in cloud.calls)
    operator.preflight = lambda:dict(status='READY_VERIFIED')
    operator.state['purpose'] = 'smoke'
    requests = []
    operator.bridge = lambda request,**options: requests.append((request,options)) or {}
    token = operator.invite('P999',cell=1,profile='without_experience')
    assert len(token) >= 32 and 'token_sha256' in requests[0][0]
    assert token not in json.dumps(requests) and requests[0][1]['private']
    assert operator.state['failover_data_reconciled'] is False


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


def test_tls_preparation_and_capacity_fail_closed_before_start(tmp_path):
    operator,cloud = installation(tmp_path)
    with pytest.raises(OperatorError,match='3 días'):
        operator.tls_prepare('2026-10-07')
    with pytest.raises(OperatorError,match='Fecha inválida'):
        operator.tls_prepare('unscheduled')
    with pytest.raises(OperatorError,match='Zona alterna'):
        operator.failover('us-east1-b')
    with pytest.raises(OperatorError,match='instantánea preparada'):
        operator.failover('us-central1-b')
    assert not cloud.calls


def test_ip_release_idempotent_and_no_tls_idle_charge(tmp_path):
    operator,cloud = installation(tmp_path)
    assert operator.ip_release()['status'] == 'ALREADY_RELEASED'
    assert not any(args[:3] == ['compute','addresses','delete'] for args,_ in cloud.calls)
