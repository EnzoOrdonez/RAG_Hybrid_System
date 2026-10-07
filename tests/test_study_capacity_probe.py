import copy
from datetime import datetime,timezone
import json

import pytest

from scripts.study_operator.capacity_probe import Probe,create_args
from scripts.study_operator.policy import OperatorError


def setup(tmp_path,monkeypatch,*,stock=False,lost=False):
    vm = dict(name='cloudrag-study-l4-20261002',id='1',zone='zones/us-central1-a',
        status='TERMINATED',machineType='types/g2-standard-4',deletionProtection=True,
        disks=[dict(autoDelete=False,source='disks/original')],networkInterfaces=[dict(name='nic0')],
        guestAccelerators=[dict(acceleratorType='types/nvidia-l4',acceleratorCount=1)],
        scheduling=dict(instanceTerminationAction='STOP',maxRunDuration=dict(seconds='10800')))
    snapshot = dict(name='qualified',id='2',status='READY',selfLink='snapshots/qualified')
    state = dict(status='ACTIVE',closure_reserved_utc='2026-10-09T23:01:39Z',cloud_cutoff_usd=90,
        cost=dict(estimated_spend_usd=12,reserved_retention_and_closure_usd=4),
        independent_closure_verified=True,resources=[dict(type='vm',name=vm['name'],id='1',zone='us-central1-a',inherited=True)])
    (tmp_path/'STATE.json').write_text(json.dumps(state))
    (tmp_path/'cpu-restoration-bootstrap01-proof.json').write_text(json.dumps(dict(
        source_snapshot_id='2',status='CPU_RESTORATION_VERIFIED',synthetic=False)))
    monkeypatch.setattr('scripts.study_operator.capacity_probe.quote_archive',lambda *args:
                        dict(usd_per_hour='.7',catalog_receipt_sha256='c'*64))
    class Cloud:
        def __init__(self):
            self.calls=[]
            self.vms=[vm]
            self.disks=[]
        def command(self,args,**kwargs):
            self.calls.append(args)
            group,action = args[1:3]
            if group == 'snapshots':
                return copy.deepcopy(snapshot)
            if action == 'list':
                rows = self.vms if group == 'instances' else self.disks
                filters = [a.split('=',2)[-1] for a in args if a.startswith('--filter=name=')]
                return copy.deepcopy([r for r in rows if not filters or r['name'] == filters[0]])
            if action == 'describe':
                return copy.deepcopy(next(r for r in self.vms if r['name'] == args[3]))
            if group == 'disks' and action == 'create':
                marker = next(a.split('=',1)[1] for a in args if a.startswith('--description='))
                zone = next(a.split('=',1)[1] for a in args if a.startswith('--zone='))
                self.disks.append(dict(name=args[3],id=str(20+len(self.disks)),description=marker,
                    zone='zones/'+zone,sourceSnapshotId='2',selfLink='disks/'+args[3],users=[],
                    creationTimestamp='2026-10-07T00:00:00Z'))
                return None
            if action in {'create','start'}:
                if not stock:
                    raise OperatorError('ZONE_RESOURCE_POOL_EXHAUSTED: fixture')
                vm['status']='RUNNING'
                if lost:
                    raise OperatorError('response lost')
            elif action == 'stop':
                vm['status']='TERMINATED'
    cloud = Cloud()
    return Probe(tmp_path,cloud,now=lambda:datetime(2026,10,7,tzinfo=timezone.utc)),cloud,snapshot


def test_exhausted_round_visits_three_zones_with_bound_and_no_public_app(tmp_path,monkeypatch):
    probe,cloud,snapshot = setup(tmp_path,monkeypatch)
    result = probe.central(snapshot,'study-net','study-subnet')
    assert result['status'] == 'COMPLETED' and len(result['results']) == 3
    assert all(r['status'] == 'CAPACITY_EXHAUSTED' and r['stopped_verified'] for r in result['results'])
    state=json.loads((tmp_path/'STATE.json').read_bytes())
    assert state['open_exposures']['capacity-round-1-us-central1-a']['maximum_usd']==0
    assert state['open_exposures']['capacity-round-1-us-central1-b']['maximum_usd']>0  # Keep owned disk exposure.
    assert all(r['compute_upper_released_after_stop_usd']==pytest.approx(2.1)
               for r in state['open_exposures'].values())
    creates = [a for a in cloud.calls if a[1:3] == ['instances','create']]
    assert len(creates) == 2
    for argv in creates:
        assert '--no-address' in argv and '--no-service-account' in argv and '--no-scopes' in argv
        assert '--max-run-duration=3h' in argv and '--instance-termination-action=STOP' in argv
        assert '--deletion-protection' in argv and '--provisioning-model=STANDARD' in argv
        assert any('auto-delete=no' in a for a in argv)
    count = len(cloud.calls)
    with pytest.raises(ValueError,match='45 minutes'):
        probe.central(snapshot,'study-net','study-subnet')
    assert len(cloud.calls) == count
    for path in tmp_path.glob('*startup.sh'):
        assert "'docker','stop'" in path.read_text()
        assert '/usr/sbin/shutdown' in path.read_text() and '+110' in path.read_text()


def test_success_is_stopped_and_explicitly_not_ready_or_gate(tmp_path,monkeypatch):
    probe,cloud,snapshot = setup(tmp_path,monkeypatch,stock=True)
    result = probe.central(snapshot,'net','subnet')
    assert len(result['results']) == 1
    row = result['results'][0]
    assert row['status'] == 'L4_RUNNING_OBSERVED_NOT_READY' and row['not_a_measurement_or_ready']
    assert cloud.vms[0]['status'] == 'TERMINATED'
    assert any(a[2] == 'stop' for a in cloud.calls)


def test_lost_start_response_still_stops_and_preserves_failure(tmp_path,monkeypatch):
    probe,cloud,snapshot = setup(tmp_path,monkeypatch,stock=True,lost=True)
    with pytest.raises(OperatorError,match='response lost'):
        probe.central(snapshot,'net','subnet')
    assert cloud.vms[0]['status'] == 'TERMINATED'
    state = json.loads((tmp_path/'STATE.json').read_bytes())
    assert state['capacity_rounds'][0]['status'] == 'FAILED_PRESERVED_RECONCILIATION_REQUIRED'
    assert json.loads(next(tmp_path.glob('capacity-round-*-receipt.json')).read_bytes())['status'] == 'TOOL_OR_RESOURCE_ERROR_NOT_CAPACITY'


@pytest.mark.parametrize('contaminant',['gpu','ip'])
def test_other_running_gpu_and_original_public_ip_prevent_start(tmp_path,monkeypatch,contaminant):
    probe,cloud,snapshot = setup(tmp_path,monkeypatch)
    if contaminant == 'gpu':
        cloud.vms.append(dict(id='foreign',guestAccelerators=[{}],status='RUNNING'))
    else:
        cloud.vms[0]['networkInterfaces'][0]['accessConfigs'] = [dict(natIP='203.0.113.9')]
    with pytest.raises((OperatorError,ValueError)):
        probe.central(snapshot,'net','subnet')
    assert not any(a[2] in {'start','create'} for a in cloud.calls)


def test_capacity_create_cannot_jump_to_region_without_preparation():
    with pytest.raises(ValueError,match='Central round'):
        create_args('name','disk','us-west1-a','startup','marker','net','subnet')
