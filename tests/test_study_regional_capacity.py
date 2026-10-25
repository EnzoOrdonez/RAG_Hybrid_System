import copy
from datetime import datetime,timezone
import hashlib
import json

import pytest

from scripts.study_operator.regional_capacity import RegionalProbe,admission_order,regional_create


def inputs():
    catalog=[dict(name='nvidia-l4',zone='zones/'+z) for z in
        ('us-central1-a','us-central1-b','us-central1-c','us-west1-a','us-west1-b','us-east4-a')]
    results=[dict(region=r,zones=z,median_ms=ms,samples=[dict(status='OK',ms=ms) for _ in range(5)])
        for r,z,ms in [('us-west1',['us-west1-a','us-west1-b'],10),('us-east4',['us-east4-a'],20)]]
    latency=dict(status='DESCRIPTIVE_REGION_LATENCY_NOT_CAPACITY',
        catalog_sha256=hashlib.sha256(json.dumps(catalog,sort_keys=True).encode()).hexdigest(),
        unresolved_regions=[],after_central_rounds_order=['us-west1','us-east4'],results=results)
    rounds=[dict(round=i+1,status='COMPLETED',at=f'2026-10-07T0{i}:00:00Z',results=[
        dict(status='CAPACITY_EXHAUSTED',stopped_verified=True,zone='us-central1-'+z)
        for z in 'abc']) for i in range(3)]
    return dict(capacity_rounds=rounds),latency,catalog


@pytest.mark.parametrize('bad',['two-rounds','incomplete','capacity-success','order','catalog','sample'])
def test_regional_admission_rejects_unmet_central_or_latency_prerequisites(bad):
    state,latency,catalog=inputs()
    if bad=='two-rounds':
        state['capacity_rounds'].pop()
    if bad=='incomplete':
        state['capacity_rounds'][2]['status']='RUNNING'
    if bad=='capacity-success':
        state['capacity_rounds'][2]['results'][0]['status']='L4_RUNNING_OBSERVED_NOT_READY'
    if bad=='order':
        latency['after_central_rounds_order'].reverse()
    if bad=='catalog':
        catalog.append(dict(name='nvidia-l4',zone='zones/us-west4-a'))
    if bad=='sample':
        latency['results'][0]['samples'][0]['status']='FAILED'
    with pytest.raises(ValueError):
        admission_order(state,latency,catalog)


def test_stopped_earlier_capacity_does_not_block_latest_exhausted_round():
    state,latency,catalog=inputs()
    state['capacity_rounds'][1]['results'][2]['status']='L4_RUNNING_OBSERVED_NOT_READY'
    assert [r for r,_ in admission_order(state,latency,catalog)] == ['us-west1','us-east4']
    state['capacity_rounds'][1]['results'][2]['stopped_verified']=False
    with pytest.raises(ValueError,match='complete central rounds'):
        admission_order(state,latency,catalog)


@pytest.mark.parametrize('bad',['zone-duplicate','round-id','interval','naive-time','tool-error'])
def test_regional_admission_requires_actual_complete_spaced_census(bad):
    state,latency,catalog=inputs()
    if bad=='zone-duplicate':
        state['capacity_rounds'][0]['results'][1]['zone']='us-central1-a'
    if bad=='round-id':
        state['capacity_rounds'][1]['round']=1
    if bad=='interval':
        state['capacity_rounds'][1]['at']='2026-10-07T00:44:59Z'
    if bad=='naive-time':
        state['capacity_rounds'][0]['at']='2026-10-07T00:00:00'
    if bad=='tool-error':
        state['capacity_rounds'][0]['results'][0]['status']='TOOL_OR_RESOURCE_ERROR_NOT_CAPACITY'
    with pytest.raises(ValueError):
        admission_order(state,latency,catalog)


def test_regional_create_has_native_stop_no_public_ip_no_account_or_spot():
    argv=regional_create('test','disk','us-west1-a','startup','marker','study','cloudrag-i5-us-west1-fixture')
    assert '--no-address' in argv and '--no-service-account' in argv and '--no-scopes' in argv
    assert '--provisioning-model=STANDARD' in argv and '--max-run-duration=3h' in argv
    assert '--instance-termination-action=STOP' in argv and '--deletion-protection' in argv
    assert '--disk=name=disk,boot=yes,auto-delete=no' in argv
    for zone,subnet in [('us-central1-a','cloudrag-i5-us-central1-fixture'),
                        ('europe-west1-a','cloudrag-i5-europe-west1-fixture'),
                        ('us-west1-a','default')]:
        with pytest.raises((ValueError,RuntimeError)):
            regional_create('test','disk',zone,'startup','marker','study',subnet)


def setup(tmp_path,monkeypatch):
    state,latency,catalog=inputs()
    state.update(status='ACTIVE',closure_reserved_utc='2026-10-09T23:01:39Z',cloud_cutoff_usd=90,
        independent_closure_verified=True,resources=[],cost=dict(estimated_spend_usd=13,reserved_retention_and_closure_usd=7))
    (tmp_path/'STATE.json').write_text(json.dumps(state))
    (tmp_path/'cpu-restoration-bootstrap01-proof.json').write_text(json.dumps(dict(
        source_snapshot_id='2',status='CPU_RESTORATION_VERIFIED',synthetic=False)))
    snapshot=dict(id='2',name='qualified',status='READY',diskSizeGb='100',storageLocations=['us-central1'])
    class Cloud:
        def __init__(self):
            self.calls=[]
            self.subnets=[]
        def command(self,args,**kwargs):
            self.calls.append(args)
            if args[:2]==['compute','snapshots']:
                return snapshot
            if args[:2]==['compute','instances']:
                return []
            if args[3]=='list':
                return copy.deepcopy(self.subnets)
            if args[3]=='create':
                self.subnets.append(dict(id='3',name=args[4],region='regions/us-west1',network='networks/study',
                    privateIpGoogleAccess=True,enableFlowLogs=False,ipCidrRange='10.43.16.0/24',
                    description=next(a.split('=',1)[1] for a in args if a.startswith('--description='))))
            if args[3]=='describe':
                return self.subnets[0]
    cloud=Cloud()
    probe=RegionalProbe(tmp_path,cloud,now=lambda:datetime(2026,10,7,12,tzinfo=timezone.utc))
    monkeypatch.setattr('scripts.study_operator.regional_capacity.quote_archive',lambda *a:
                        dict(usd_per_hour='.7',catalog_receipt_sha256='c'*64))
    monkeypatch.setattr('scripts.study_operator.regional_capacity.disk_quote_archive',lambda *a:
                        dict(estimated_usd_gib_h='.0002',usage_unit='GiBy.mo',estimated_month_hours=730))
    return probe,cloud,snapshot,latency,catalog


def test_regions_and_zones_follow_measured_order_and_stop_after_first_capacity(tmp_path,monkeypatch):
    probe,cloud,snapshot,latency,catalog=setup(tmp_path,monkeypatch)
    visits=[]
    def attempt(zone,*args):
        visits.append(zone)
        return dict(status='CAPACITY_EXHAUSTED' if len(visits)==1 else 'L4_RUNNING_OBSERVED_NOT_READY',
                    zone=zone,stopped_verified=True,not_a_measurement_or_ready=True)
    monkeypatch.setattr(probe,'attempt',attempt)
    result=probe.regional(snapshot,'study',latency,catalog)
    assert visits==['us-west1-a','us-west1-b']
    assert result['status']=='CAPACITY_OBSERVED_STOPPED_NOT_READY'
    assert result['not_a_measurement_or_ready']
    state=json.loads((tmp_path/'STATE.json').read_bytes())
    assert state['resources'][0]['type']=='subnet' and state['resources'][0]['id']=='3'
    assert state['resources'][0]['idle_usd_day']==0
    with pytest.raises(ValueError,match='already exists'):
        probe.regional(snapshot,'study',latency,catalog)
    assert visits==['us-west1-a','us-west1-b']


def test_existing_subnet_requires_own_marker_private_access_and_no_flow_logs(tmp_path,monkeypatch):
    probe,cloud,*_=setup(tmp_path,monkeypatch)
    first=probe.subnet('us-west1','study',0)
    count=len([a for a in cloud.calls if a[3]=='create'])
    assert probe.subnet('us-west1','study',0)==first
    assert len([a for a in cloud.calls if a[3]=='create'])==count
    cloud.subnets[0]['enableFlowLogs']=True
    with pytest.raises(ValueError,match='scope or private access'):
        probe.subnet('us-west1','study',0)


def test_regional_disk_transfer_and_compute_reservations_are_separate(tmp_path,monkeypatch):
    probe,cloud,*_=setup(tmp_path,monkeypatch)
    state=json.loads((tmp_path/'STATE.json').read_bytes())
    value=probe.extra_exposure('us-west1-a',state,{})
    hours=(datetime.fromisoformat(state['closure_reserved_utc'])-probe.now()).total_seconds()/3600
    assert value==pytest.approx(2+100*.0002*hours)
    assert 'snapshot restore transfer upperUSD2' in (tmp_path/'COST_LEDGER.md').read_text()
