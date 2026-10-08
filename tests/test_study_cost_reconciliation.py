from decimal import Decimal
import json

import pytest

from scripts.study_operator.cost_reconciliation import interval_cost, reconcile, static_ip_upper
from scripts.study_operator.retention import digest


def interval(**changes):
    return dict(kind='disk',id='1',created_utc='2026-10-07T00:00:00Z',
        retired_utc='2026-10-07T02:00:00Z',units=100,usd_unit_h='.001',**changes)


def test_creation_and_retirement_are_clipped_once_and_old_retirement_costs_zero():
    rows=[interval(),dict(interval(),id='2',created_utc='2026-10-06T00:00:00Z',retired_utc='2026-10-06T02:00:00Z')]
    result=interval_cost(rows,'2026-10-07T01:00:00Z','2026-10-07T03:00:00Z')
    assert Decimal(result['estimated_increment_usd'])==Decimal('.1')
    assert result['intervals'][1]['interval_s']=='0'
    assert all(r['not_invoice'] for r in result['intervals'])


def test_live_resource_charge_ends_at_recorded_census_and_not_wall_clock():
    result=interval_cost([dict(interval(),retired_utc=None)],'2026-10-07T01:00:00Z','2026-10-07T03:00:00Z')
    assert Decimal(result['estimated_increment_usd'])==Decimal('.2')


@pytest.mark.parametrize('defect',['duplicate','naive','reversed','future_retirement','nan','boolean','negative'])
def test_invalid_intervals_never_change_ledger(defect):
    rows=[interval()]
    if defect=='duplicate':
        rows*=2
    elif defect=='naive':
        rows[0]['created_utc']='2026-10-07T00:00:00'
    elif defect=='reversed':
        rows[0]['retired_utc']='2026-10-06T00:00:00Z'
    elif defect=='future_retirement':
        rows[0]['retired_utc']='2026-10-08T00:00:00Z'
    else:
        rows[0]['usd_unit_h']={'nan':'NaN','boolean':True,'negative':-1}[defect]
    with pytest.raises(ValueError):
        interval_cost(rows,'2026-10-07T01:00:00Z','2026-10-07T03:00:00Z')


def ip_sku():
    return dict(skuId='official-fixture',pricingInfo=[dict(effectiveTime='2026-10-06T00:00:00Z',
        pricingExpression=dict(usageUnit='h',tieredRates=[
            dict(startUsageAmount=0,unitPrice=dict(currencyCode='USD',units='0',nanos=0)),
            dict(startUsageAmount=1,unitPrice=dict(currencyCode='USD',units='0',nanos=10000000))]))])


def test_ip_upper_explicitly_keeps_free_tier_as_an_upper_estimate():
    result=static_ip_upper(ip_sku())
    assert result['usd_h_upper']=='0.01' and result['not_invoice']
    assert result['method']=='MAXIMUM_TIER_UPPER_FREE_TIER_NOT_DEDUCTED'


@pytest.mark.parametrize('defect',['currency','unit','order'])
def test_unquoted_or_invalid_ip_tariffs_are_rejected(defect):
    sku=ip_sku()
    expression=sku['pricingInfo'][0]['pricingExpression']
    if defect=='currency':
        expression['tieredRates'][1]['unitPrice']['currencyCode']='PEN'
    elif defect=='unit':
        expression['usageUnit']='mo'
    else:
        expression['tieredRates'][1]['startUsageAmount']=0
    with pytest.raises(ValueError):
        static_ip_upper(sku)


def reconciliation_fixture(tmp_path, monkeypatch):
    from scripts.study_operator import cost_reconciliation as module

    begin='2026-10-07T00:00:00Z'
    end='2026-10-07T02:00:00Z'
    cost=dict(as_of_utc=begin,estimated_spend_usd=10,initial_margin_separate_usd=2.72,
        image_egress_upper_separate_usd=.484,unreconciled_snapshot_transfer_upper_separate_usd=20)
    disk=dict(id='1',sizeGb='100',type='pd-balanced',zone='us-central1-a',creationTimestamp=begin)
    snapshot=dict(id='3',status='READY',storageBytes=str(2**30),storageLocations=['us-central1'],creationTimestamp=begin)
    resources=dict(vms=[dict(id='4',status='TERMINATED',deletionProtection=True)],
                   disks=[disk],snapshots=[snapshot],ips=[])
    live=dict(at=end,resources=resources,listing_sha256=digest(resources),protected_ids=dict(vm='4',disk='1'))
    extra=json.loads(json.dumps(live))
    extra['resources']['disks'].append(dict(disk,id='2',zone='us-west1-a',creationTimestamp='2026-10-07T00:30:00Z'))
    extra['listing_sha256']=digest(extra['resources'])
    values=dict(baseline=dict(cost=cost),inventory=live,previous_inventory=live,extra_inventory=extra,
        deleted_2=dict(resource_id='2',status='RESOURCE_ABSENCE_VERIFIED',at='2026-10-07T01:00:00Z'),
        ip_start=dict(started_utc='2026-10-07T00:30:00Z',exit_code=0,command=['cli','ip-reserve']),
        ip_end=dict(ended_utc='2026-10-07T01:00:00Z',exit_code=0,command=['cli','ip-release']))
    inputs=dict(ip_id='5')
    for key,value in values.items():
        name=key+'.json'
        inputs[key]=name
        (tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    state=dict(cost=cost,deadline_utc='2026-10-10T00:00:00Z',open_exposures={'held_not_spent':5})
    (tmp_path/'STATE.json').write_text(json.dumps(state),encoding='utf-8')
    monkeypatch.setattr(module,'verify',lambda root:None)
    monkeypatch.setattr(module,'disk_quote_archive',lambda *args:dict(estimated_usd_gib_h=.1/730))
    monkeypatch.setattr(module,'quote_archive',lambda *args:dict(usd_per_hour=.7))
    prices=[]
    for description,regions,unit,nanos in [
        ('Storage PD Snapshot',['us-central1'],'GiBy.mo',50000000),
        ('PD snapshot Data Transfer Out within North America',['global'],'GiBy',20000000)]:
        row=ip_sku()
        row.update(description=description,serviceRegions=regions)
        expression=row['pricingInfo'][0]['pricingExpression']
        expression['usageUnit']=unit
        expression['tieredRates']=[dict(startUsageAmount=0,unitPrice=dict(currencyCode='USD',units='0',nanos=nanos))]
        prices.append(row)
    ip=ip_sku()
    ip.update(description='Static Ip Charge',serviceRegions=['us-west1'])
    prices.append(ip)
    monkeypatch.setattr(module,'catalog_rows',lambda *args:prices)
    return inputs,state


def test_complete_synthetic_reconciliation_pins_inputs_keeps_margins_and_rejects_replay(tmp_path,monkeypatch):
    inputs,state=reconciliation_fixture(tmp_path,monkeypatch)
    output=tmp_path/'result.json'
    result=reconcile(tmp_path,inputs,output)
    assert result['status']=='ESTIMATED_RECONCILIATION_NOT_INVOICE'
    assert result['gpu_runtime_not_inferred'] and result['remaining_operator_reservations_not_spend']
    assert result['cost']['unreconciled_snapshot_transfer_upper_separate_usd']==22
    assert result['cost']['estimated_spend_usd']==pytest.approx(10+100*.1/730*2+.05/730*2+100*.1/730*.5+.005+.01*2/24)
    current=json.loads((tmp_path/'STATE.json').read_bytes())
    assert current['open_exposures']==state['open_exposures']
    assert current['cost']['as_of_utc']=='2026-10-07T02:00:00Z'
    assert all(len(row['sha256'])==64 for row in result['inputs'].values())
    before=(tmp_path/'STATE.json').read_bytes()
    with pytest.raises(ValueError,match='no replay'):
        reconcile(tmp_path,inputs,output)
    assert (tmp_path/'STATE.json').read_bytes()==before


def test_snapshot_storage_growth_uses_largest_actual_listing_not_stale_prior_size(tmp_path,monkeypatch):
    inputs,_=reconciliation_fixture(tmp_path,monkeypatch)
    path=tmp_path/inputs['inventory']
    live=json.loads(path.read_bytes())
    live['resources']['snapshots'][0]['storageBytes']=str(4*2**30)
    live['listing_sha256']=digest(live['resources'])
    path.write_text(json.dumps(live),encoding='utf-8')
    result=reconcile(tmp_path,inputs,tmp_path/'result.json')
    snapshot=next(r for r in result['recorded_intervals']['intervals'] if r['kind']=='snapshots')
    assert Decimal(snapshot['units'])==4
    assert Decimal(snapshot['estimated_usd'])==pytest.approx(Decimal(4)*Decimal('.05')/730*2)


@pytest.mark.parametrize('defect',['failed_ip','altered_listing','unproved_deletion','advanced_ledger'])
def test_invalid_synthetic_evidence_does_not_write_receipt_or_update_ledger(tmp_path,monkeypatch,defect):
    inputs,state=reconciliation_fixture(tmp_path,monkeypatch)
    if defect=='advanced_ledger':
        state['cost']['as_of_utc']='2026-10-07T01:00:00Z'
        path=tmp_path/'STATE.json'
        value=state
    else:
        key={'failed_ip':'ip_end','altered_listing':'inventory','unproved_deletion':'deleted_2'}[defect]
        path=tmp_path/inputs[key]
        value=json.loads(path.read_bytes())
        if defect=='failed_ip':
            value['exit_code']=1
        elif defect=='altered_listing':
            value['listing_sha256']='altered'
        else:
            value['status']='NOT_VERIFIED'
    path.write_text(json.dumps(value),encoding='utf-8')
    before=(tmp_path/'STATE.json').read_bytes()
    output=tmp_path/'result.json'
    with pytest.raises(ValueError):
        reconcile(tmp_path,inputs,output)
    assert not output.exists() and (tmp_path/'STATE.json').read_bytes()==before
