from datetime import datetime,timezone

import pytest


def test_all_skus_and_retention_contribute_to_budget_cut():
    from scripts.study_operator.i7_costs import KEYS,exposure
    rates={key:{'usd':.1,'sku_id':key} for key in KEYS}
    quantities={key:0. for key in KEYS}
    quantities.update(g2_vcpu=4,g2_ram=16,l4=1,pd_gib_h=100,snapshot_gib_h=50,
                      ip_used_h=2,ip_unused_h=2,egress_gib=1,snapshot_transfer_gib=1)
    result=exposure({'status':'PRICED_LIST','rates':rates},quantities)
    assert result['decision']=='STOP' and result['total_with_closure_usd']>20
    assert all(key in result['rows'] for key in KEYS)
    quantities['pd_gib_h']=quantities['snapshot_gib_h']=0
    assert exposure({'status':'PRICED_LIST','rates':rates},quantities)['decision']=='ADMIT'


@pytest.mark.parametrize('fault',['missing','negative','nan','unpriced'])
def test_unpriced_or_invalid_costs_never_admit(fault):
    from scripts.study_operator.i7_costs import KEYS,exposure
    rates={key:{'usd':.01,'sku_id':key} for key in KEYS}
    quantities={key:1. for key in KEYS}
    if fault=='missing':
        rates.pop('ip_unused_h')
    if fault=='negative':
        quantities['l4']=-1
    if fault=='nan':
        rates['l4']['usd']=float('nan')
    result=exposure({'status':'UNPRICED' if fault=='unpriced' else 'PRICED_LIST','rates':rates},quantities)
    assert result['decision']=='STOP'


def test_union_reservation_and_vm_but_storage_continues_after_stop():
    from scripts.study_operator.i7_costs import allocated_hours,retained_gib_hours
    intervals=[('2026-11-01T00:00:00Z','2026-11-01T02:00:00Z')]
    running=[('2026-11-01T01:00:00Z','2026-11-01T03:00:00Z')]
    assert allocated_hours(intervals,running)==3
    assert retained_gib_hours(100,'2026-11-01T00:00:00Z',datetime(2026,11,2,tzinfo=timezone.utc))==2400


def test_operator_blocks_unpriced_before_cloud_and_after_retention_deadline(tmp_path):
    from tests.test_study_i7_operator import setup
    operator,cloud,_=setup(tmp_path)
    operator.config['cost_plan']['status']='UNPRICED'
    with pytest.raises(ValueError,match='UNPRICED'):
        operator.admit_paid(0)
    assert not cloud.calls
    operator.config['cost_plan']['status']='PRICED_LIST'
    operator.now=lambda:datetime(2026,11,2,1,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match='RETENTION'):
        operator.admit_paid(0)
    assert not cloud.calls
