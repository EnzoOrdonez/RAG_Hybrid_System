"""Phase B reservation hours: closable cost basis, boot TTL headroom and the USD 20 cut."""
import json
from datetime import datetime, timezone

import pytest

from scripts.study_operator.i7 import (PHASE_B_GATE_RESERVATION_H, PHASE_B_RESERVATION_1_H,
                                       PHASE_B_RESERVATION_TOTAL_H)
from tests.test_study_i7_human import closed_phase_b
from tests.test_study_i7_reservation_time import MillisecondCloud

# Archived October list prices (installation cost_plan, LIST_PRICES_NOT_INVOICE).
LIST_USD = {'g2_vcpu':0.024988212,'g2_ram':0.002927448,'l4':0.560040239,'e2_vcpu':0.02181159,
    'e2_ram':0.00292353,'pd_gib_h':0.00013440860215053763,'snapshot_gib_h':6.720430107526882e-05,
    'ip_used_h':0.005,'ip_unused_h':0.01,'egress_gib':0.19,'snapshot_transfer_gib':0.02,
    'gcs_gib_h':2.6881720430107527e-05,'gcs_a':5e-06,'gcs_b':4e-07,'gcs_egress_gib':0.12}


class Held(Exception):
    pass


def test_both_reservations_fit_the_phase_b_cut_with_closure():
    from scripts.study_operator.i7_costs import default_quantities, exposure
    prices = {'status':'PRICED_LIST','rates':{k:{'usd':v,'sku_id':k} for k,v in LIST_USD.items()}}
    quantities = default_quantities(24)
    assert quantities['l4'] == PHASE_B_RESERVATION_TOTAL_H
    result = exposure(prices, quantities)
    assert result['decision'] == 'ADMIT'
    assert result['total_with_closure_usd'] == pytest.approx(19.75136847, abs=1e-6)


def test_cost_basis_of_reservation_1_is_closable(tmp_path):
    from tests.test_study_i7_operator import setup
    op, _, _ = setup(tmp_path)
    rate = op.config['rates']['compute_usd_h']
    op.state['reservation']['cost_basis'].update(hours=PHASE_B_RESERVATION_1_H,
        exposure_usd=PHASE_B_RESERVATION_1_H*rate+.25)
    assert op.guard()['status'] == 'RESERVATION_DELETED_VERIFIED'


def reserving(path, used):
    path.mkdir()
    op, _ = closed_phase_b(path)
    op.state['spent_usd'] = 0
    op.state['reservation_gpu_hours_used'] = used
    op.cloud = MillisecondCloud(op)
    return op


def test_reservation_1_and_gate_hours_are_admitted(tmp_path):
    held = 'SPECIFIC_RESERVATION_HELD'
    assert reserving(tmp_path/'r1', 0).reserve(PHASE_B_RESERVATION_1_H)['status'] == held
    assert reserving(tmp_path/'gate', PHASE_B_RESERVATION_1_H).reserve(PHASE_B_GATE_RESERVATION_H)['status'] == held


def test_hours_beyond_reservation_1_or_the_phase_plan_are_refused(tmp_path):
    with pytest.raises(ValueError, match='^RESERVATION_TTL_REQUIRED$'):
        reserving(tmp_path/'r1', 0).reserve(PHASE_B_RESERVATION_1_H+.25)
    with pytest.raises(ValueError, match='EXCEED_COST_PLAN'):
        reserving(tmp_path/'gate', PHASE_B_RESERVATION_1_H).reserve(PHASE_B_GATE_RESERVATION_H+.25)


def test_gpu_restore_holds_reservation_1(monkeypatch):
    from scripts.study_operator import i7_cpu_pipeline
    monkeypatch.setattr(i7_cpu_pipeline, 'verify_image_or_stop_cpu', lambda operator: None)
    class Cloud:
        def command(self, argv, **kwargs):
            if argv[:3] == ['compute','snapshots','describe']:
                return {'id':'1','status':'READY','storageLocations':['us-central1']}
            if argv[:3] == ['compute','instances','list']:
                return []
            raise AssertionError(argv)
    class Operator:
        state = {'qualified_service_snapshot':{'name':'q','id':'1'}}
        cloud = Cloud()
        def reserve(self, hours):
            raise Held(hours)
    with pytest.raises(Held) as held:
        i7_cpu_pipeline.gpu_restore(Operator())
    assert held.value.args == (PHASE_B_RESERVATION_1_H,)


def test_gate_holds_the_gate_reservation(tmp_path, monkeypatch):
    from scripts.study_operator import i7_admission, i7_phase_b, stimulus_evidence
    from src.ui.components import study_protocol
    (tmp_path/'stimulus-acceptance.json').write_text(json.dumps({'status':'ACCEPTED'}))
    reference = tmp_path/'i3.json'
    reference.write_text(json.dumps({'rag':{'contexts':{}}}))
    monkeypatch.setattr(study_protocol, 'verify_draw', lambda path: {'config':{}})
    monkeypatch.setattr(stimulus_evidence, 'accept', lambda proofs, config: {'status':'ACCEPTED','software_sha256':'s'})
    monkeypatch.setattr(i7_admission, 'compare_contexts', lambda live, reference: None)
    class Operator:
        root = tmp_path
        config = {'local_reviewed_config':'cfg','i3_context_reference_path':str(reference),'image_id':'i'}
        state = {'reservation':{'deleted_verified':True}}
        def now(self):
            return datetime(2026,11,1,tzinfo=timezone.utc)
        def reserve(self, hours):
            raise Held(hours)
    with pytest.raises(Held) as held:
        i7_phase_b._gate_after_prerequisites(Operator(), {'software_sha256':'s'})
    assert held.value.args == (PHASE_B_GATE_RESERVATION_H,)
