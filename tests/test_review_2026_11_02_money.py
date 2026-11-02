"""Auditoría 2026-11-02, pasada de dinero/TTL (A-1..A-4) sobre la base b470d3e.

Salvo A-1 (xfail estricto en los casos que hoy fallan), estos tests FIJAN el comportamiento
defectuoso actual: cuando se corrija el defecto deben fallar y hay que invertir su aserción.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_operator.i7 import (PHASE_B_GATE_RESERVATION_H, PHASE_B_RESERVATION_1_H)
from tests.test_study_i7_human import closed_phase_b
from tests.test_study_i7_reservation_budget import LIST_USD
from tests.test_study_i7_reservation_time import MillisecondCloud

T0 = datetime(2026, 11, 2, tzinfo=timezone.utc)   # CPU-build admission = budget_started_utc
# i7_cpu_pipeline.build line 59 with the archived October rates:
# e2-standard-2 = 2 vCPU + 8 GiB; Balanced PD 0.10 USD/GiB-mo / 730 (pricing.disk_quote_archive).
E2_H = 2*LIST_USD['e2_vcpu'] + 8*LIST_USD['e2_ram']
PD_GIB_H = 0.10/730
CPU_EXPOSURE = 6*E2_H + 360*PD_GIB_H*72


class Clock:
    def __init__(self, at):
        self.at = at

    def __call__(self):
        self.at += timedelta(microseconds=1)   # real clocks advance; avoids receipt-name collisions
        return self.at


def phase_b_operator(path, clock, *, spent):
    path.mkdir()
    op, _ = closed_phase_b(path)
    op.now = clock
    op.config['rates']['verified_utc'] = T0.isoformat()
    op.config['cost_plan'] = {'status': 'PRICED_LIST', 'verified_utc': T0.isoformat(),
                              'rates': {k: {'usd': v, 'sku_id': k} for k, v in LIST_USD.items()}}
    op.state.update(spent_usd=spent, reservation_gpu_hours_used=0, budget_started_utc=T0.isoformat())
    op.state.pop('open_exposure_usd', None)
    op.persist()
    op.cloud = MillisecondCloud(op)
    return op


# A-1 ---------------------------------------------------------------------------------------
_A1_STOP = pytest.mark.xfail(strict=True, raises=ValueError, reason='A-1: I7_BUDGET_STOP con la cota de CPU en spent_usd')


@pytest.mark.parametrize('held_h', [12.0, pytest.param(13.25, marks=_A1_STOP),
                                    pytest.param(PHASE_B_RESERVATION_1_H, marks=_A1_STOP)])
def test_A1_gate_reserve_after_long_reservation_1(tmp_path, held_h):
    clock = Clock(T0 + timedelta(hours=2))
    op = phase_b_operator(tmp_path/'r', clock, spent=CPU_EXPOSURE)  # build() charged its exposure
    op.reserve(PHASE_B_RESERVATION_1_H)
    clock.at += timedelta(hours=held_h)
    op.guard()                                     # series finally: stop + guard
    rate = op.config['rates']['compute_usd_h']
    print(f'\nCPU_EXPOSURE={CPU_EXPOSURE:.4f} spent_after_r1={op.state["spent_usd"]:.4f} '
          f'gate_exposure={PHASE_B_GATE_RESERVATION_H*rate+.25:.4f} '
          f'max_r1_hours_for_gate={(20-3-(PHASE_B_GATE_RESERVATION_H*rate+.25)-CPU_EXPOSURE)/rate:.3f}')
    try:
        op.reserve(PHASE_B_GATE_RESERVATION_H)
        print(f'held {held_h} h -> gate ADMITTED')
    except ValueError as error:
        print(f'held {held_h} h -> gate {error}')
        raise


# A-2 ---------------------------------------------------------------------------------------
def test_A2_reserve_refused_after_24h_window_even_for_capacity_retry(tmp_path):
    """ADR-0026 says gpu-restore can be retried in the same root; not after T0+24h."""
    clock = Clock(T0 + timedelta(hours=24, minutes=1))
    op = phase_b_operator(tmp_path/'r', clock, spent=CPU_EXPOSURE)
    with pytest.raises(ValueError) as error:
        op.reserve(PHASE_B_RESERVATION_1_H)
    print('\nreserve at T0+24h01:', error.value)
    assert str(error.value) == 'I7_RETENTION_DEADLINE_CLOSE_RESOURCES'


def test_A2_reservation_admitted_but_unusable_and_gate_attempt_burnt(tmp_path, monkeypatch):
    """reserve(5) at T0+23h50 is admitted with no look-ahead; switch_profile/start at T0+24h+
    hit the retention cutoff after gate-cohort-intent.json ('x', no replay) was written."""
    from scripts.study_operator import i7_admission, i7_phase_b, stimulus_evidence
    from src.ui.components import study_protocol
    clock = Clock(T0 + timedelta(hours=23, minutes=50))
    op = phase_b_operator(tmp_path/'r', clock, spent=CPU_EXPOSURE + 9.0)
    (op.root/'stimulus-acceptance.json').write_text(json.dumps({'status': 'ACCEPTED'}))
    ref = op.root/'i3.json'
    ref.write_text(json.dumps({'rag': {'contexts': {}}}))
    op.config.update(local_reviewed_config='cfg', i3_context_reference_path=str(ref))
    monkeypatch.setattr(study_protocol, 'verify_draw', lambda p: {'config': {}})
    monkeypatch.setattr(stimulus_evidence, 'accept', lambda proofs, c: {'status': 'ACCEPTED', 'software_sha256': 's'})
    monkeypatch.setattr(i7_admission, 'compare_contexts', lambda a, b: None)
    real_reserve = op.reserve
    def reserve_then_time_passes(hours):
        result = real_reserve(hours)
        clock.at += timedelta(minutes=15)          # stop/set-scheduling/smoke take minutes
        return result
    op.reserve = reserve_then_time_passes
    with pytest.raises(ValueError) as error:
        i7_phase_b._gate_after_prerequisites(op, {'software_sha256': 's'})
    print('\ngate first run:', error.value, '| attempt file exists:', (op.root/'gate-cohort-intent.json').exists(),
          '| reservation deleted & charged:', op.state['reservation'].get('deleted_verified'))
    assert str(error.value) == 'I7_RETENTION_DEADLINE_CLOSE_RESOURCES'
    with pytest.raises(ValueError) as again:
        i7_phase_b._gate_after_prerequisites(op, {'software_sha256': 's'})
    print('gate second run:', again.value)
    assert str(again.value) == 'GATE_ATTEMPT_EXISTS_NO_REPLAY'


def test_A2_series_slot_burnt_by_start_admission(tmp_path):
    """series() writes the INTENT record before operator.start(); an admission refusal
    (24 h cutoff, budget) leaves a non-COMPLETE record that blocks every later run."""
    from scripts.study_operator import i7_workflows
    class Op:
        root = tmp_path
        def start(self):
            raise ValueError('I7_RETENTION_DEADLINE_CLOSE_RESOURCES')
        def stop(self):
            pass
        def guard(self):
            pass
    with pytest.raises(ValueError, match='I7_RETENTION_DEADLINE'):
        i7_workflows.series(Op())
    with pytest.raises(ValueError) as error:
        i7_workflows.series(Op())
    print('\nseries rerun:', error.value)
    assert str(error.value) == 'TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW'


# A-3 ---------------------------------------------------------------------------------------
def test_A3_open_cpu_exposure_overwritten_by_reservation(tmp_path):
    """build() left open_exposure_usd=CPU (spent not charged: clean_snapshot's finally STOP raised
    after qualified_service_snapshot was persisted). reserve() overwrites the scalar; guard() zeroes it."""
    clock = Clock(T0 + timedelta(hours=3))
    op = phase_b_operator(tmp_path/'r', clock, spent=0)
    op.state['open_exposure_usd'] = CPU_EXPOSURE
    op.persist()
    op.reserve(PHASE_B_RESERVATION_1_H)
    rate = op.config['rates']['compute_usd_h']
    r1 = PHASE_B_RESERVATION_1_H*rate+.25
    print(f'\nafter reserve: open={op.state["open_exposure_usd"]:.4f} expected(cpu+r1)={CPU_EXPOSURE+r1:.4f}')
    clock.at += timedelta(hours=PHASE_B_RESERVATION_1_H)
    op.guard()
    print(f'after guard: open={op.state["open_exposure_usd"]} spent={op.state["spent_usd"]:.4f}')
    correct = CPU_EXPOSURE + op.state['spent_usd'] + PHASE_B_GATE_RESERVATION_H*rate+.25 + 3
    op.reserve(PHASE_B_GATE_RESERVATION_H)
    print(f'gate ADMITTED although correct ledger total = {correct:.4f} > 20')
    assert correct > 20
    assert op.state['open_exposure_usd'] == pytest.approx(PHASE_B_GATE_RESERVATION_H*rate+.25)


# A-4 ---------------------------------------------------------------------------------------
def test_A4_cli_default_hours_is_stale_and_exhausts_phase_b(tmp_path):
    import argparse, inspect
    from scripts.study_operator import i7
    src = inspect.getsource(i7.main)
    assert "add_argument('--hours',type=float,default=12)" in src
    clock = Clock(T0 + timedelta(hours=3))
    op = phase_b_operator(tmp_path/'r', clock, spent=CPU_EXPOSURE)
    op.reserve(12)                     # `i7 --live reserve` without --hours
    clock.at += timedelta(hours=1)
    op.guard()
    with pytest.raises(ValueError) as error:
        op.reserve(PHASE_B_RESERVATION_1_H)   # what gpu_restore then needs
    print('\ngpu_restore reserve after CLI default:', error.value)
