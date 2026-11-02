"""Auditoría 2026-11-02, tarea 2 de Enzo: archivos de un solo uso frente a un rechazo gratuito de
capacidad en `instances start` con la reserva 1 ya tomada (solo dobles, sin nube).

Fijan el comportamiento actual de la base b470d3e: gate-cohort-intent.json y stimulus-NN.json se
consumen; live-contexts.json no. Al corregir el orden (registrar justo antes del primer efecto
pagado) los dos primeros deben invertirse.
"""
import json

import pytest

from scripts.study_operator.policy import CapacityRejected
from tests.test_review_2026_11_02_recovery import make_op, reload

START = ('compute', 'instances', 'start')


def capacity():
    return CapacityRejected('ZONE_RESOURCE_POOL_EXHAUSTED: fixture', code='ZONE_RESOURCE_POOL_EXHAUSTED')


def no_startup(monkeypatch):
    monkeypatch.setattr('scripts.study_operator.startup.write_startup', lambda path, config, **k: None)


def test_stimulus_slot_is_consumed_by_capacity_refusal_on_start(tmp_path, monkeypatch):
    from scripts.study_operator.i7_workflows import series
    no_startup(monkeypatch)
    op, cloud = make_op(tmp_path, reservation_ttl_h=12)
    cloud.fail[START] = capacity()
    with pytest.raises(CapacityRejected):
        series(op)
    record = json.loads((tmp_path/'attempts'/'stimulus-01.json').read_text())
    assert record == {'status': 'INTENT', 'boot_index': 1}
    assert not (tmp_path/'attempts'/'stimulus-01.complete.json').exists()
    assert reload(op).state['reservation']['deleted_verified']      # finally: guard() cerró R1
    # El reintento se bloquea en la ranura 1; el finally (guard sobre R1 cerrada) tapa la causa (B-2b).
    with pytest.raises(ValueError, match='^RESERVATION_ALREADY_CLOSED$') as retry:
        series(reload(op))
    assert str(retry.value.__context__) == 'TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW'


def test_gate_attempt_is_consumed_by_capacity_refusal_on_smoke_start(tmp_path, monkeypatch):
    from scripts.study_operator import i7_admission, i7_phase_b, i7_smoke, stimulus_evidence
    from src.ui.components import study_protocol
    no_startup(monkeypatch)
    op, cloud = make_op(tmp_path, reservation_ttl_h=12)
    (tmp_path/'stimulus-acceptance.json').write_text(json.dumps({'status': 'ACCEPTED'}))
    ref = tmp_path/'ref.json'
    ref.write_text(json.dumps({'rag': {'contexts': {}}}))
    op.config.update(local_reviewed_config=str(tmp_path), i3_context_reference_path=str(ref))
    monkeypatch.setattr(stimulus_evidence, 'accept', lambda p, c: {'status': 'ACCEPTED', 'software_sha256': 's'})
    monkeypatch.setattr(study_protocol, 'verify_draw', lambda path: {'config': {}})
    monkeypatch.setattr(i7_admission, 'compare_contexts', lambda a, b: None)
    monkeypatch.setattr(i7_smoke, 'validate_receipt', lambda path: {})
    cloud.fail[START] = capacity()
    with pytest.raises(CapacityRejected):
        i7_phase_b._gate_after_prerequisites(op, {'software_sha256': 's'})
    assert json.loads((tmp_path/'gate-cohort-intent.json').read_text())['replay_allowed'] is False
    assert sum(c[:3] == list(START) for c in cloud.calls) == 1        # un intento de arranque rechazado
    with pytest.raises(ValueError, match='^GATE_ATTEMPT_EXISTS_NO_REPLAY$'):
        i7_phase_b._gate_after_prerequisites(reload(op), {'software_sha256': 's'})


def test_live_contexts_is_not_consumed_by_capacity_refusal(tmp_path, monkeypatch):
    from scripts.study_operator.i7_phase_b import dispatch
    no_startup(monkeypatch)
    op, cloud = make_op(tmp_path, reservation_ttl_h=12)
    cloud.fail[START] = capacity()
    with pytest.raises(CapacityRejected):
        dispatch(op, 'contexts', {})
    assert not (tmp_path/'live-contexts.json').exists()
    # El reintento vuelve a llegar al arranque: no hay registro previo que lo bloquee.
    before = sum(c[:3] == list(START) for c in cloud.calls)
    with pytest.raises(CapacityRejected):
        dispatch(reload(op), 'contexts', {})
    assert sum(c[:3] == list(START) for c in cloud.calls) == before + 1
