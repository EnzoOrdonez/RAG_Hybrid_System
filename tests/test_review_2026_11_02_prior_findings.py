"""Auditoría 2026-11-02: re-validación de R-01 y R-02 del informe anterior sobre la base b470d3e.

R-01 (REFUTADO en esta base): la hora de borrado se envía sin fracción de segundo, así que la
lectura del SDK (redondeo a milisegundos, cualquier desfase horario) siempre coincide.
R-02 (OBSOLETO, ADR-0026): un rechazo de capacidad con listado vacío libera horas y exposición y
permite reintentar en la misma raíz. El residual documentado (listado fallido => UNKNOWN) se fija
aquí como comportamiento actual, no como defecto.
"""
import json
from datetime import timedelta, timezone

import pytest

from scripts.study_operator.i7 import PHASE_B_RESERVATION_1_H, reservation_deadline_matches
from scripts.study_operator.policy import CapacityRejected
from tests.test_study_i7_human import closed_phase_b
from tests.test_study_i7_reservation_capacity import RejectingCloud, ledger
from tests.test_study_i7_reservation_time import MillisecondCloud, sdk_delete_at_time


def phase_b(tmp_path):
    op, _ = closed_phase_b(tmp_path)
    op.state['spent_usd'] = 0
    op.persist()
    return op


@pytest.mark.parametrize('microsecond', [0, 1, 499, 500, 999_499, 999_500, 999_999])
@pytest.mark.parametrize('offset', [0, -7, 5.5])
def test_r01_whole_second_deadline_round_trips_sdk_milliseconds(tmp_path, microsecond, offset):
    op = phase_b(tmp_path)
    zone = timezone(timedelta(hours=offset))
    instant = op.now().replace(microsecond=microsecond).astimezone(zone)
    op.now = lambda: instant
    cloud = MillisecondCloud(op, offset=offset)
    op.cloud = cloud
    result = op.reserve(PHASE_B_RESERVATION_1_H)
    assert result['status'] == 'SPECIFIC_RESERVATION_HELD'
    sent = next(a.split('=', 1)[1] for a in cloud.calls[0] if a.startswith('--delete-at-time='))
    assert '.' not in sent and sent.endswith('+00:00')
    assert 'reservation_intent' not in json.loads(op.state_path.read_bytes())


def test_r01_matcher_still_rejects_a_real_one_second_change():
    sent = '2026-11-02T06:55:06+00:00'
    assert reservation_deadline_matches(sent, '2026-11-02T06:55:06.000Z')
    assert reservation_deadline_matches(sent, '2026-11-01T22:55:06.000-08:00')
    assert not reservation_deadline_matches(sent, '2026-11-02T06:55:07.000Z')
    assert not reservation_deadline_matches(sent, sdk_delete_at_time(
        __import__('datetime').datetime(2026, 11, 2, 6, 55, 5, 998_000, tzinfo=timezone.utc)))


def test_r02_released_refusal_allows_same_root_retry_with_full_hours(tmp_path):
    op = phase_b(tmp_path)
    before = ledger(op)
    op.cloud = RejectingCloud(op)
    with pytest.raises(CapacityRejected):
        op.reserve(PHASE_B_RESERVATION_1_H)
    assert ledger(op) == before
    op.cloud.reject = False
    assert op.reserve(PHASE_B_RESERVATION_1_H)['status'] == 'SPECIFIC_RESERVATION_HELD'
    assert op.state['reservation_gpu_hours_used'] == PHASE_B_RESERVATION_1_H


def test_r02_residual_failed_listing_keeps_the_admission_unknown(tmp_path):
    """[DECLARADO en ADR-0026 §4] Conservador: sin listado no se libera nada."""
    op = phase_b(tmp_path)
    op.cloud = RejectingCloud(op, list_fails=True)
    with pytest.raises(TypeError, match='RESERVATION_LIST_NOT_SUCCESSFUL'):
        op.reserve(PHASE_B_RESERVATION_1_H)
    intent, used, exposure, _ = ledger(op)
    assert intent['status'] == 'UNKNOWN' and used == PHASE_B_RESERVATION_1_H and exposure > 0
