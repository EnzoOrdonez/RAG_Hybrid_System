"""Reproductores de la auditoría 2026-11-02 (hallazgos R-01 y R-02, ver docs/review/2026-11-02/).

`scripts/study_operator/**` es zona congelada: estos tests documentan el defecto con
`xfail(strict=True)`. Al aplicar los parches de docs/review/2026-11-02/patches/ pasan a XPASS
y la suite falla, lo que obliga a retirar la marca (DECISIONES.md D-04).
"""
import copy
from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_operator.policy import OperatorError
from tests.test_study_i7_operator import setup


def _gcloud_millis(value: str) -> str:
    """gcloud 570 envía deleteAtTime con FormatDateTime('%Y-%m-%dT%H:%M:%S.%3f%Ez'): milisegundos."""
    dt = datetime.fromisoformat(value)
    return dt.replace(microsecond=dt.microsecond // 1000 * 1000).isoformat(timespec="milliseconds")


def _operator(tmp_path, *, now, stockout=False):
    op, cloud, res = setup(tmp_path)
    op.state.pop("reservation")
    clock = {"t": now}
    op.now = lambda: clock["t"]
    store = {}
    base = cloud.command

    def command(argv, **kwargs):
        cloud.calls.append(argv)
        if argv[:4] == ["beta", "compute", "reservations", "create"]:
            if stockout:
                raise OperatorError("ZONE_RESOURCE_POOL_EXHAUSTED: L4 sin capacidad en la zona")
            at = [a for a in argv if a.startswith("--delete-at-time=")][0].split("=", 1)[1]
            store["r"] = dict(copy.deepcopy(res), name=argv[4], deleteAtTime=_gcloud_millis(at),
                              creationTimestamp=now.isoformat())
            return None
        if argv[:3] == ["compute", "reservations", "describe"]:
            return copy.deepcopy(store["r"])
        if argv[:3] == ["compute", "reservations", "list"]:
            return [copy.deepcopy(store["r"])] if "r" in store else []
        return base(argv, **kwargs)

    cloud.command = command
    return op, clock, store


@pytest.mark.xfail(strict=True, reason="R-01 (S0): i7.py:108,120,279 compara deleteAtTime con igualdad al microsegundo")
def test_reserve_records_reservation_when_clock_has_microseconds(tmp_path):
    op, _, store = _operator(tmp_path, now=datetime(2026, 11, 1, 12, 0, 0, 123456, tzinfo=timezone.utc))
    op.reserve(12)
    assert "r" in store                       # la reserva existe (y factura) en la nube
    assert "reservation" in op.state          # ...y el operador debe poder gestionarla y borrarla


def test_reserve_with_whole_millisecond_clock_still_works(tmp_path):
    """Control: con microsegundos múltiplos de 1000 el mismo flujo pasa (el defecto es de precisión)."""
    op, _, store = _operator(tmp_path, now=datetime(2026, 11, 1, 12, 0, 0, 123000, tzinfo=timezone.utc))
    op.reserve(12)
    assert "r" in store and "reservation" in op.state


@pytest.mark.xfail(strict=True, reason="R-02 (S1): i7.py:102-110,286-289 un create rechazado consume horas y presupuesto")
def test_rejected_reservation_create_does_not_consume_plan(tmp_path):
    t0 = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone.utc)
    op, clock, store = _operator(tmp_path, now=t0, stockout=True)
    with pytest.raises(OperatorError):
        op.reserve(12)
    assert not store                          # nada se creó en la nube
    clock["t"] = t0 + timedelta(minutes=5)
    op.reconcile()                            # hoy: UNKNOWN_CREATE_OUTCOME_PRESERVE_INTENT durante 12 h
    assert op.state.get("reservation_gpu_hours_used", 0) == 0
    assert op.state.get("spent_usd", 0) == 0
