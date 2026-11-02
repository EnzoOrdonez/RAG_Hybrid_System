"""Auditoría 2026-11-02, hallazgo F-01: start('smoke') persiste el period_id del smoke y
cleanup_cpu deriva de él el nombre esperado de la VM de CPU.

Los tests marcados xfail(strict=True) reproducen el defecto en la base b470d3e: pasan a XPASS
(fallo de la suite) en cuanto se aplique el parche docs/review/2026-11-02/patches/001-*.patch,
y entonces se les quita el marcador.
"""
import copy
import hashlib
import json
import uuid

import pytest

from scripts.study_operator.i7 import Operator
from scripts.study_operator.i7_cpu_pipeline import cleanup_cpu, cpu_label
from tests.test_study_i7_operator import setup


def i7_root(tmp_path):
    """Raíz I7 como la deja i7_config.create: dos periodos, purpose='technical'."""
    op, cloud, reservation = setup(tmp_path)
    periods = {'technical': uuid.uuid4().hex, 'smoke': uuid.uuid4().hex}
    op.config.update(period_ids=periods, period_id=periods['technical'], purpose='technical')
    label = cpu_label(op.config)
    vm = {'name': 'cloudrag-i5-restore-'+label, 'id': '31', 'description': 'CloudRAG-I5-restore-cpu-'+label}
    archive = tmp_path/'private'/'legacy'/'i3-nine-originals.tar'
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b'synthetic archive')
    op.state.update(cpu_resources={'vm': vm, 'source_disk': {}, 'service_disk': {}, 'stage_snapshot': {}},
                    legacy_archive_sha256=hashlib.sha256(b'synthetic archive').hexdigest())
    op.persist()
    return op, periods, vm, reservation


class SmokeThenCleanupCloud:
    """Proveedor falso: arranque con reserva específica y, después, la limpieza de la CPU."""
    def __init__(self, op, vm, reservation):
        self.op, self.vm, self.reservation, self.calls = op, vm, reservation, []

    def require_beta(self):
        return {'status': 'GCLOUD_BETA_INSTALLED', 'beta_version': 'fixture'}

    def command(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[:3] == ['compute', 'instances', 'describe'] and argv[3] == self.vm['name']:
            return dict(self.vm, status='TERMINATED')
        if argv[:3] == ['compute', 'instances', 'describe']:
            return dict(self.op.selected(), status='TERMINATED',
                        machineType='machineTypes/g2-standard-4', deletionProtection=True,
                        disks=[{'autoDelete': False}],
                        scheduling={'instanceTerminationAction': 'STOP', 'maxRunDuration': {'seconds': '10800'}},
                        reservationAffinity={'consumeReservationType': 'SPECIFIC_RESERVATION',
                                             'values': [self.reservation['name']]})
        if argv[:3] == ['compute', 'reservations', 'describe']:
            return copy.deepcopy(self.reservation)
        if argv[:3] in (['compute', 'instances', 'add-metadata'], ['compute', 'instances', 'start']):
            return None
        # Cualquier llamada destructiva de la limpieza demuestra que la identidad se aceptó.
        raise RuntimeError('REACHED_DESTRUCTIVE_CLEANUP:'+' '.join(argv[:3]))


def smoke_started(tmp_path):
    op, periods, vm, reservation = i7_root(tmp_path)
    # La reserva vigente debe cubrir native_stop_s + 60 s desde now() (2026-11-01T00:00Z).
    op.state['reservation']['deleteAtTime'] = '2026-11-01T13:30:00Z'
    op.cloud = SmokeThenCleanupCloud(op, vm, reservation)
    assert op.start('smoke') == {'status': 'STARTED_NOT_READY'}
    return op, periods, vm


def test_smoke_start_persists_the_smoke_period_in_installation(tmp_path):
    """[VERIFICADO] Es el comportamiento que hace posible F-01 (no es el defecto en sí)."""
    _, periods, _ = smoke_started(tmp_path)
    saved = json.loads((tmp_path/'installation.json').read_bytes())
    assert saved['purpose'] == 'smoke'
    assert saved['period_id'] == periods['smoke'] != periods['technical']


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason='F-01: CPU_CLEANUP_IDENTITY_CHANGED tras start(smoke) (base b470d3e)')
def test_cleanup_cpu_after_smoke_still_recognises_the_cpu_of_this_root(tmp_path):
    op, _, _ = smoke_started(tmp_path)
    # close se ejecuta en otro proceso: Operator nuevo leído del disco.
    reopened = Operator(tmp_path, op.cloud, now=op.now)
    with pytest.raises(RuntimeError, match='^REACHED_DESTRUCTIVE_CLEANUP'):
        cleanup_cpu(reopened)


def test_control_cleanup_cpu_after_a_later_technical_start_is_accepted(tmp_path):
    """Control: en el camino feliz (gate, stimulus) un start() técnico restaura el periodo."""
    op, periods, _ = smoke_started(tmp_path)
    assert op.start() == {'status': 'STARTED_NOT_READY'}
    assert op.config['period_id'] == periods['technical']
    with pytest.raises(RuntimeError, match='^REACHED_DESTRUCTIVE_CLEANUP'):
        cleanup_cpu(Operator(tmp_path, op.cloud, now=op.now))
