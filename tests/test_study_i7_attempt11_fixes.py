"""Correcciones tras el intento 10 de la Fase B (ADR-0030).

stimulus-10: el boot 2 del estímulo terminó porque una sola conexión OpenSSH por IAP salió con 255
en `stimulus-ack`, después de descargar, verificar y guardar la prueba en local. En el invitado,
la descarga, el ACK y la finalización son idempotentes; por eso el puente los reintenta igual que
los sondeos de solo lectura. Los arranques que registran un trabajo siguen con un solo intento.
"""
import json
from pathlib import Path

import pytest

from scripts.study_operator.policy import TransportUnavailable
from scripts.study_operator.stimulus_host_evidence import sha
from tests.test_study_i7_attempt10_fixes import OK, _operator, _transport

IDEMPOTENT_COLLECTION = ('stimulus-evidence', 'stimulus-ack', 'gate-evidence', 'gate-ack',
                         'gate-terminal-evidence', 'gate-finalized')


# --- Puente: reintento acotado de la recolección idempotente ------------------------------------

@pytest.mark.parametrize('operation', IDEMPOTENT_COLLECTION)
def test_idempotent_collection_retries_transport_and_then_succeeds(tmp_path, operation):
    op, sleeps = _operator(tmp_path, [_transport(), _transport(), OK])
    assert op.bridge({'operation': operation}, private=True)['status'] == 'OK'
    assert len(op.cloud.calls) == 3 and sleeps == [15, 45]


@pytest.mark.parametrize('operation', IDEMPOTENT_COLLECTION)
def test_idempotent_collection_gives_up_after_bounded_attempts(tmp_path, operation):
    op, sleeps = _operator(tmp_path, [_transport(), _transport(), _transport()])
    with pytest.raises(TransportUnavailable):
        op.bridge({'operation': operation}, private=True)
    assert len(op.cloud.calls) == 3 and sleeps == [15, 45]


@pytest.mark.parametrize('operation', ['stimulus-start', 'gate-start'])
def test_registering_starts_keep_a_single_attempt(tmp_path, operation):
    op, sleeps = _operator(tmp_path, [_transport(), OK])
    with pytest.raises(TransportUnavailable):
        op.bridge({'operation': operation}, private=True)
    assert len(op.cloud.calls) == 1 and sleeps == []


def test_retryable_operations_are_read_only_polls_and_idempotent_collection():
    from scripts.study_operator.lifecycle import RETRYABLE_BRIDGE_OPERATIONS, TRANSPORT_RETRY_DELAYS_S
    polls = {'stimulus-status', 'gate-status', 'i7-smoke-observe', 'preflight', 'i7-preparation-check'}
    assert RETRYABLE_BRIDGE_OPERATIONS == frozenset(polls | set(IDEMPOTENT_COLLECTION))
    assert TRANSPORT_RETRY_DELAYS_S == (15, 45)


# --- Invitado: repetir la recolección no cambia ni la respuesta ni el disco ---------------------

PROOF = {'inventory': {'observed': {'boot_id': '0' * 36}}, 'rows': [1, 2, 3]}
DIGEST = sha(PROOF)


def _refuse(*args, **kwargs):
    raise AssertionError('la recolección no ejecuta comandos ni cambia el servicio')


def _active(tmp_path):
    return {'boot_root': str(tmp_path / 'boot'), 'boot_id': '0' * 36, 'config': {'purpose': 'technical'}}


def _job(tmp_path, name, status, *, terminal=False):
    job = tmp_path / 'boot' / name
    (job / 'coded-P999').mkdir(parents=True)
    (job / 'result.json').write_text(json.dumps({'status': status, 'proof_sha256': DIGEST}), encoding='utf-8')
    (job / 'coded-P999' / 'complete.json').write_text(json.dumps(PROOF), encoding='utf-8')
    if terminal:
        (job / 'terminal.json').write_text(
            json.dumps({'host_evidence': {'status': 'INCOMPLETE_TERMINAL'}}), encoding='utf-8')
    return job


def _snapshot(folder: Path):
    return {str(path.relative_to(folder)): path.read_bytes() for path in sorted(folder.rglob('*')) if path.is_file()}


def _twice(dispatch, active, request, job):
    first = dispatch(active, request, execute=_refuse, controller=_refuse, maintenance=_refuse)
    after_first = _snapshot(job)
    second = dispatch(active, request, execute=_refuse, controller=_refuse, maintenance=_refuse)
    assert second == first
    assert _snapshot(job) == after_first
    return first


@pytest.mark.parametrize('request_', [{'operation': 'stimulus-evidence'},
                                      {'operation': 'stimulus-ack', 'proof_sha256': DIGEST}])
def test_guest_stimulus_collection_is_idempotent(tmp_path, request_):
    from scripts.study_operator.stimulus_guest import dispatch_stimulus
    job = _job(tmp_path, 'stimulus', 'BOOT_COMPLETE_UNANALYZED')
    result = _twice(dispatch_stimulus, _active(tmp_path), request_, job)
    assert result['proof_sha256'] == DIGEST


@pytest.mark.parametrize('request_', [{'operation': 'gate-evidence'},
                                      {'operation': 'gate-ack', 'proof_sha256': DIGEST},
                                      {'operation': 'gate-finalized', 'proof_sha256': DIGEST}])
def test_guest_gate_collection_is_idempotent(tmp_path, request_):
    from scripts.study_operator.i7_gate_guest import dispatch_gate
    job = _job(tmp_path, 'i7-gate', 'GATE_COMPLETE_UNANALYZED')
    result = _twice(dispatch_gate, _active(tmp_path), request_, job)
    assert result['proof_sha256'] == DIGEST


def test_guest_gate_terminal_evidence_is_idempotent(tmp_path):
    from scripts.study_operator.i7_gate_guest import dispatch_gate
    job = _job(tmp_path, 'i7-gate', 'INCOMPLETE_TERMINAL', terminal=True)
    result = _twice(dispatch_gate, _active(tmp_path), {'operation': 'gate-terminal-evidence'}, job)
    assert result['proof']['host_evidence']['status'] == 'INCOMPLETE_TERMINAL'


def test_guest_ack_with_a_different_hash_is_still_rejected(tmp_path):
    from scripts.study_operator.stimulus_guest import dispatch_stimulus
    job = _job(tmp_path, 'stimulus', 'BOOT_COMPLETE_UNANALYZED')
    with pytest.raises(ValueError, match='STIMULUS_DOWNLOAD_HASH_DIFFERS'):
        dispatch_stimulus(_active(tmp_path), {'operation': 'stimulus-ack', 'proof_sha256': 'f' * 64},
                          execute=_refuse, controller=_refuse, maintenance=_refuse)
    assert not (job / 'download-ack.json').exists()
