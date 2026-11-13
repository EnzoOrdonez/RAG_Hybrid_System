"""Correcciones tras el intento 9 de la Fase B (ADR-0029).

privacy-09: el escáner del invitado no completaba caddy/app/runner con el almacén containerd de
Docker 29 (sin GraphDriver.UpperDir) ni backups con sockets. stimulus-09: la serie programada
arrancaba sin detener la VM que deja withdraw-purge. Transporte: un solo bridge SSH fallido en un
sondeo de solo lectura abortaba el paso completo.
"""
import errno
import json
import os
import stat
import subprocess

import pytest

from scripts.study_operator import i7_privacy
from scripts.study_operator.cloud_client import Cloud
from scripts.study_operator.policy import OperatorError, TransportUnavailable


# --- Transporte: clasificación en cloud_client -------------------------------------------------

def _cloud(tmp_path, invoke):
    return Cloud('gcloud-fixture', 'pure-loop-474323-a8', tmp_path, invoke=invoke)


def test_ssh_timeout_is_transport_unavailable_with_unchanged_message(tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('ssh', 1)
    cloud = _cloud(tmp_path, timeout)
    with pytest.raises(TransportUnavailable, match='Venció el límite'):
        cloud.command(['compute', 'ssh', 'vm', '--zone=us-central1-a', '--command=true'], json_output=False)


def test_non_ssh_timeout_stays_plain_operator_error(tmp_path):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('list', 1)
    cloud = _cloud(tmp_path, timeout)
    with pytest.raises(OperatorError) as caught:
        cloud.command(['compute', 'instances', 'list'])
    assert not isinstance(caught.value, TransportUnavailable)


@pytest.mark.parametrize('code,transport', [(255, True), (1, False)])
def test_only_ssh_connection_exit_255_is_transport(tmp_path, code, transport):
    cloud = _cloud(tmp_path, lambda *a, **k: subprocess.CompletedProcess([], code, b'', b'fixture'))
    with pytest.raises(OperatorError) as caught:
        cloud.command(['compute', 'ssh', 'vm', '--zone=us-central1-a', '--command=true'], json_output=False)
    assert isinstance(caught.value, TransportUnavailable) is transport


def test_exit_255_of_a_non_ssh_command_is_not_transport(tmp_path):
    cloud = _cloud(tmp_path, lambda *a, **k: subprocess.CompletedProcess([], 255, b'', b'fixture'))
    with pytest.raises(OperatorError) as caught:
        cloud.command(['compute', 'instances', 'list'])
    assert not isinstance(caught.value, TransportUnavailable)


@pytest.mark.skipif(os.name != 'nt', reason='la ruta RPC privada con OpenSSH nativo solo existe en Windows')
def test_private_rpc_rejection_is_transport_unavailable(tmp_path):
    cloud = _cloud(tmp_path, lambda *a, **k: subprocess.CompletedProcess([], 0, b'', b''))
    with pytest.raises(TransportUnavailable, match='El transporte SSH no pudo verificarse'):
        cloud.command(['compute', 'ssh', 'vm', '--zone=europe-west4-a', '--command=x'],
                      input_data=b'{}', json_output=False)


# --- Transporte: reintento acotado en bridge ----------------------------------------------------

class _SshCloud:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def command(self, argv, **kwargs):
        self.calls.append(argv)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _operator(tmp_path, outcomes):
    from tests.test_study_i7_operator import setup
    op, _, _ = setup(tmp_path)
    op.cloud = _SshCloud(outcomes)
    op.config['host_code'] = '/srv/cloudrag/i7/service/code'
    sleeps = []
    op.sleep = sleeps.append
    return op, sleeps


OK = json.dumps({'status': 'OK', 'result': {'status': 'RUNNING'}}).encode()


def _transport():
    return TransportUnavailable('El transporte SSH no pudo verificarse. Conserva runs.')


def test_read_only_poll_retries_transport_and_then_succeeds(tmp_path):
    op, sleeps = _operator(tmp_path, [_transport(), _transport(), OK])
    assert op.bridge({'operation': 'stimulus-status'}, private=True)['result']['status'] == 'RUNNING'
    assert len(op.cloud.calls) == 3 and sleeps == [15, 45]


def test_read_only_poll_gives_up_after_bounded_attempts(tmp_path):
    op, sleeps = _operator(tmp_path, [_transport(), _transport(), _transport()])
    with pytest.raises(TransportUnavailable):
        op.bridge({'operation': 'gate-status'}, private=True)
    assert len(op.cloud.calls) == 3 and sleeps == [15, 45]


@pytest.mark.parametrize('operation', ['stimulus-start', 'gate-start', 'i7-privacy', 'stop', 'restore',
                                       'stimulus-ack', 'gate-finalized', 'i7-contexts', 'export'])
def test_operations_with_effects_never_retry(tmp_path, operation):
    op, sleeps = _operator(tmp_path, [_transport(), OK])
    with pytest.raises(TransportUnavailable):
        op.bridge({'operation': operation}, private=True)
    assert len(op.cloud.calls) == 1 and sleeps == []


def test_non_transport_errors_never_retry(tmp_path):
    op, sleeps = _operator(tmp_path, [OperatorError('Google Cloud rechazó la operación.'), OK])
    with pytest.raises(OperatorError):
        op.bridge({'operation': 'stimulus-status'}, private=True)
    assert len(op.cloud.calls) == 1 and sleeps == []


def test_retryable_operations_are_exactly_the_read_only_polls():
    from scripts.study_operator.lifecycle import RETRYABLE_BRIDGE_OPERATIONS, TRANSPORT_RETRY_DELAYS_S
    assert RETRYABLE_BRIDGE_OPERATIONS == frozenset(
        {'stimulus-status', 'gate-status', 'i7-smoke-observe', 'preflight', 'i7-preparation-check'})
    assert TRANSPORT_RETRY_DELAYS_S == (15, 45)


# --- stimulus-09: la serie programada detiene la VM y aplica el perfil primero -------------------

def test_scheduled_series_applies_stimulus_profile_before_first_boot(tmp_path, monkeypatch):
    from scripts.study_operator import i7_phase_b, i7_workflows
    calls = []
    monkeypatch.setattr(i7_phase_b, 'switch_profile', lambda op, kind: calls.append(('switch_profile', kind)))
    monkeypatch.setattr(i7_workflows, 'series', lambda op: calls.append(('series',)) or 'SERIES')
    assert i7_workflows.dispatch(type('Op', (), {'root': tmp_path})(), 'series') == 'SERIES'
    assert calls == [('switch_profile', 'stimulus'), ('series',)]


# --- privacy-09: capa escribible con containerd y archivos especiales ---------------------------

def _docker(tmp_path, *, graph=None, mounts_text=None, ctr_fails=False, names=('cloudrag-i4-caddy-x',)):
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ['docker', 'ps']:
            return subprocess.CompletedProcess(argv, 0, '\n'.join(names).encode())
        if argv[:2] == ['docker', 'inspect']:
            return subprocess.CompletedProcess(argv, 0, json.dumps(
                [{'Id': 'c0ffee' * 10 + 'abcd', 'Mounts': [], 'GraphDriver': graph, 'LogPath': ''}]).encode())
        if argv[:1] == ['ctr']:
            if ctr_fails:
                return subprocess.CompletedProcess(argv, 1, b'')
            return subprocess.CompletedProcess(argv, 0, mounts_text.encode())
        raise AssertionError(argv)
    return invoke, calls


def _upper(tmp_path, content=b'CANARY-IN-UPPER'):
    upper = tmp_path / 'snapshots' / '1018' / 'fs'
    (upper / 'tmp').mkdir(parents=True)
    (upper / 'tmp' / 'note.txt').write_bytes(content)
    return upper


def _collect(provider):
    return [(locator, b''.join(blocks)) for locator, blocks in provider()]


def test_containerd_snapshotter_resolves_writable_layer_by_container_id(tmp_path):
    upper = _upper(tmp_path)
    text = f'mount -t overlay overlay /nonexistent -o workdir=/w,upperdir={upper.as_posix()},lowerdir=/l1:/l2\n'
    invoke, calls = _docker(tmp_path, graph=None, mounts_text=text)
    found = _collect(i7_privacy.host_sources({'boot_root': str(tmp_path / 'b' / 'boots' / '1')}, invoke=invoke)['caddy'])
    assert any(b'CANARY-IN-UPPER' in data for _, data in found)
    ctr = [c for c in calls if c[0] == 'ctr']
    assert ctr == [['ctr', '-n', 'moby', 'snapshots', '--snapshotter', 'overlayfs', 'mounts',
                    '/nonexistent', 'c0ffee' * 10 + 'abcd']]


def test_overlay2_upperdir_is_used_without_containerd(tmp_path):
    upper = _upper(tmp_path)
    invoke, calls = _docker(tmp_path, graph={'Name': 'overlay2', 'Data': {'UpperDir': str(upper)}})
    found = _collect(i7_privacy.host_sources({'boot_root': str(tmp_path / 'b' / 'boots' / '1')}, invoke=invoke)['caddy'])
    assert any(b'CANARY-IN-UPPER' in data for _, data in found)
    assert not [c for c in calls if c[0] == 'ctr']


@pytest.mark.parametrize('text', ['mount -t overlay overlay /x -o workdir=/w,lowerdir=/l\n',
                                  'upperdir=/a upperdir=/b\n', ''])
def test_missing_or_ambiguous_upperdir_fails_closed(tmp_path, text):
    invoke, _ = _docker(tmp_path, graph=None, mounts_text=text)
    with pytest.raises(ValueError, match='CONTAINER_WRITABLE_LAYER_UNRESOLVED'):
        _collect(i7_privacy.host_sources({'boot_root': str(tmp_path / 'b' / 'boots' / '1')}, invoke=invoke)['caddy'])


def test_failed_containerd_query_fails_closed(tmp_path):
    invoke, _ = _docker(tmp_path, graph=None, ctr_fails=True)
    with pytest.raises(ValueError, match='PRIVACY_SOURCE_COMMAND_FAILED'):
        _collect(i7_privacy.host_sources({'boot_root': str(tmp_path / 'b' / 'boots' / '1')}, invoke=invoke)['caddy'])


def _fake_types(monkeypatch, kinds, links=None):
    real = os.lstat

    class Fake:
        def __init__(self, mode, rdev=0):
            self.st_mode, self.st_rdev = mode, rdev

    def lstat(path):
        name = os.path.basename(str(path))
        if name in kinds:
            mode, rdev = kinds[name]
            return Fake(mode, rdev)
        return real(path)
    monkeypatch.setattr(i7_privacy, '_lstat', lstat)
    monkeypatch.setattr(i7_privacy, '_readlink', lambda path: (links or {})[os.path.basename(str(path))])


def _tree(tmp_path, *names):
    root = tmp_path / 'tree'
    root.mkdir()
    (root / 'regular.txt').write_bytes(b'regular CANARY')
    for name in names:
        (root / name).write_bytes(b'')
    return root


def test_sockets_and_fifos_are_counted_without_reading(tmp_path, monkeypatch):
    root = _tree(tmp_path, 'backup.sock', 'pipe')
    _fake_types(monkeypatch, {'backup.sock': (stat.S_IFSOCK, 0), 'pipe': (stat.S_IFIFO, 0)})
    found = dict(_collect(lambda: i7_privacy.tree_sources(root)))
    assert found[str(root / 'regular.txt')] == b'regular CANARY'
    assert found[str(root / 'backup.sock') + '#socket'] == b''
    assert found[str(root / 'pipe') + '#fifo'] == b''


def test_symlink_text_is_scanned_without_following(tmp_path, monkeypatch):
    root = _tree(tmp_path, 'libcuda.so.1')
    _fake_types(monkeypatch, {'libcuda.so.1': (stat.S_IFLNK, 0)}, links={'libcuda.so.1': 'CANARY-TARGET-NAME'})
    found = dict(_collect(lambda: i7_privacy.tree_sources(root)))
    assert found[str(root / 'libcuda.so.1') + '@symlink'] == b'CANARY-TARGET-NAME'


def test_overlay_whiteout_is_counted_but_other_devices_fail_closed(tmp_path, monkeypatch):
    root = _tree(tmp_path, 'deleted')
    _fake_types(monkeypatch, {'deleted': (stat.S_IFCHR, 0)})
    found = dict(_collect(lambda: i7_privacy.tree_sources(root)))
    assert found[str(root / 'deleted') + '#whiteout'] == b''
    _fake_types(monkeypatch, {'deleted': (stat.S_IFCHR, 0x0501)})
    with pytest.raises(ValueError, match='SCAN_SPECIAL_FILE_REQUIRES_REVIEW'):
        _collect(lambda: i7_privacy.tree_sources(root))
    _fake_types(monkeypatch, {'deleted': (stat.S_IFBLK, 0)})
    with pytest.raises(ValueError, match='SCAN_SPECIAL_FILE_REQUIRES_REVIEW'):
        _collect(lambda: i7_privacy.tree_sources(root))


def test_symlinked_scan_root_still_fails_closed(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setattr(i7_privacy, '_lstat', lambda path: type('S', (), {'st_mode': stat.S_IFLNK, 'st_rdev': 0})())
    with pytest.raises(ValueError, match='SCAN_ROOT_REQUIRED'):
        _collect(lambda: i7_privacy.tree_sources(root))


@pytest.mark.parametrize('error,code', [
    (ValueError('CONTAINER_WRITABLE_LAYER_UNRESOLVED'), 'CONTAINER_WRITABLE_LAYER_UNRESOLVED'),
    (ValueError('/srv/private/path leaked text'), 'UNCLASSIFIED'),
    (OSError(errno.ENXIO, 'No such device or address', '/srv/private/boots/x/sockets/a.sock'), 'ENXIO'),
    (KeyError('secret-key-name'), 'UNCLASSIFIED')])
def test_scan_records_fixed_error_code_without_private_text(error, code):
    def broken():
        raise error
        yield  # pragma: no cover
    sources = {domain: (lambda: iter(())) for domain in i7_privacy.DOMAINS}
    sources['backups'] = broken
    result = i7_privacy.scan(sources, (b'canary-ip', b'canary-ua'))
    assert result['status'] == 'INCOMPLETE'
    assert result['domains']['backups']['error_code'] == code
    assert 'private' not in json.dumps(result) and 'secret' not in json.dumps(result)


# --- Revisión independiente: un boot a medias se rechaza antes de detener nada ------------------

def _journal(tmp_path, complete):
    journal = tmp_path / 'attempts'
    journal.mkdir()
    (journal / 'stimulus-01.json').write_text(json.dumps({'status': 'INTENT', 'boot_index': 1}))
    if complete:
        (journal / 'stimulus-01.complete.json').write_text(json.dumps({'status': 'COMPLETE', 'boot_index': 1}))
    return type('Op', (), {'root': tmp_path})()


@pytest.mark.parametrize('module_step', [('i7_workflows', 'series'), ('i7_phase_b', 'stimulus')])
def test_unfinished_boot_is_refused_before_switching_profile(tmp_path, monkeypatch, module_step):
    from scripts.study_operator import i7_phase_b, i7_workflows
    calls = []
    monkeypatch.setattr(i7_phase_b, 'switch_profile', lambda op, kind: calls.append('switch_profile'))
    monkeypatch.setattr(i7_workflows, 'series', lambda op: calls.append('series'))
    op = _journal(tmp_path, complete=False)
    with pytest.raises(ValueError, match='TERMINAL_STIMULUS_REQUIRES_HUMAN_REVIEW'):
        if module_step[0] == 'i7_workflows':
            i7_workflows.dispatch(op, 'series')
        else:
            i7_phase_b.dispatch(op, 'stimulus', None)
    assert calls == []


def test_completed_boots_still_continue_the_series(tmp_path, monkeypatch):
    from scripts.study_operator import i7_phase_b, i7_workflows
    calls = []
    monkeypatch.setattr(i7_phase_b, 'switch_profile', lambda op, kind: calls.append('switch_profile'))
    monkeypatch.setattr(i7_workflows, 'series', lambda op: calls.append('series'))
    i7_workflows.dispatch(_journal(tmp_path, complete=True), 'series')
    assert calls == ['switch_profile', 'series']


# --- Revisión independiente: ramas del escáner sin cubrir ----------------------------------------

def test_symlinked_directory_is_scanned_as_link_text(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    (root / 'linked-dir').mkdir()
    real = os.lstat

    def lstat(path):
        if os.path.basename(str(path)) == 'linked-dir':
            return type('S', (), {'st_mode': stat.S_IFLNK, 'st_rdev': 0})()
        return real(path)
    monkeypatch.setattr(i7_privacy, '_lstat', lstat)
    monkeypatch.setattr(i7_privacy, '_readlink', lambda path: 'CANARY-DIR-TARGET')
    found = dict(_collect(lambda: i7_privacy.tree_sources(root)))
    assert found[str(root / 'linked-dir') + '@symlink'] == b'CANARY-DIR-TARGET'


def test_linked_archive_member_fails_closed(tmp_path):
    import io
    import tarfile
    root = tmp_path / 'tree'
    root.mkdir()
    with tarfile.open(root / 'backup.tar', 'w') as archive:
        member = tarfile.TarInfo('link')
        member.type, member.linkname = tarfile.SYMTYPE, '/etc/passwd'
        archive.addfile(member)
        data = tarfile.TarInfo('data.txt')
        data.size = 6
        archive.addfile(data, io.BytesIO(b'CANARY'))
    with pytest.raises(ValueError, match='ARCHIVE_LINK_REQUIRES_REVIEW'):
        _collect(lambda: i7_privacy.tree_sources(root))


def test_unreadable_directory_fails_closed(tmp_path, monkeypatch):
    root = _tree(tmp_path)

    def walk(top, onerror=None, followlinks=False):
        onerror(PermissionError(errno.EACCES, 'denied', str(top)))
        yield from ()
    monkeypatch.setattr(i7_privacy.os, 'walk', walk)
    with pytest.raises(PermissionError):
        _collect(lambda: i7_privacy.tree_sources(root))
