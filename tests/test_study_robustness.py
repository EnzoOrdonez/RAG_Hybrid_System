"""Fault injection at query, checkpoint and live in-flight lock boundaries."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from filelock import Timeout
import pytest

from src.ui.components import session_storage, study_service as service
from src.ui.components.session_storage import SessionConflict, SessionStorageError
from tests.test_study_sessions import active as active, response


@pytest.mark.parametrize('failure', [ConnectionResetError, OSError, RuntimeError])
def test_ollama_failure_is_durable_technical_error_and_retry_is_explicit(active, failure):
    store, session, token = active
    session.familiarization_done()
    calls = []

    def query(_):
        calls.append('started')
        raise failure('DO_NOT_PERSIST_EXCEPTION_CONTENT')

    service.answer(session, lambda _: SimpleNamespace(query=query))
    assert calls == ['started']
    restored = store.admit(token)
    assert restored.pending['status'] == 'error'
    assert restored.pending['error'] == 'query_failed'
    assert restored.pending['decline_class'] is None
    assert 'DO_NOT_PERSIST' not in restored.path.read_text(encoding='utf-8')
    with pytest.raises(ValueError):
        restored.acknowledge()
    service.answer(restored, lambda _: SimpleNamespace(query=lambda q: response()))
    assert len(restored.data['attempts']) == 2
    with pytest.raises(SessionConflict):
        service.answer(restored, lambda _: pytest.fail('Must not send again'))


def test_reload_disconnect_and_double_submit_while_worker_is_alive(active):
    store, session, token = active
    session.familiarization_done()
    entered, release = Event(), Event()
    calls = []

    def query(_):
        calls.append('one')
        entered.set()
        assert release.wait(10)
        return response('Survives browser reconnection')

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service.answer, session, lambda _: SimpleNamespace(query=query))
        try:
            assert entered.wait(5)
            reloaded = store.admit(token)
            assert reloaded.pending['status'] == 'running'
            before = reloaded.path.read_bytes()
            with pytest.raises(Timeout):
                service.answer(reloaded, lambda _: pytest.fail('Duplicate request'))
            with pytest.raises(Timeout):
                service.recover(reloaded)
            assert reloaded.path.read_bytes() == before
        finally:
            release.set()
        future.result(timeout=5)
    reloaded = store.admit(token)
    assert reloaded.pending['answer'] == 'Survives browser reconnection'
    assert reloaded.pending['status'] == 'success'
    assert calls == ['one'] and len(reloaded.data['attempts']) == 1


@pytest.mark.parametrize('failure', [OSError(errno.ENOSPC, 'disk full'), PermissionError('denied')])
@pytest.mark.parametrize('flush', ['request', 'result', 'instrument'])
def test_failed_publication_preserves_checkpoint_and_rolls_back_memory(active, monkeypatch, failure, flush):
    from src.ui.components.study_protocol import LIKERT_IDS

    store, session, token = active
    session.familiarization_done()
    if flush == 'instrument':
        for _ in range(4):
            service.answer(session, lambda _: SimpleNamespace(query=lambda q: response()), 'free')
            session.shown()
            session.acknowledge()
    if flush == 'result':
        session.begin()
    before = session.path.read_bytes()
    state = json.loads(before)
    with monkeypatch.context() as patch:
        patch.setattr(session_storage.os, 'replace', lambda *_: (_ for _ in ()).throw(failure))
        with pytest.raises(OSError):
            if flush == 'request':
                session.begin()
            elif flush == 'result':
                session.finish(answer='UNCOMMITTED', elapsed_ms=12)
            else:
                session.submit_instruments([3] * 10, {k: 3 for k in LIKERT_IDS}, [4]*8)
    assert session.path.read_bytes() == before
    assert session.data == state
    assert not list(session.path.parent.glob('.pending-*'))
    reloaded = store.admit(token)
    if flush == 'result':
        service.recover(reloaded)
        assert reloaded.pending['error'] == 'interrupted'
        assert reloaded.pending['elapsed_ms'] is None
        assert len(reloaded.data['attempts']) == 1
    elif flush == 'request':
        reloaded.begin()
        assert len(reloaded.data['attempts']) == 1
    else:
        reloaded.submit_instruments([3] * 10, {k: 3 for k in LIKERT_IDS}, [4]*8)
        assert len(reloaded.data['instruments']) == 1


def test_wall_clock_jump_does_not_change_monotonic_query_duration(active, monkeypatch):
    _, session, _ = active
    session.familiarization_done()
    ticks = iter([10, 11, 14, 15])
    civil = [2_000_000_000]
    monkeypatch.setattr('src.ui.components.study_sessions.time.time', lambda: civil[0])

    def query(_):
        civil[0] -= 100_000
        return response()

    service.answer(session, lambda _: SimpleNamespace(query=query), clock=lambda: next(ticks))
    assert session.pending['finished_at'] < session.pending['started_at']
    assert session.pending['elapsed_ms'] == 5000


def test_result_flush_failure_is_not_mislabeled_as_ollama_error(active, monkeypatch):
    store, session, token = active
    session.familiarization_done()
    original = session_storage.os.replace
    publications = []

    def replace(source, target):
        if Path(target) == session.path:
            publications.append(target)
            if len(publications) == 2:
                raise OSError(errno.ENOSPC, 'full')
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(session_storage.os, 'replace', replace)
        with pytest.raises(OSError):
            service.answer(session, lambda _: SimpleNamespace(query=lambda q: response()))
    restored = store.admit(token)
    assert restored.pending['status'] == 'running'
    assert restored.pending['error'] is None
    assert len(restored.data['attempts']) == 1
    service.recover(restored)
    assert restored.pending['error'] == 'interrupted'


@pytest.mark.parametrize('point', ['directory', 'copy', 'publish'])
def test_interrupted_backup_is_pending_retryable_and_never_repeats_instruments(active, tmp_path, monkeypatch, point):
    from src.ui.components import study_backup as backup
    from tests.test_study_sessions import finish

    store, session, _ = active
    finish(session)
    export = session.export()
    original = export.read_bytes()
    destination = tmp_path / 'second-disk'
    real_mkdir, real_replace = Path.mkdir, Path.replace

    def mkdir(path, *args, **kwargs):
        if path == destination:
            raise PermissionError('denied')
        return real_mkdir(path, *args, **kwargs)

    def replace(path, target):
        if path.parent == destination:
            raise OSError(errno.ENOSPC, 'full')
        return real_replace(path, target)

    with monkeypatch.context() as patch:
        if point == 'directory':
            patch.setattr(Path, 'mkdir', mkdir)
        elif point == 'copy':
            patch.setattr(backup.shutil, 'copytree', lambda *_: (_ for _ in ()).throw(OSError('interrupted')))
        else:
            patch.setattr(Path, 'replace', replace)
        with pytest.raises(OSError):
            backup.backup_export(export.parent, destination, same_physical_disk=lambda *_: False)
    assert json.loads((export.parent / 'backup_state.json').read_text())['status'] == 'pending'
    with pytest.raises(SessionStorageError, match='backup'):
        store.issue('P02')
    state = backup.backup_export(export.parent, destination, same_physical_disk=lambda *_: False)
    assert state['status'] == 'complete'
    assert backup.backup_export(export.parent, destination, same_physical_disk=lambda *_: False) == state
    assert export.read_bytes() == original
    assert (Path(state['destination']) / export.name).read_bytes() == original
    assert len(session.data['instruments']) == 2


def test_changed_seal_blocks_query_before_model_or_new_attempt(active):
    store, session, _ = active
    session.familiarization_done()
    before = session.path.read_bytes()
    source = Path(store.protocol['paths']['config'])
    source.write_bytes(source.read_bytes() + b' ')
    with pytest.raises(SessionStorageError):
        service.answer(session, lambda _: pytest.fail('No model call with changed seal'))
    assert session.path.read_bytes() == before
