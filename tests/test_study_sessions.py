import json
import logging
from types import SimpleNamespace

import pytest

from src.ui.components.study_sessions import StudyStore, StudySession
from src.ui.components.study_protocol import LIKERT_IDS, load_protocol
from src.ui.components.session_storage import SessionConflict, SessionStorageError
from src.ui.components import study_service as service
from tests.study_helpers import configured


@pytest.fixture
def active(tmp_path):
    _, _, p = configured(tmp_path)
    store = StudyStore(tmp_path / 'sessions', p)
    store.freeze()
    token = store.issue('P01')
    return store, store.admit(token), token


def response(text='Saved exact answer'):
    return SimpleNamespace(answer=text, error=None, confidence='LOW', sources=[dict(provider='aws', service=None, section='N/A')],
                           hallucination_report={'method': 'nli'})


def complete_block(session):
    session.familiarization_done()
    for _ in range(4):
        service.answer(session, lambda _: SimpleNamespace(query=lambda q: response()), 'FREE_SENTINEL')
        session.shown()
        session.acknowledge()
    session.submit_instruments([3]*10, {k: 3 for k in LIKERT_IDS})


def finish(session):
    complete_block(session)
    complete_block(session)
    session.submit_comparative(dict(C1='A', C2='iguales', C3='ninguno', C4=''))
    session.submit_blinding('No sabría decir', '')


def test_complete_two_blocks_and_export_reconstructs_without_personal_fields(active):
    store, session, token = active
    finish(session)
    with pytest.raises(ValueError, match='Invitación'):
        store.admit(token)
    restored = StudySession.load(store, session.session_id)
    exported = json.loads(restored.export().read_text(encoding='utf-8'))
    assert len(exported['attempts']) == 8 and len(exported['instruments']) == 2
    assert [i['sus_score'] for i in exported['instruments']] == [50, 50]
    assert [i['condition'] for i in exported['instruments']] == ['hybrid', 'no_rag']
    assert sum(a['analysis_role'] == 'free_query' for a in exported['attempts']) == 2
    assert all(not a['sources'] for a in exported['attempts'] if a['condition'] == 'no_rag')
    def keys(v):
        if isinstance(v, dict):
            return set(v) | set().union(*(keys(x) for x in v.values()))
        if isinstance(v, list):
            return set().union(*(keys(x) for x in v))
        return set()
    assert not keys(exported) & {'email', 'name', 'employer', 'ip', 'consent_signature', 'token'}
    assert token not in json.dumps(exported)
    old = restored.export().read_bytes()
    assert restored.export().read_bytes() == old


def test_practice_never_persists_content_sources_latency_or_retry_error(tmp_path):
    path, csv_path, p = configured(tmp_path)
    c = p['config']
    c['familiarization'] = 'PRACTICE_QUESTION_SENTINEL'
    path.write_text(json.dumps(c), encoding='utf-8')
    store = StudyStore(tmp_path / 'sessions', load_protocol(path, csv_path))
    store.freeze()
    session = store.admit(store.issue('P01'))
    before = session.path.read_bytes()
    def query(q):
        assert q == 'PRACTICE_QUESTION_SENTINEL'
        logging.error('PRACTICE_ERROR_SENTINEL')
        print('PRACTICE_PRINT_SENTINEL')
        return response('PRACTICE_ANSWER_SENTINEL')
    assert service.practice(session, lambda _: SimpleNamespace(query=query))['answer'] == 'PRACTICE_ANSWER_SENTINEL'
    assert service.practice(session, lambda _: (_ for _ in ()).throw(ValueError('PRACTICE_ERROR_SENTINEL'))) is None
    assert session.path.read_bytes() == before
    session.incident()
    finish(session)
    for file in store.root.rglob('*.json'):
        data = file.read_text(encoding='utf-8')
        assert 'PRACTICE_' not in data
    events = [e for e in session.data['events'] if e['kind'] == 'familiarization_done']
    assert len(events) == 2 and all(set(e) == {'kind', 'timestamp', 'block_index'} for e in events)
    assert session.data['incidents'][0].keys() == {'kind', 'timestamp'}
    assert all(a['question'] != c['familiarization'] for a in session.data['attempts'])


def test_practice_error_restores_logging_and_never_records_retry(active):
    _, session, _ = active
    old = logging.root.manager.disable
    assert service.practice(session, lambda _: (_ for _ in ()).throw(RuntimeError())) is None
    assert logging.root.manager.disable == old and not session.data['incidents'] and not session.data['attempts']


def test_error_cannot_advance_and_reconnect_recovers_without_imputation(active):
    store, session, token = active
    session.familiarization_done()
    session.begin()
    session = store.admit(token)
    service.recover(session)
    assert session.pending['error'] == 'interrupted' and session.pending['elapsed_ms'] is None
    with pytest.raises(ValueError):
        session.acknowledge()
    service.answer(session, lambda _: SimpleNamespace(query=lambda q: response()))
    assert len(session.data['attempts']) == 2
    saved = store.admit(token)
    assert saved.pending['answer'] == 'Saved exact answer'
    with pytest.raises(SessionConflict):
        saved.begin()


def test_clock_includes_preparation_and_query_but_not_network(active):
    _, session, _ = active
    session.familiarization_done()
    elapsed = [0]
    def factory(_):
        elapsed[0] += 2
        def query(q):
            elapsed[0] += 3
            return response()
        return SimpleNamespace(query=query)
    service.answer(session, factory, clock=lambda: elapsed[0])
    assert session.pending['elapsed_ms'] == 5000


def test_source_placeholders_removed_without_editing_answer():
    r = response('Unchanged answer with None and N/A as literal subject matter')
    original = r.answer
    payload = service.presentation(r, 'hybrid')
    assert payload['answer'] == original and payload['sources'] == [{'label': 'aws'}]
    assert r.sources[0]['service'] is None
    assert service.presentation(r, 'no_rag')['sources'] == []
    assert service.presented_sources([dict(provider=None, service='N/A'), dict(url='javascript:bad')]) == []


def test_stale_tab_cannot_overwrite_state(active):
    store, session, _ = active
    other = StudySession.load(store, session.session_id)
    session.familiarization_done()
    with pytest.raises(SessionConflict):
        other.familiarization_done()


def test_no_switch_to_second_system_before_block_instruments(active):
    _, session, _ = active
    with pytest.raises(ValueError):
        session.submit_instruments([3]*10, {k: 3 for k in LIKERT_IDS})
    session.familiarization_done()
    assert session.data['block_index'] == 0
    with pytest.raises(ValueError):
        session.submit_blinding('Sistema A', '')


def test_frozen_files_cannot_change_after_admission(active):
    store, session, _ = active
    path = store.protocol['paths']['config']
    with open(path, 'a') as stream:
        stream.write(' ')
    with pytest.raises(SessionStorageError):
        session.familiarization_done()


def test_replacement_preserves_cell_profile_and_prevents_duplicate_slot(active):
    store, session, _ = active
    with pytest.raises(ValueError):
        store.issue('P21')
    with pytest.raises(ValueError):
        store.replace('P04', 'P21')  # wrong profile
    with pytest.raises(ValueError):
        store.replace('P01', 'P21')  # active original
    store.abandon(session.session_id)
    store.replace('P01', 'P21')
    reserve = store.admit(store.issue('P21'))
    assert reserve.data['assignment']['primary_slot'] == 'P01'
    with pytest.raises(ValueError):
        store.replace('P02', 'P21')
    with pytest.raises(ValueError):
        store.issue('P01')


def test_corrupt_checkpoint_and_unknown_fields_fail_closed(active):
    store, session, _ = active
    data = json.loads(session.path.read_text())
    data['email'] = 'not-an-address'
    session.path.write_text(json.dumps(data))
    before = session.path.read_bytes()
    with pytest.raises(SessionStorageError):
        StudySession.load(store, session.session_id)
    assert session.path.read_bytes() == before


def test_pilots_are_fictitious_and_separate(tmp_path):
    _, _, p = configured(tmp_path)
    store = StudyStore(tmp_path / 'pilot', p, purpose='pilot')
    store.freeze()
    with pytest.raises(ValueError):
        store.issue('P01', cell=1, profile='without_experience')
    session = store.admit(store.issue('P900', cell=1, profile='without_experience'))
    finish(session)
    assert json.loads(session.export().read_text())['purpose'] == 'pilot'
