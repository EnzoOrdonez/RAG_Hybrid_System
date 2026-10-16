"""Gate must cross the durable app boundary, not only its inner query."""
import json
from types import SimpleNamespace

import pytest

from tests.study_helpers import configured


def test_gate_answer_times_durable_begin_and_backs_up_technical_session(tmp_path, monkeypatch):
    from scripts.study_operator.i7_gate_daemon import answer_slot
    from src.ui.components import study_service
    from src.ui.components.study_sessions import StudySession, StudyStore

    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path/'sessions', protocol, purpose='technical')
    store.freeze()
    elapsed = [0.]
    begin = StudySession.begin
    def durable_begin(session, *args):
        elapsed[0] += 7
        return begin(session, *args)
    monkeypatch.setattr(StudySession, 'begin', durable_begin)
    def factory(condition):
        elapsed[0] += 3
        return SimpleNamespace(query=lambda question: SimpleNamespace(
            answer='synthetic', error=None, confidence='LOW', sources=[],
            hallucination_report={'method':'nli'}))
    backups = []
    def backup(folder):
        data=json.loads((folder/'full_session.json').read_bytes())
        assert data['stage']=='abandoned'
        backups.append(folder)
        return {'status':'complete','sha256':'a'*64,'objects':{}}
    result = answer_slot(store, {'condition':'hybrid','query_id':protocol['config']['tasks']['T1'][0]},
                         1, factory, backup, clock=lambda:elapsed[0])
    assert result['elapsed_s']==pytest.approx(10)
    assert result['status']=='success' and len(backups)==1
    assert result['backup']['status']=='complete'
    assert study_service.answer.__module__=='src.ui.components.study_service'


def test_gate_answer_retains_failed_attempt_and_still_exports(tmp_path):
    from scripts.study_operator.i7_gate_daemon import answer_slot
    from src.ui.components.study_sessions import StudyStore
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path/'sessions', protocol, purpose='technical')
    store.freeze()
    result=answer_slot(store, {'condition':'hybrid','query_id':protocol['config']['tasks']['T1'][0]},
        1, lambda _: (_ for _ in ()).throw(ValueError('synthetic')), lambda _: {'status':'complete'})
    assert result['status']=='failed' and not result['valid']
    assert result['error']=='query_failed' and result['backup']['status']=='complete'
