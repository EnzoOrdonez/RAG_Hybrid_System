"""Preparation requires real work, identity continuity, and live model residency."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.ui.components import interview_preparation as prep


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    monkeypatch.setenv('CLOUDRAG_ARTIFACT_MANIFEST', str(manifest))
    monkeypatch.setenv('CLOUDRAG_BUILD_ID', 'a' * 40)
    monkeypatch.setenv('CLOUDRAG_MODEL_DIGEST', 'b' * 64)
    calls = []
    resident = {'models': [dict(name='granite4.1:8b', digest='b' * 64,
        expires_at=datetime.fromtimestamp(5000, timezone.utc).isoformat(), context_length=4096)]}
    def factory(system):
        def query(question):
            calls.append(('query', system))
            return SimpleNamespace(model_dump=lambda **kw: dict(answer='Declined', error=None,
                confidence='LOW', hallucination_report={'method': 'decline'}))
        def predict(*args, **kw):
            calls.append(('nli', system))
            assert kw['apply_softmax']
            return [[.1, .8, .1]]
        return SimpleNamespace(query=query, hallucination_detector=SimpleNamespace(
            nli_model=SimpleNamespace(predict=predict)))
    obj = prep.Preparation(tmp_path / 'operations', factory=factory,
                           probe=lambda: resident, clock=lambda: 1000)
    return obj, calls, resident


def test_all_pipelines_and_nli_are_exercised_even_for_declines(prepared):
    obj, calls, _ = prepared
    receipt = obj.prepare('P900-session')
    assert calls == [(stage, system) for system in prep.SYSTEMS for stage in ('query', 'nli')]
    assert obj.ready('P900-session')
    assert set(obj.pipelines) == set(prep.SYSTEMS)
    assert receipt['status'] == 'ready'
    assert not obj.ready('another-session')


def test_residency_loss_invalidates_receipt_without_silent_warmup(prepared):
    obj, calls, resident = prepared
    obj.prepare('session')
    resident['models'] = []
    assert not obj.ready('session')
    assert len(calls) == 6
    assert obj.receipt is None
    assert any(json.loads(p.read_text())['kind'] == 'preparation_invalidated' for p in obj.root.glob('*.json'))


def test_new_process_object_cannot_load_an_old_receipt(prepared):
    obj, _, _ = prepared
    obj.prepare('session')
    other = prep.Preparation(obj.root, factory=obj.factory, probe=obj.probe, clock=obj.clock)
    assert not other.ready('session')


@pytest.mark.parametrize('change', ['digest', 'build', 'expiry', 'context'])
def test_identity_expiry_and_context_drift_reject_readiness(prepared, monkeypatch, change):
    obj, _, resident = prepared
    obj.prepare('session')
    if change == 'digest':
        resident['models'][0]['digest'] = 'c' * 64
    elif change == 'build':
        monkeypatch.setenv('CLOUDRAG_BUILD_ID', 'd' * 40)
    elif change == 'expiry':
        resident['models'][0]['expires_at'] = datetime.fromtimestamp(1100, timezone.utc).isoformat()
    else:
        resident['models'][0]['context_length'] = 8192
    assert not obj.ready('session')


def test_nli_failure_preserves_started_and_failed_records(prepared):
    obj, _, _ = prepared
    obj.factory = lambda key: SimpleNamespace(query=lambda q: SimpleNamespace(model_dump=lambda **kw:
        dict(answer='ok', error=None, confidence='HIGH', hallucination_report={'method': 'nli'})),
        hallucination_detector=SimpleNamespace(nli_model=None))
    with pytest.raises(prep.PreparationRequired):
        obj.prepare('session')
    records = [json.loads(p.read_text()) for p in obj.root.glob('*.json')]
    assert {r['kind'] for r in records} >= {'preparation_started', 'preparation_failed', 'warmup_response'}
    assert not obj.ready('session')


def test_preparation_is_idempotent_for_same_session_but_repeated_for_new_one(prepared):
    obj, calls, _ = prepared
    original = obj.prepare('session1')
    assert obj.prepare('session1') == original
    assert len(calls) == 6
    new = obj.prepare('session2')
    assert len(calls) == 12 and new['id'] != original['id']
    assert not obj.ready('session1')


def test_manifest_change_cannot_reuse_readiness(prepared, monkeypatch):
    obj, _, _ = prepared
    obj.prepare('session')
    from pathlib import Path
    import os
    Path(os.environ['CLOUDRAG_ARTIFACT_MANIFEST']).write_text('{"changed":true}')
    assert not obj.ready('session')
    with pytest.raises(prep.PreparationRequired, match='Restart process'):
        obj.prepare('session')
