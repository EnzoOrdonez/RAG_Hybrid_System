"""Prospective warm acceptance must not relabel historical cohorts or hide failures."""
import pytest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from scripts import measure_interview_gate as gate


def protocol():
    return dict(protocol_version=2, systems=list(gate.SYSTEMS),
        http_timeouts={'read': 180, 'connect': 5, 'write': 60, 'pool': 60},
        keep_alive='30m', warm_preparation='all_three_in_process',
        acceptance='prepared_warm_three_systems_v1')


def records():
    return [dict(system=s, phase=p, index=i, warmup=False, consumes_slot=True,
                 status='success', elapsed_s=83 if p == 'cold' else 40)
            for s in gate.SYSTEMS for p in gate.PHASES for i in range(20)]


def test_warm_policy_does_not_relabel_old_cold_failure():
    rows = records()
    assert not gate.summarize(rows)['passed']
    new = gate.summarize(rows, protocol=protocol())
    assert new['passed']
    assert new['cells'][0]['p95_s'] == 83


@pytest.mark.parametrize('status,invalid', [('error', False), ('aborted', False), ('success', True)])
def test_one_bad_warm_position_blocks_even_when_success_p95_is_fast(status, invalid):
    rows = records()
    rows[20].update(status=status, conditions_invalid=invalid)
    assert not gate.summarize(rows, protocol=protocol())['passed']


def test_unmeasured_cold_still_prevents_claim_of_completed_cohort():
    assert not gate.summarize(records()[1:], protocol=protocol())['passed']


def test_cold_errors_are_disclosed_without_changing_warm_criterion():
    rows = records()
    rows[0].update(status='error', elapsed_s=None)
    report = gate.summarize(rows, protocol=protocol())
    assert report['passed'] and report['cells'][0]['failures'] == 1
    assert report['cells'][0]['successes'] == 19


def test_recipe_rejects_unregistered_timeout_drift():
    current = protocol()
    assert gate.recipe(current)['read'] == 180
    current['http_timeouts']['read'] = 60
    with pytest.raises(ValueError, match='recipe'):
        gate.recipe(current)
    assert gate.recipe({})['read'] == 60


def test_condition_selection_never_expands_scope():
    assert gate.selected_conditions(protocol(), 'lexical', 'warm') == [('lexical', 'warm')]
    with pytest.raises(ValueError):
        gate.selected_conditions(protocol(), 'lexical', None)
    with pytest.raises(ValueError):
        gate.selected_conditions(dict(protocol(), systems=['hybrid']), 'lexical', 'warm')


def test_deadline_reserves_time_without_inventing_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv('CLOUDRAG_GATE_DEADLINE', datetime.now(timezone.utc).isoformat())
    gate.write_new(tmp_path / 'source-manifest.json', dict(protocol=protocol()))
    gate.worker(tmp_path, 'lexical', 'warm', [0])
    assert not gate.local_records(tmp_path)
    assert gate.read_json(next((tmp_path / 'pauses').glob('*.json')))['next_index'] == -1
    monkeypatch.setenv('CLOUDRAG_GATE_DEADLINE', (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
    assert gate.window_has_margin(900)


@pytest.mark.parametrize('runtime_cuda', [None, '12.6'])
def test_warm_worker_prepares_three_and_persists_readiness_per_response(
    tmp_path, monkeypatch, runtime_cuda
):
    import torch
    from src.ui.components import interview_preparation as prep, index_loader
    # The fake pipelines exercise the legacy CPU-only worker on either platform.
    # A CUDA build must still be rejected before any fake pipeline is loaded.
    monkeypatch.setattr(torch.version, 'cuda', runtime_cuda)
    monkeypatch.delenv('CLOUDRAG_GATE_DEADLINE', raising=False)
    artifact = tmp_path / 'artifact.json'
    artifact.write_text('{}')
    monkeypatch.setenv('CLOUDRAG_ARTIFACT_MANIFEST', str(artifact))
    monkeypatch.setenv('CLOUDRAG_BUILD_ID', 'build')
    monkeypatch.setenv('CLOUDRAG_MODEL_DIGEST', 'b' * 64)
    monkeypatch.setattr(gate, 'check_model', lambda: 'granite4.1:8b')
    monkeypatch.setattr(gate, 'check_environment', lambda p: None)
    resident = {'models': [dict(name='granite4.1:8b', digest='b' * 64, context_length=4096,
        expires_at=(datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat())]}
    original = prep.Preparation
    monkeypatch.setattr(prep, 'Preparation', lambda root, **kw: original(root, probe=lambda: resident, **kw))
    calls = []
    def factory(key, **kwargs):
        def query(q):
            calls.append((key, q))
            return SimpleNamespace(model_dump=lambda **kw: dict(answer='ok', confidence='HIGH', error=None))
        return SimpleNamespace(query=query, config=SimpleNamespace(model_dump=lambda **kw: {}),
            hybrid_index=SimpleNamespace(deployment_manifest_sha256='manifest'),
            hallucination_detector=SimpleNamespace(nli_model=SimpleNamespace(predict=lambda *a, **k: [[.1,.8,.1]])),
            llm=SimpleNamespace(cache_enabled=False, seed=42, timeout=60, max_retries=1,
                read_timeout=180, default_keep_alive='30m', num_ctx=4096, model_digest='b' * 64))
    monkeypatch.setattr(index_loader, 'load_pipeline', factory)
    monkeypatch.setattr(index_loader, 'load_hybrid_index', lambda: None)
    p = dict(protocol(), queries=[{'question': 'Measured query'}] * 20, build_id='build', model_digest='b' * 64)
    gate.write_new(tmp_path / 'source-manifest.json', dict(protocol=p, abort_consumes_slot=True))
    if runtime_cuda is not None:
        with pytest.raises(RuntimeError, match='Warmup failed'):
            gate.worker(tmp_path, 'semantic', 'warm', [0, 1])
        rows = gate.local_records(tmp_path)
        assert len(rows) == 1 and rows[0]['status'] == 'error'
        assert 'CPU-only auxiliary runtime required' in rows[0]['error']
        assert calls == []
        return
    gate.worker(tmp_path, 'semantic', 'warm', [0, 1])
    rows = sorted(gate.local_records(tmp_path), key=lambda r: r['index'])
    assert [r['status'] for r in rows] == ['success'] * 3
    assert calls == [(s, prep.WARM_QUERY) for s in prep.SYSTEMS] + [('semantic', 'Measured query')] * 2
    assert rows[0]['warmup_kind'] == 'all_system_preparation'
    assert all(r['http_timeouts']['read'] == 180 for r in rows)
    assert all(r['preparation_check']['resident'] == resident for r in rows[1:])
    assert len({r['preparation_id'] for r in rows[1:]}) == 1
