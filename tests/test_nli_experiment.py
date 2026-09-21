"""No real inference, inventory or service changes; registered NLI experiment contracts."""
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from scripts import measure_interview_gate as gate
from scripts import nli_batch_experiment as experiment
from scripts import unattended_diagnostic as unattended


def rows():
    result = []
    for position, (system, index, arm) in enumerate(experiment.schedule()):
        row = unattended.synthetic_row(system, index)
        row.update(arm=arm, window_id='same', started_at=f'{position:04d}', attempt_id=f'a{position}')
        result.append(row)
    return result


def publish(root, row):
    gate.write_new(root / 'attempts' / row['attempt_id'] / 'result.json', row)


def test_registered_calendar_120_balanced_order_and_rotating_systems():
    slots = experiment.schedule()
    assert len(slots) == len(set(slots)) == 120
    for system in experiment.SYSTEMS:
        first = [slots[i][2] for i in range(0, 120, 2) if slots[i][0] == system]
        assert first.count('control') == first.count('candidate') == 10
    for i in range(20):
        assert slots[6*i][0] == experiment.SYSTEMS[i % 3]
    assert all(slots[i][:2] == slots[i+1][:2] for i in range(0, 120, 2))


@pytest.mark.parametrize('kind', ['duplicate', 'foreign_arm', 'cold', 'gap'])
def test_calendar_rejects_corruption(kind):
    data = rows()[:4]
    if kind == 'duplicate':
        data[1] = data[0]
    elif kind == 'foreign_arm':
        data[0]['arm'] = 'other'
    elif kind == 'cold':
        data[0]['phase'] = 'cold'
    else:
        data.pop(0)
    with pytest.raises(ValueError):
        experiment.remaining(data)


def test_unresolved_interrupted_pair_refuses_mate_across_windows():
    with pytest.raises(RuntimeError, match='Interrupted pair'):
        experiment.resume_boundary(rows()[:1])
    experiment.resume_boundary(rows()[:2])


def test_summary_forbids_cross_window_pairs():
    data = rows()[:2]
    data[1]['window_id'] = 'other'
    with pytest.raises(ValueError, match='crosses'):
        experiment.summarize(data)


def test_failures_invalid_and_energy_do_not_enter_percentiles():
    data = rows()
    data[0].update(status='error', elapsed_s=900)
    data[1].update(status='aborted', elapsed_s=None)
    data[2].update(conditions_invalid=True, elapsed_s=800)
    report = experiment.summarize(data)
    assert report['complete'] and not report['confirmation_ready']
    assert sum(g['failures'] for g in report['groups']) == 2
    assert sum(g['invalid'] for g in report['groups']) == 1
    assert all(g['metrics']['total_s']['p95'] == .001 for g in report['groups'])
    energy = experiment.summarize(data, energy=True)
    assert not energy['complete'] and not energy['latency_pass_candidate']
    assert all(g['valid'] == 0 for g in energy['groups'])


def test_candidate_latency_threshold_and_control_reported_separately():
    data = rows()
    for r in data:
        r['elapsed_s'] = 70 if r['arm'] == 'control' else 60
    assert experiment.summarize(data)['latency_pass_candidate']
    for r in data:
        if r['arm'] == 'candidate' and r['system'] == 'hybrid':
            r['elapsed_s'] = 60.001
    assert not experiment.summarize(data)['latency_pass_candidate']


def test_same_calendar_resumes_at_block_without_duplicates(tmp_path):
    calls = []
    data = rows()
    def work(system, index, arm, window_id):
        pos = experiment.schedule().index((system, index, arm))
        row = dict(data[pos], window_id=window_id)
        calls.append((system, index, arm))
        publish(tmp_path, row)
        return row
    experiment.execute_slots(tmp_path, work, window_id='one', margin=lambda _: len(calls) < 12)
    assert len(calls) == 12
    first = {p: p.read_bytes() for p in tmp_path.rglob('result.json')}
    experiment.execute_slots(tmp_path, work, window_id='two', margin=lambda _: True)
    assert calls == experiment.schedule()
    assert all(p.read_bytes() == value for p, value in first.items())
    assert len(experiment.summarize(gate.local_records(tmp_path))['pairs']) == 60


def test_expired_window_never_invokes_measure(tmp_path):
    called = []
    experiment.execute_slots(tmp_path, lambda *args: called.append(args), window_id='expired', margin=lambda _: False)
    assert not called and not gate.local_records(tmp_path)


def test_ordinary_invalid_does_not_replace_or_stop_calendar(tmp_path):
    data = rows()
    data[0].update(conditions_invalid=True, control_reasons=['prohibited_process'])
    iterator = iter(data)
    def work(*args):
        row = next(iterator)
        publish(tmp_path, row)
        return row
    experiment.execute_slots(tmp_path, work, window_id='same', margin=lambda _: True)
    report = experiment.summarize(gate.local_records(tmp_path))
    assert report['complete'] and not report['confirmation_ready']
    assert sum(g['invalid'] for g in report['groups']) == 1


def test_technical_failure_durable_and_stops(tmp_path):
    data = rows()[0]
    data.update(status='error', error='batch:TimeoutError')
    def work(*args):
        publish(tmp_path, data)
        return data
    with pytest.raises(RuntimeError, match='Technical failure'):
        experiment.execute_slots(tmp_path, work, window_id='same', margin=lambda _: True)
    assert gate.local_records(tmp_path)[0]['error'] == 'batch:TimeoutError'


def test_quality_equivalence_uses_same_input_reports_and_raw_scores():
    detector, response = experiment.synthetic_response()
    original = copy.deepcopy(response)
    result = experiment.equivalence(detector, response)
    assert result['passed'] and result['matches_recorded']
    assert len(result['arms']['control']['calls']) == 2
    assert len(result['arms']['candidate']['calls']) == 1
    assert len(result['arms']['candidate']['calls'][0]['raw_scores']) == 2
    assert detector.nli_pair_schedule == 'per_claim'
    assert response == original


def test_quality_rejects_changed_public_score_without_new_tolerance():
    detector, response = experiment.synthetic_response()
    response['hallucination_report']['faithfulness_score'] = .9999
    result = experiment.equivalence(detector, response)
    assert result['identical'] and not result['passed'] and not result['matches_recorded']


def test_quality_rejects_batch_error_despite_sequential_recovery():
    detector, response = experiment.synthetic_response()
    predict = detector.nli_model.predict
    def fail_batch(sentences, **kwargs):
        if len(sentences) > 1:
            raise TimeoutError('synthetic')
        return predict(sentences, **kwargs)
    detector.nli_model.predict = fail_batch
    result = experiment.equivalence(detector, response)
    assert not result['healthy'] and not result['passed']
    assert result['arms']['candidate']['batch_error'] == 'TimeoutError'


def test_quality_checkpoint_idempotent_and_inputs_hash_bound(tmp_path, monkeypatch):
    detector, response = experiment.synthetic_response()
    row = dict(response=response)
    experiment.quality_check(tmp_path, 'new-a', row, detector)
    content = (tmp_path / 'quality/new-a.json').read_bytes()
    monkeypatch.setattr(experiment, 'equivalence', lambda *args: pytest.fail('checkpoint re-executed'))
    experiment.quality_check(tmp_path, 'new-a', row, detector)
    assert (tmp_path / 'quality/new-a.json').read_bytes() == content
    row['response']['answer'] = 'changed'
    with pytest.raises(ValueError, match='input changed'):
        experiment.quality_check(tmp_path, 'new-a', row, detector)


def test_quality_inventory_requires_exact_approved_source_ids(tmp_path):
    gate.write_new(tmp_path / 'source-manifest.json', dict(protocol=dict(quality_source_ids=[str(i) for i in range(40)])))
    for i in range(40):
        gate.write_new(tmp_path / 'quality' / f'base-wrong-{i}.json', dict(passed=True))
    assert experiment.quality_summary(tmp_path, [])['base_passed'] == 0


def test_raw_score_capture_only_opt_in_and_probe_restores(monkeypatch):
    from scripts.lexical_diagnostic import PipelineTrace
    detector, _ = experiment.synthetic_response()
    pipeline = SimpleNamespace(hallucination_detector=detector, llm=SimpleNamespace(generate=lambda prompt: 'answer'))
    predict = detector.nli_model.predict
    for enabled in (False, True):
        monkeypatch.setenv('CLOUDRAG_NLI_EXPERIMENT', '1' if enabled else '0')
        with PipelineTrace(pipeline) as trace:
            detector.nli_model.predict([('text', 'claim')], batch_size=32, apply_softmax=True)
        assert ('raw_scores' in trace.export()['nli'][0]) == enabled
        assert detector.nli_model.predict == predict


def test_worker_propagates_experiment_only_for_registered_protocol():
    from scripts.run_managed_gate import observer_environment
    original = dict(CLOUDRAG_NLI_EXPERIMENT='1')
    assert 'CLOUDRAG_NLI_EXPERIMENT' not in observer_environment(original, {})
    assert observer_environment({}, dict(nli_experiment=experiment.EXPERIMENT))['CLOUDRAG_NLI_EXPERIMENT'] == '1'
    assert original == dict(CLOUDRAG_NLI_EXPERIMENT='1')


def test_first_real_window_requires_explicit_authorization_before_any_work(tmp_path):
    with pytest.raises(PermissionError, match='including first'):
        unattended.prepare(tmp_path / 'absent', nli_experiment=True)
    assert not (tmp_path / 'absent').exists()


def test_bootstrap_fixed_seed_paired_delta():
    data = rows()
    for row in data:
        row['elapsed_s'] = row['index'] + (20 if row['arm'] == 'control' else 10)
    result = experiment.bootstrap(data)
    assert result == experiment.bootstrap(data)
    assert all(r['fields']['total_s']['delta_p95'] == pytest.approx(-10) for r in result)
    assert all(r['fields']['total_s']['ci95'] == pytest.approx([-10, -10]) for r in result)
    assert all(r['resamples'] == 10000 and r['seed'] == 42 for r in result)


def test_synthetic_end_to_end_120_and_hashes(tmp_path, monkeypatch):
    # pytest temporary roots may be under the worktree; production still rejects those.
    monkeypatch.setattr(unattended, 'external_root', lambda p: p)
    root = tmp_path / 'dry'
    experiment.dry_run(root)
    summary = gate.read_json(root / 'summary.json')
    assert summary['complete'] and summary['confirmation_ready'] and summary['quality_pass']
    assert summary['verdict'].startswith('NO-GO')
    assert len(gate.local_records(root / 'cohort')) == 120
    assert all(g['valid'] == 20 for g in summary['groups'])
    unattended.verify_package(root)
    target = next((root / 'cohort/attempts').glob('*/result.json'))
    target.write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        unattended.verify_package(root)


def test_replay_uses_original_llm_text_not_formatted_answer():
    detector, response = experiment.synthetic_response()
    original = detector.check
    seen = []
    def checked(text, chunks):
        seen.append(text)
        return original(text, chunks)
    detector.check = checked
    response['answer'] = 'Truncated or reformatted presentation must not enter the verifier.'
    assert experiment.equivalence(detector, response)['passed']
    assert seen == [response['llm_response']['text']] * 2


@pytest.mark.parametrize('arm', ['control', 'candidate'])
def test_arm_selector_restores_after_exception_and_does_not_change_config(arm):
    detector, _ = experiment.synthetic_response()
    config = dict(model='fixed', temperature=0, chunks=5)
    pipeline = SimpleNamespace(hallucination_detector=detector, config=config.copy())
    preparation = SimpleNamespace(pipelines={'lexical': pipeline})
    def fail():
        assert detector.nli_pair_schedule == experiment.STRATEGIES[arm]
        raise TimeoutError('controlled')
    with pytest.raises(TimeoutError):
        experiment.measure_arm(preparation, 'lexical', arm, fail)
    assert detector.nli_pair_schedule == 'per_claim' and pipeline.config == config


@pytest.mark.parametrize('method', ['bm25', 'dense', 'hybrid'])
@pytest.mark.parametrize('answer', ['The service supports private networks. The service encrypts data.',
                                  'I cannot find sufficient information in the provided context.'])
def test_pipeline_same_retrieval_prompt_response_sources_and_decline(method, answer):
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.pipeline.pipeline_config import PipelineConfig
    from src.generation.llm_manager import LLMResponse
    from src.generation.response_formatter import ResponseFormatter
    subject = object.__new__(RAGPipeline)
    subject.config = PipelineConfig(name='test', retrieval_method=method, final_top_k=2)
    subject.query_processor = subject._routing_qp = subject.hybrid_index = None
    candidates = [SimpleNamespace(chunk_id=str(i), chunk_text='The service supports private networks.',
        cloud_provider='aws', service_name='service', heading_path='topic') for i in range(3)]
    retrieval_calls, prompts = [], []
    def search(question, **kwargs):
        retrieval_calls.append((question, kwargs))
        return candidates
    def generate(**kwargs):
        prompts.append(kwargs)
        return LLMResponse(text=answer, model='mock', provider='mock', tokens_input=10,
                           tokens_output=5, latency_ms=0, from_cache=False)
    subject.retriever = SimpleNamespace(search=search)
    subject.reranker = SimpleNamespace(rerank=lambda q, c, top_k: list(reversed(c))[:top_k]) if method == 'hybrid' else None
    subject.llm = SimpleNamespace(generate=generate)
    subject.response_formatter = ResponseFormatter()
    subject.hallucination_detector, _ = experiment.synthetic_response()
    # Exercise actual extraction, including the decline path.
    from src.generation.hallucination_detector import HallucinationDetector
    subject.hallucination_detector._extract_claims = HallucinationDetector._extract_claims.__get__(subject.hallucination_detector)
    preparation = SimpleNamespace(pipelines={method: subject})
    outputs = []
    for arm in experiment.STRATEGIES:
        response = experiment.measure_arm(preparation, method, arm, lambda: subject.query('question'))
        assert response.error is None
        payload = response.model_dump(mode='json')
        payload.pop('latency')
        payload['hallucination_report'].pop('processing_time_ms')
        outputs.append(payload)
    assert outputs[0] == outputs[1]
    assert prompts[0] == prompts[1] and retrieval_calls[0] == retrieval_calls[1]
    assert outputs[0]['confidence'] == ('HONEST_DECLINE' if answer.startswith('I cannot') else 'LOW')


def test_package_quality_missing_cannot_confirm_despite_fast_complete_cohort(tmp_path):
    gate.write_new(tmp_path / 'cohort/source-manifest.json',
                   dict(protocol=dict(nli_experiment=experiment.EXPERIMENT, quality_source_ids=[])))
    for row in rows():
        publish(tmp_path / 'cohort', row)
    report = unattended.package(tmp_path)
    assert report['complete'] and report['latency_pass_candidate']
    assert not report['quality_pass'] and not report['confirmation_ready']


@pytest.mark.parametrize('field', ['gpu', 'hardware', 'server', 'model', 'model_digest', 'python', 'packages', 'lock_sha256'])
def test_registration_refuses_changed_reference_environment(tmp_path, monkeypatch, field):
    environment = {k: 'fixed' for k in ('gpu', 'hardware', 'server', 'model', 'model_digest', 'python', 'packages', 'lock_sha256')}
    original = dict(environment=environment, queries=['fixed'])
    gate.write_new(tmp_path / 'cohort/source-manifest.json', dict(protocol=original))
    monkeypatch.setattr(experiment, 'BASE_ROOT', tmp_path)
    monkeypatch.setattr(unattended, 'verify_package', lambda p: True)
    monkeypatch.setattr(gate, 'digest', lambda p: experiment.BASE_MANIFEST)
    changed = copy.deepcopy(original)
    changed['environment'][field] = 'changed'
    with pytest.raises(ValueError, match='identity changed'):
        experiment.register(changed)


@pytest.mark.parametrize('fault', ['none', 'missing', 'expired', 'restored', 'proof', 'overlong', 'wrong_cohort'])
def test_real_inference_requires_live_bounded_proven_window(tmp_path, monkeypatch, fault):
    cohort = tmp_path / 'cohort'
    window = tmp_path / 'windows/one'
    now = datetime.now(timezone.utc)
    deadline = now + timedelta(minutes=-1 if fault == 'expired' else 118)
    hard = now + timedelta(minutes=121 if fault == 'overlong' else 120)
    monkeypatch.setenv('CLOUDRAG_GATE_WINDOW_ID', 'id')
    monkeypatch.setenv('CLOUDRAG_GATE_DEADLINE', deadline.isoformat())
    gate.write_new(window / 'window.json', dict(id='id', unattended=True, lexical_diagnostic=True,
        manage_anydesk=False, cohort=str(cohort if fault != 'wrong_cohort' else tmp_path / 'other'),
        at=now.isoformat(), hard_deadline_utc=hard.isoformat()))
    gate.write_new(window / 'armed.json', dict(deadline_utc=deadline.isoformat()))
    if fault == 'missing':
        monkeypatch.delenv('CLOUDRAG_GATE_DEADLINE')
    if fault == 'restored':
        gate.write_new(window / 'restored.json', {})
    if fault != 'proof':
        for kind in ('deadline', 'controller'):
            gate.write_new(window / 'proof' / kind / 'selftest-passed.json', dict(simulated=True))
    if fault == 'none':
        experiment.require_window(cohort)
    else:
        with pytest.raises(RuntimeError):
            experiment.require_window(cohort)
