"""Diagnostic probes must observe the real verifier without changing its decisions."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.lexical_diagnostic import PipelineTrace, paired_schedule
from src.generation.hallucination_detector import HallucinationDetector


class Model:
    def predict(self, sentences, **kwargs):
        return np.tile([0.05, 0.9, 0.05], (len(sentences), 1))


def pipeline():
    detector = HallucinationDetector()
    detector._nli_model = Model()
    detector._nli_available = True
    response = SimpleNamespace(text='The service supports private networks.', error=None)
    llm = SimpleNamespace(generate=lambda prompt, **kwargs: response)
    return SimpleNamespace(llm=llm, hallucination_detector=detector)


def test_schedule_is_paired_counterbalanced_and_fixed():
    schedule = paired_schedule()
    assert len(schedule) == 40
    assert schedule[:4] == [('hybrid', 0), ('lexical', 0), ('lexical', 1), ('hybrid', 1)]
    assert len(set(schedule)) == 40
    for key in ('hybrid', 'lexical'):
        assert sorted(i for s, i in schedule if s == key) == list(range(20))


def test_trace_preserves_generate_identity_arguments_and_restores():
    subject = pipeline()
    original = subject.llm.generate
    expected = original('prompt', temperature=0)
    with PipelineTrace(subject) as trace:
        assert subject.llm.generate('prompt', system_prompt='sys', temperature=0) is expected
    assert subject.llm.generate is original
    record = trace.export()
    assert record['generation'][0]['prompt'] == 'prompt'
    assert record['generation'][0]['system_prompt'] == 'sys'
    assert record['generation'][0]['prompt_chars'] == 6
    assert record['generation'][0]['status'] == 'success'


def test_real_verifier_predictions_identical_and_actual_pairs_recorded():
    subject = pipeline()
    detector = subject.hallucination_detector
    claims = ['The service supports private networks.', 'The service supports private endpoints.']
    texts, ids = ['private networks', 'private endpoints'], ['a', 'b']
    expected = detector._nli_matching(claims, texts, ids)
    with PipelineTrace(subject) as trace:
        actual = detector._nli_matching(claims, texts, ids)
    assert actual == expected
    data = trace.export()
    assert data['nli_predict_calls'] == 2
    assert data['nli_pairs_attempted'] == 4
    assert data['nli_pairs_successful'] == 4
    assert all(r['batch_size'] == 32 and r['apply_softmax'] for r in data['nli'])


def test_artifact_skip_does_not_count_as_nli(monkeypatch):
    subject = pipeline()
    monkeypatch.setattr('src.generation.hallucination_detector.classify_artifact', lambda claim: 'H1')
    with PipelineTrace(subject) as trace:
        result = subject.hallucination_detector._nli_matching(['artifact'], ['context'], ['a'])
    assert result[0].status == 'not_a_claim'
    assert trace.export()['nli_predict_calls'] == 0


def test_failed_prediction_still_counted_and_real_fallback_unchanged():
    subject = pipeline()
    def failure(*args, **kwargs):
        raise RuntimeError('synthetic verifier error')
    subject.hallucination_detector._nli_model.predict = failure
    with PipelineTrace(subject) as trace:
        result = subject.hallucination_detector._nli_matching(['A private network is supported.'], ['private network'], ['a'])
    assert result[0].verification_method == 'keyword_fallback'
    assert trace.export()['nli_pairs_attempted'] == 1
    assert trace.export()['nli_pairs_successful'] == 0
    assert trace.export()['nli'][0]['error_type'] == 'RuntimeError'


def test_claim_extraction_observed_without_a_second_extraction():
    subject = pipeline()
    text = 'This service provides private networks. This service supports public endpoints.'
    expected = subject.hallucination_detector._extract_claims(text)
    with PipelineTrace(subject) as trace:
        assert subject.hallucination_detector._extract_claims(text) == expected
    assert trace.export()['extractions'][0]['claims'] == expected
    assert len(trace.export()['extractions']) == 1


def test_cold_model_rejected_without_loading():
    subject = pipeline()
    subject.hallucination_detector._nli_model = None
    with pytest.raises(ValueError, match='prepared'):
        with PipelineTrace(subject):
            pytest.fail('cold trace admitted')
    assert subject.hallucination_detector._nli_model is None


def test_exception_restores_all_methods_and_is_not_swallowed():
    subject = pipeline()
    detector = subject.hallucination_detector
    original = (subject.llm.generate, detector._extract_claims, detector._nli_model.predict)
    with pytest.raises(RuntimeError, match='interrupted'):
        with PipelineTrace(subject):
            raise RuntimeError('interrupted')
    assert (subject.llm.generate, detector._extract_claims, detector._nli_model.predict) == original


def test_generation_error_propagates_and_is_recorded():
    subject = pipeline()
    def fail(prompt, **kwargs):
        raise TimeoutError('synthetic timeout')
    subject.llm.generate = fail
    with PipelineTrace(subject) as trace:
        with pytest.raises(TimeoutError):
            subject.llm.generate('question')
    assert trace.export()['generation'][0]['status'] == 'error'
    assert subject.llm.generate is fail


def test_historical_verification_rejects_changed_or_escaping_files(tmp_path):
    from scripts.analyze_lexical_diagnostic import verified_path
    from scripts.measure_interview_gate import digest
    source = tmp_path / 'cohort'
    source.mkdir()
    record = source / 'result.json'
    record.write_text('{}')
    inventory = {'result.json': digest(record)}
    assert verified_path(source, record, inventory) == record
    record.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='changed'):
        verified_path(source, record, inventory)
    with pytest.raises(ValueError, match='escapes'):
        verified_path(source, tmp_path / 'external.json', inventory)


def test_historical_workload_never_invents_nli_call_counts():
    from scripts.analyze_lexical_diagnostic import workload
    row = dict(attempt_id='a', query={'query_id': 'q'}, index=0, system='lexical',
        started_at='2026-09-11T00:00:00Z', elapsed_s=5,
        response=dict(llm_response={'text': 'answer', 'tokens_input': 10, 'tokens_output': 2},
            hallucination_report={'total_claims': 1, 'not_a_claim_claims': 0},
            latency={'retrieval_ms': 1000, 'generation_ms': 2000, 'total_ms': 3500},
            retrieved_chunks=[{'chunk_id': 'chunk'}]))
    calls = [dict(method='chat', status='success', request={'messages': [
        {'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'question'}]})]
    result = workload(row, calls)
    assert result['other_s'] == 2
    assert result['prompt_chars'] == 8
    assert result['nli_predict_calls'] is None and result['nli_pairs_attempted'] is None
    assert result['chunks'] == 1 and result['claims'] == 1


def test_trace_nested_same_instance_rejected_and_original_restored():
    subject = pipeline()
    original = subject.llm.generate
    trace = PipelineTrace(subject)
    with trace:
        with pytest.raises(ValueError, match='reused'):
            with trace:
                pass
    assert subject.llm.generate is original


@pytest.mark.parametrize('failed', [False, True])
def test_trace_published_with_durable_terminal_record_even_on_query_error(tmp_path, failed):
    from scripts.lexical_diagnostic import measure_traced_attempt
    from scripts import measure_interview_gate as gate
    subject = pipeline()
    def query(question):
        subject.llm.generate(question)
        if failed:
            raise RuntimeError('synthetic query failure')
        return SimpleNamespace(model_dump=lambda **kwargs: dict(answer='response', confidence='HIGH'))
    subject.query = query
    row = measure_traced_attempt(tmp_path,
        dict(system='lexical', phase='warm', index=0, query={'question': 'question'}), subject)
    stored = gate.read_json(tmp_path / 'attempts' / row['attempt_id'] / 'result.json')
    assert stored['status'] == ('error' if failed else 'success')
    assert stored['diagnostic_trace']['generation'][0]['prompt'] == 'question'
    assert stored['elapsed_s'] > 0
    journal = tmp_path / 'attempts' / row['attempt_id'] / 'events.jsonl'
    assert stored['journal_sha256'] == gate.digest(journal)


def test_real_stage_tracker_still_times_and_records_failure_boundaries():
    from src.pipeline.rag_pipeline import LatencyTracker
    original = LatencyTracker.measure
    tracker = LatencyTracker()
    with PipelineTrace(pipeline()) as trace:
        with tracker.measure('retrieval'):
            pass
        with pytest.raises(RuntimeError):
            with tracker.measure('generation'):
                raise RuntimeError('synthetic failure')
    assert LatencyTracker.measure is original
    assert tracker.get_breakdown().retrieval_ms >= 0
    stages = trace.export()['stages']
    assert [r['name'] for r in stages] == ['retrieval', 'generation']
    assert stages[0]['status'] == 'success' and stages[1]['status'] == 'error'
    assert all(r['started_at'] <= r['finished_at'] and r['elapsed_s'] >= 0 for r in stages)
