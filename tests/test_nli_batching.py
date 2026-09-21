"""Scheduling equivalence with deterministic scores; never load a real model."""
import copy

import numpy as np
import pytest

from src.generation import hallucination_detector as module
from src.generation.hallucination_detector import HallucinationDetector


class Model:
    def __init__(self, scalar=False):
        self.calls = []
        self.scalar = scalar

    def predict(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        def score(text, claim):
            if self.scalar:
                return .91 if claim.startswith('supported') else .4
            if claim.startswith('supported'):
                return [.05, .90 if text == 'first' else .80, .05 if text == 'first' else .15]
            if claim.startswith('contradicted'):
                return [.9, .05, .05]
            return [.2, .3, .5]
        return np.array([score(*pair) for pair in sentences])


def detector(schedule='per_claim', model=None):
    result = HallucinationDetector(nli_pair_schedule=schedule)
    result._nli_model = model if model is not None else Model()
    return result


@pytest.mark.parametrize('claims_count', [0, 1, 6, 7, 13, 20])
@pytest.mark.parametrize('chunks_count', [0, 1, 5])
def test_batch_equivalence_and_internal_limit(claims_count, chunks_count):
    claims = [f'{["supported", "contradicted", "neutral"][i % 3]} claim {i}' for i in range(claims_count)]
    chunks = ['first'] + [f'chunk-{i}' for i in range(1, chunks_count)] if chunks_count else []
    ids = [f'id-{i}' for i in range(chunks_count)]
    control, candidate = detector(), detector('cross_claim')
    assert candidate._nli_matching(claims, chunks, ids) == control._nli_matching(claims, chunks, ids)
    calls = candidate._nli_model.calls
    assert len(calls) == (1 if claims_count and chunks_count else 0)
    if calls:
        assert calls[0][0] == [(text, claim) for claim in claims for text in chunks]
        assert calls[0][1] == dict(batch_size=32, show_progress_bar=False, apply_softmax=True)


def test_golden_status_evidence_and_artifact_positions(monkeypatch):
    monkeypatch.setattr(module, 'classify_artifact', lambda c: c == 'artifact')
    claims = ['artifact', 'supported claim', 'contradicted claim', 'artifact', 'neutral claim']
    d = detector('cross_claim')
    result = d._nli_matching(claims, ['first', 'second'], ['A', 'B'])
    assert [(x.status, x.evidence_chunk_id, x.nli_score) for x in result] == [
        ('not_a_claim', None, 0), ('supported', 'A', .9), ('contradicted', 'A', .9),
        ('not_a_claim', None, 0), ('unsupported', 'A', .3)]
    assert len(d._nli_model.calls[0][0]) == 6
    assert [x.claim_text for x in result] == claims


def test_scalar_scores_preserve_previous_supported_interface():
    args = (['supported claim', 'neutral claim'], ['first', 'second'], ['A', 'B'])
    assert detector('cross_claim', Model(True))._nli_matching(*args) == detector(model=Model(True))._nli_matching(*args)


@pytest.mark.parametrize('fault', [TimeoutError('timed out'), RuntimeError('inference failed'),
                                   [], [[.1, .9]], [[float('nan'), .2, .3]] * 4, None])
def test_batch_failure_recovers_but_stays_visible(fault):
    class FailingFirst(Model):
        def predict(self, sentences, **kwargs):
            if not self.calls:
                self.calls.append((list(sentences), kwargs))
                if isinstance(fault, Exception):
                    raise fault
                return fault
            return super().predict(sentences, **kwargs)
    d = detector('cross_claim', FailingFirst())
    d._extract_claims = lambda _: ['supported claim', 'contradicted claim']
    chunks = [dict(text='first', chunk_id='A'), dict(text='second', chunk_id='B')]
    report = d.check('answer', chunks)
    assert report.method == 'mixed'
    assert [c.status for c in report.claim_details] == ['supported', 'contradicted']
    assert all(c.verification_error.startswith('batch:') for c in report.claim_details)
    assert d.nli_batch_error is not None
    assert d.check('answer', chunks).method == 'nli'
    assert d.nli_batch_error is None


def test_failed_recovery_preserves_keyword_fallback():
    class Fails(Model):
        def predict(self, *args, **kwargs):
            raise TimeoutError('controlled')
    d = detector('cross_claim', Fails())
    d._extract_claims = lambda _: ['The service supports private network endpoints.']
    result = d.check('answer', [dict(text='The service supports private network endpoints.', chunk_id='A')])
    assert result.method == 'mixed'
    assert result.claim_details[0].verification_method == 'keyword_fallback'
    assert result.claim_details[0].verification_error == 'batch:TimeoutError;TimeoutError'


def test_whole_report_equivalence_without_changing_inputs():
    answer = 'Supported claim one. Contradicted claim two. Neutral claim three.'
    chunks = [dict(text='first', chunk_id='A'), dict(text='second', chunk_id='B')]
    original = copy.deepcopy(chunks)
    reports = []
    for schedule in ('per_claim', 'cross_claim'):
        d = detector(schedule)
        d._extract_claims = lambda _: ['supported claim', 'contradicted claim', 'neutral claim']
        reports.append(d.check(answer, chunks).model_dump(exclude={'processing_time_ms'}))
    assert reports[0] == reports[1]
    assert reports[0]['faithfulness_score'] == .3333
    assert chunks == original


def test_default_is_control_and_unknown_schedule_fails_closed():
    assert HallucinationDetector().nli_pair_schedule == 'per_claim'
    with pytest.raises(ValueError, match='schedule'):
        HallucinationDetector(nli_pair_schedule='unknown')
    d = detector()
    d.nli_pair_schedule = 'unknown'
    with pytest.raises(ValueError, match='schedule'):
        d._nli_matching(['supported claim'], ['first'], ['A'])
