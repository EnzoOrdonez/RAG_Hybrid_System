"""ADR-0026: the single gate attempt is recorded only after reservation 2 is held."""
import json
from datetime import datetime, timezone

import pytest

from scripts.study_operator.i7 import PHASE_B_GATE_RESERVATION_H
from scripts.study_operator.policy import CapacityRejected


class Measured(Exception):
    """Raised where the first measurement step would begin."""


@pytest.fixture
def gate_after(tmp_path, monkeypatch):
    from scripts.study_operator import i7_admission, i7_phase_b, stimulus_evidence
    from src.ui.components import study_protocol
    (tmp_path/'stimulus-acceptance.json').write_text(json.dumps({'status':'ACCEPTED'}))
    reference = tmp_path/'i3.json'
    reference.write_text(json.dumps({'rag':{'contexts':{}}}))
    monkeypatch.setattr(study_protocol, 'verify_draw', lambda path: {'config':{}})
    monkeypatch.setattr(stimulus_evidence, 'accept', lambda proofs, config: {'status':'ACCEPTED','software_sha256':'s'})
    monkeypatch.setattr(i7_admission, 'compare_contexts', lambda live, reference: None)

    def first_measurement(operator, profile):
        operator.measured_with_intent = (tmp_path/'gate-cohort-intent.json').exists()
        raise Measured(profile)
    monkeypatch.setattr(i7_phase_b, 'switch_profile', first_measurement)

    class Operator:
        root = tmp_path
        config = {'local_reviewed_config':'cfg','i3_context_reference_path':str(reference),'image_id':'i'}

        def __init__(self, outcome=None, state=None):
            self.outcome = outcome
            self.state = state if state is not None else {'reservation':{'deleted_verified':True}}
            self.reserved = []

        def now(self):
            return datetime(2026,11,1,tzinfo=timezone.utc)

        def reserve(self, hours):
            self.reserved.append(hours)
            if self.outcome:
                raise self.outcome
            self.state['reservation'] = {'name':'held'}

        def stop(self):
            self.closed = getattr(self, 'closed', []) + ['stop']

        def guard(self):
            self.closed = getattr(self, 'closed', []) + ['guard']

    def run(operator):
        return i7_phase_b._gate_after_prerequisites(operator, {'software_sha256':'s'})
    return Operator, run, tmp_path/'gate-cohort-intent.json'


def test_released_capacity_refusal_keeps_the_single_gate_attempt(gate_after):
    Operator, run, intent = gate_after
    refused = Operator(CapacityRejected('ZONE_RESOURCE_POOL_EXHAUSTED: fixture', code='ZONE_RESOURCE_POOL_EXHAUSTED'))
    with pytest.raises(CapacityRejected):
        run(refused)
    assert refused.reserved == [PHASE_B_GATE_RESERVATION_H]
    assert not intent.exists()
    retry = Operator()
    with pytest.raises(Measured):
        run(retry)
    assert retry.reserved == [PHASE_B_GATE_RESERVATION_H]
    assert retry.measured_with_intent and retry.closed == ['stop', 'guard']
    assert json.loads(intent.read_bytes())['replay_allowed'] is False


@pytest.mark.parametrize('outcome', [TimeoutError('create outcome unknown'),
                                     ValueError('UNKNOWN_RESERVATION_RECONCILE_FIRST')])
def test_unknown_reservation_outcome_records_no_gate_attempt(gate_after, outcome):
    Operator, run, intent = gate_after
    with pytest.raises(type(outcome)):
        run(Operator(outcome))
    assert not intent.exists()


def test_existing_attempt_is_never_replayed_or_paid_again(gate_after):
    Operator, run, intent = gate_after
    intent.write_text(json.dumps({'started_utc':'2026-11-01T00:00:00+00:00','replay_allowed':False}))
    operator = Operator()
    with pytest.raises(ValueError, match='GATE_ATTEMPT_EXISTS_NO_REPLAY'):
        run(operator)
    assert operator.reserved == []


def test_open_reservation_is_reused_and_the_attempt_recorded_before_measuring(gate_after):
    Operator, run, intent = gate_after
    operator = Operator(state={'reservation':{'name':'held','deleted_verified':False}})
    with pytest.raises(Measured):
        run(operator)
    assert operator.reserved == []
    assert operator.measured_with_intent
