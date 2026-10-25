from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from scripts.study_gate_environment import assess
from scripts.study_operator.service_transition import transition


def marker(phase='RESETTING'):
    return dict(schema_version=1, mode='fresh_runner', phase=phase, request_id='a' * 32,
                boot_id='boot-fixture', written_monotonic_s=49., deadline_monotonic_s=59.)


def row(models=None):
    return dict(at=datetime.now(timezone.utc).isoformat(), monotonic_s=50.,
                ollama_ps_api={'models': models or []}, service_state=marker(), processes=[], gpu_pids=[], errors=[])


CONFIG = dict(model_digest='d' * 64, service_mode='fresh_runner', service_boot_id='boot-fixture')


def test_only_bounded_owned_transition_allows_observed_empty_ps():
    assert transition(marker(), 50., 'boot-fixture') == (True, True)
    assert not assess([row()], CONFIG)
    assert 'model_residency' in assess([row()], CONFIG, admission=True)


@pytest.mark.parametrize('changed', [dict(boot_id='other'), dict(deadline_monotonic_s=50.),
    dict(deadline_monotonic_s=61.), dict(written_monotonic_s=51.), dict(phase='FAILED'),
    dict(request_id='invalid'), dict(deadline_monotonic_s=float('nan'))])
def test_missing_expired_foreign_or_unbounded_marker_is_invalid(changed):
    sample = row()
    sample['service_state'].update(changed)
    reasons = assess([sample], CONFIG)
    assert 'service_transition' in reasons and 'model_residency' in reasons


def test_transition_cannot_excuse_another_digest_or_missing_marker():
    models = [dict(digest='e' * 64, context_length=4096,
                   expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())]
    sample = row(models)
    assert 'model_residency' in assess([sample], CONFIG)
    sample.pop('service_state')
    assert 'service_transition' in assess([sample], CONFIG)


def test_generating_requires_model_and_lease_and_never_hides_raw_sample():
    sample = row()
    sample['service_state'] = marker('GENERATING')
    before = deepcopy(sample)
    assert 'model_residency' in assess([sample], CONFIG)
    assert sample == before
