"""Legacy precision compatibility, with exact matching for whole-second requests."""
from datetime import datetime, timedelta

import pytest

from scripts.study_operator.i7 import reservation_deadline_matches


@pytest.mark.parametrize('fraction,transmitted', [
    ('968532', '969'), ('968499', '968'), ('968500', '969'), ('999700', '999')])
@pytest.mark.parametrize('offset', ['Z', '-05:00'])
def test_legacy_accepts_sdk_half_up_with_saturated_overflow(fraction, transmitted, offset):
    assert reservation_deadline_matches(
        f'2026-11-01T04:00:00.{fraction}{offset}',
        f'2026-11-01T04:00:00.{transmitted}{offset}')


@pytest.mark.parametrize('fraction,truncated', [('968532', '968'), ('968499', '968'), ('999700', '999')])
def test_legacy_also_accepts_explicit_truncation_compatibility(fraction, truncated):
    assert reservation_deadline_matches(
        f'2026-11-01T04:00:00.{fraction}Z', f'2026-10-31T23:00:00.{truncated}-05:00')


@pytest.mark.parametrize('offset', ['Z', '-05:00'])
@pytest.mark.parametrize('delta_ms', [-1, 1])
def test_whole_second_requests_reject_any_millisecond_change(offset, delta_ms):
    expected = f'2026-11-01T04:00:00{offset}'
    observed = datetime.fromisoformat(expected) + timedelta(milliseconds=delta_ms)
    assert not reservation_deadline_matches(expected, observed.isoformat())


@pytest.mark.parametrize('delta_us', [-1, 1])
def test_whole_second_requests_reject_submillisecond_change(delta_us):
    expected = '2026-11-01T04:00:00Z'
    observed = datetime.fromisoformat(expected) + timedelta(microseconds=delta_us)
    assert not reservation_deadline_matches(expected, observed.isoformat())


@pytest.mark.parametrize('observed', ['2026-11-01T04:00:00.970Z',
    '2026-11-01T04:00:00.967Z', '2026-11-01T04:00:01.000Z',
    '2026-11-01T04:00:00.968532Z'])
def test_legacy_rejects_instants_outside_the_transmitted_candidates(observed):
    assert not reservation_deadline_matches('2026-11-01T04:00:00.968532Z', observed)
