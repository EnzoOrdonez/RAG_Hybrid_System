"""Independent vectors for the SDK boundary used by the simulated provider."""
import pytest

from tests.test_study_i7_ledger import human_operator
from tests.test_study_i7_reservation_time import MillisecondCloud


@pytest.mark.parametrize('fraction,transmitted', [
    ('968532', '969'), ('968499', '968'), ('999700', '999')])
@pytest.mark.parametrize('offset', ['Z', '-05:00'])
def test_fake_provider_matches_sdk_format_datetime(tmp_path, fraction, transmitted, offset):
    # SDK 587.0.0: command_lib/compute/reservations/util.py:325 calls
    # core/util/times.py:320 (%3f); :350-365 rounds half-up, keeping 999 on overflow.
    # These vectors were also checked with the installed FormatDateTime, not gcloud.
    op, _ = human_operator(tmp_path)
    cloud = MillisecondCloud(op, offset=0 if offset == 'Z' else -5)
    cloud.command(['beta', 'compute', 'reservations', 'create', 'cloudrag-i7-sdk-fixture',
        f'--delete-at-time=2026-11-01T04:00:00.{fraction}{offset}'])
    assert cloud.resource['deleteAtTime'] == f'2026-11-01T04:00:00.{transmitted}{offset}'
