"""I7 money boundaries, exercised without a provider or participant data."""
import pytest

from scripts.study_operator.lifecycle import Operator as LegacyOperator
from tests.test_study_i7_human import closed_phase_b
from tests.test_study_i7_operator import setup


@pytest.mark.parametrize('human', [False, True])
@pytest.mark.parametrize('legacy_instance', [False, True])
@pytest.mark.parametrize('operation', ['reserve_cost', 'ip_reserve'])
def test_i7_blocks_legacy_spending_before_cloud(tmp_path, human, legacy_instance, operation):
    if human:
        operator, cloud = closed_phase_b(tmp_path)
        operator.human_phase_open()
    else:
        operator, cloud, _ = setup(tmp_path)
    cloud.root = tmp_path
    if legacy_instance:
        operator = LegacyOperator(tmp_path, cloud)
    operator.config['ip_name'] = 'cloudrag-i7-fixture'
    with pytest.raises(ValueError, match='I7_LEGACY_PAID_OPERATION_FORBIDDEN'):
        if operation == 'reserve_cost':
            operator.reserve_cost('fixture', .72)
        else:
            operator.ip_reserve()
    assert cloud.calls == []
