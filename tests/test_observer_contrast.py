"""Observer overhead decisions use paired raw times, never adjusted RAG latencies."""
import pytest

from scripts import contrast_interview_observer as contrast


def pairs(ratio=1.02):
    return [dict(index=i, bare_s=10 + i, observed_s=(10 + i) * ratio,
                 valid=True, statuses=['success', 'success']) for i in range(10)]


def test_paired_bootstrap_accepts_small_cost_and_rejects_material_cost():
    result = contrast.summarize(pairs())
    assert result['passed'] is True
    assert result['upper_95_percent'] == pytest.approx(2)
    assert contrast.summarize(pairs(1.06))['passed'] is False
    assert contrast.summarize(pairs()) == result


def test_variable_cost_uses_upper_bound_not_only_median():
    rows = pairs(1.01)
    for row in rows[-4:]:
        row['observed_s'] = row['bare_s'] * 1.3
    result = contrast.summarize(rows)
    assert result['median_percent'] == pytest.approx(1)
    assert result['upper_95_percent'] > 5
    assert not result['passed']


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'failed', 'invalid', 'nan', 'zero'])
def test_incomplete_or_invalid_contrasts_cannot_pass(kind):
    rows = pairs()
    if kind == 'missing':
        rows.pop()
    elif kind == 'duplicate':
        rows[-1]['index'] = 0
    elif kind == 'failed':
        rows[-1]['statuses'][1] = 'error'
    elif kind == 'invalid':
        rows[-1]['valid'] = False
    elif kind == 'nan':
        rows[-1]['observed_s'] = float('nan')
    else:
        rows[-1]['bare_s'] = 0
    with pytest.raises(ValueError):
        contrast.summarize(rows)


def test_alternation_and_fixed_work_are_identical_between_arms():
    assert contrast.order(0) == ('bare', 'observed')
    assert contrast.order(1) == ('observed', 'bare')
    result = contrast.hash_work(b'constant buffer', 3)
    assert result == contrast.hash_work(b'constant buffer', 3)
    assert result['iterations'] == 3 and result['status'] == 'success'


def sensor():
    import time
    return dict(monotonic_s=time.monotonic(), errors=[], ac=True,
                scheme=contrast.observe.BALANCED, overlay=contrast.observe.BEST_PERFORMANCE,
                cpu_percent=0, gpu={'utilization.gpu': '0'}, ram_available_bytes=100, processes=[])


def test_complete_real_recorder_executes_exactly_twenty_arms(tmp_path):
    calls = []

    def work():
        calls.append(True)
        return contrast.hash_work(b'small fixed test buffer', 3)

    def observer(path, **kwargs):
        return contrast.observe.Observer(path, sampler=sensor, **kwargs)

    rows = contrast.execute_pairs(tmp_path, work, sensor, observer)
    assert len(calls) == 20
    assert len(rows) == len(list((tmp_path / 'pairs').glob('*.json'))) == 10
    assert len(contrast.gate.local_records(tmp_path / 'bare')) == 10
    assert len(contrast.gate.local_records(tmp_path / 'observed')) == 10
    assert rows[0]['order'] == ('bare', 'observed')
    assert rows[1]['order'] == ('observed', 'bare')
    with pytest.raises(FileExistsError):
        contrast.execute_pairs(tmp_path, work, sensor, observer)
    assert len(calls) == 20


def test_failed_arm_is_durable_and_prevents_second_arm(tmp_path):
    def failed():
        raise OSError('synthetic failure')

    with pytest.raises(RuntimeError, match='partial evidence retained'):
        contrast.execute_pairs(tmp_path, failed, sensor)
    rows = contrast.gate.local_records(tmp_path / 'bare')
    assert len(rows) == 1 and rows[0]['status'] == 'error'
    assert not (tmp_path / 'observed').exists()
    assert not (tmp_path / 'pairs').exists()
