"""Memory units, native counters and isolated WPR profile validation."""
import subprocess
import sys
import time

import pytest

from scripts import gate_memory


def test_commit_units_use_system_page_size_and_do_not_call_it_physical_usage():
    result = gate_memory.commit_metrics(total=100, limit=400, physical=200, available=50, page_size=4096)
    assert result['committed_bytes'] == 409600
    assert result['commit_limit_bytes'] == 1638400
    assert result['commit_fraction'] == .25
    assert result['ram_available_bytes'] == 204800


@pytest.mark.parametrize('total,limit,page_size', [(1, 0, 4096), (-1, 4, 4096), (1, 4, 0)])
def test_invalid_commit_counters_do_not_become_zero_pressure(total, limit, page_size):
    with pytest.raises(ValueError):
        gate_memory.commit_metrics(total, limit, 10, 2, page_size)


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows reference memory counters')
def test_native_system_commit_snapshot():
    result = gate_memory.system_memory()
    assert result['commit_limit_bytes'] >= result['committed_bytes'] > 0
    assert result['ram_total_bytes'] > result['ram_available_bytes'] > 0


@pytest.mark.skipif(sys.platform != 'win32', reason='WPR profile on reference Windows')
def test_wpr_accepts_profile_without_starting_a_recording():
    result = subprocess.run(['wpr', '-profiles', str(gate_memory.PROFILE)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'GateMemory' in result.stdout


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows PDH counters')
def test_paging_primes_before_reporting_rates_and_does_not_mislabel_hard_faults():
    counters = gate_memory.PagingCounters()
    try:
        assert all(v is None for v in counters.sample().values())
        time.sleep(.2)
        result = counters.sample()
        assert set(result) == {'page_faults_total_per_s', 'page_read_operations_per_s', 'pages_input_per_s'}
        assert all(v >= 0 for v in result.values())
    finally:
        counters.finalizer()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows uncached file IO')
def test_synthetic_fault_workload_is_bounded_and_refuses_overwrite(tmp_path):
    path = tmp_path / 'synthetic.bin'
    result = gate_memory.synthetic_hard_faults(path)
    assert result['bytes'] == path.stat().st_size == 16 * 1024 * 1024
    assert result['checksum'] == 0
    with pytest.raises(OSError):
        gate_memory.synthetic_hard_faults(path)
