"""Memory diagnosis must preserve interval boundaries, missingness and evidence integrity."""
import pytest

from scripts import analyze_gate_memory as analysis
from scripts import measure_interview_gate as gate


def sample(second):
    return dict(at=f'2026-09-09T00:00:{second:02d}+00:00', ram_available_bytes=2 * analysis.GIB,
        committed_bytes=20 * analysis.GIB, commit_limit_bytes=40 * analysis.GIB, ac=True,
        gpu={'utilization.gpu': '50', 'temperature.gpu': '60', 'memory.free': '1024'},
        ollama_ps_cli='43%/57% CPU/GPU', paging={'pages_input_per_s': 9999})


def test_chat_window_excludes_model_loading_faults_and_uses_its_own_duration():
    start = '2026-09-09T00:00:10+00:00'
    origin = analysis.stamp(start)
    faults = [dict(timestamp_s=origin + offset, bytes_read=4096, process='ollama', file='weights')
              for offset in (-5, 0, 5, 6, 20)]
    result = analysis.window_metrics([sample(s) for s in (0, 10, 15, 16, 30)], faults, start, 6)
    assert result['samples'] == 2
    assert result['available_gib']['median'] == 2
    assert result['commit_fraction']['median'] == .5
    assert result['hard_fault_count'] == 2
    assert result['hard_faults_per_s'] == pytest.approx(2 / 6)
    assert result['hard_faults_per_s'] != result['page_input_per_s']['median']


def test_missing_samples_do_not_prove_zero_memory_or_ac():
    result = analysis.window_metrics([], [], '2026-09-09T00:00:00+00:00', 1)
    assert result['available_gib']['median'] is None
    assert result['all_sampled_ac'] is None


def test_missing_memory_value_cannot_become_zero_available():
    row = sample(0)
    del row['ram_available_bytes']
    with pytest.raises(KeyError):
        analysis.window_metrics([row], [], row['at'], 1)


def test_timestamps_require_timezone_and_nonfinite_stats_are_rejected():
    with pytest.raises(ValueError, match='timezone'):
        analysis.stamp('2026-09-09T00:00:00')
    with pytest.raises(ValueError, match='Nonfinite'):
        analysis.stats([float('nan')])


def test_changed_journal_is_rejected_before_analysis(tmp_path):
    gate.write_new(tmp_path / 'source-manifest.json', {'mode': 'fresh',
        'protocol': {'systems': ['hybrid'], 'build_id': 'measured'}})
    row = gate.measure_attempt(tmp_path, dict(system='hybrid', phase='cold', index=0, warmup=False,
                                            consumes_slot=True),
                               lambda: dict(status='error', error='synthetic'))
    journal = tmp_path / 'attempts' / row['attempt_id'] / 'events.jsonl'
    with journal.open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        analysis.analyze(tmp_path)
