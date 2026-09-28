import json

import pytest

from scripts.study_gate_draft import main, run, schedule, verify


def test_synthetic_complete_and_integrity(tmp_path):
    root = tmp_path / 'package'
    result = run(root)
    assert result['completed'] == result['planned'] == 120
    assert all(x['n'] == 60 for x in result['systems'].values())
    assert result['verdict'] == 'NOT_A_GO_DECISION'
    verify(root)
    (root / 'attempt-000.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        verify(root)


def test_preflight_rejects_before_work(tmp_path):
    root = tmp_path / 'dirty'
    with pytest.raises(ValueError, match='preflight'):
        run(root, contaminated=True, callback=lambda _: pytest.fail('Executed despite contamination'))
    assert not list(root.glob('attempt*'))


def test_resume_authorization_and_no_duplicates(tmp_path):
    root = tmp_path / 'resume'
    run(root, stop_after=4)
    previous = (root / 'attempt-000.json').read_bytes()
    with pytest.raises(ValueError, match='authorization'):
        run(root, resume=True)
    result = run(root, resume=True, authorize_new_window=True)
    assert result['completed'] == 120 and (root / 'attempt-000.json').read_bytes() == previous
    assert len(list(root.glob('attempt*'))) == 120
    verify(root)


def test_interrupted_pair_terminal_no_imputation_or_next_window_work(tmp_path):
    root = tmp_path / 'interrupted'
    n = [0]
    def interrupt(_):
        n[0] += 1
        if n[0] == 2:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(root, callback=interrupt)
    result = run(root, resume=True, authorize_new_window=True, callback=lambda _: pytest.fail('No work'))
    assert result['status'] == 'interrupted'
    assert result['systems']['no_rag']['failures'] == 1
    assert result['systems']['no_rag']['n'] == 0
    aborted = json.loads((root / 'aborted-001.json').read_text())
    assert aborted['elapsed_s'] is None
    verify(root)


def test_expired_never_calls_work_and_invalid_keeps_response(tmp_path):
    root = tmp_path / 'expired'
    run(root, stop_after=2, wall=lambda: 0, invalid_at=0)
    result = run(root, resume=True, authorize_new_window=True, wall=lambda: 7201,
                 callback=lambda _: pytest.fail('Expired inference'))
    assert result['status'] == 'expired' and result['completed'] == 2
    row = json.loads((root / 'attempt-000.json').read_text())
    assert row['status'] == 'success' and row['elapsed_s'] is not None and not row['valid']
    assert result['systems']['hybrid']['invalid'] == 1


def test_clock_boundary_and_balanced_schedule(tmp_path):
    clock = [0.0]
    def work(_):
        clock[0] += 2  # readiness
        clock[0] += 3  # query
    result = run(tmp_path / 'boundary', clock=lambda: clock[0], callback=work)
    assert result['systems']['hybrid']['p95'] == 5
    for q in {row['query_id'] for row in schedule()}:
        for condition in ('hybrid', 'no_rag'):
            assert sum(row['query_id'] == q and row['condition'] == condition for row in schedule()) == 10


def test_real_launch_not_available(tmp_path):
    with pytest.raises(SystemExit):
        main(['--output', str(tmp_path / 'real')])
