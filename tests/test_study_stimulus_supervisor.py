import time

import pytest

from scripts.study_operator.stimulus_supervisor import ColdSupervisor


def fixture_worker(connection, config, parent_pid):
    try:
        if config.get('preparation_hang'):
            time.sleep(30)
        connection.send(dict(ready=True))
        while True:
            index = connection.recv()
            if config.get('hang'):
                time.sleep(30)
            connection.send(dict(row=dict(index=index, synthetic=True), proof=dict(mode='SYNTHETIC')))
    finally:
        connection.close()


def supervisor(tmp_path, **settings):
    config = dict(root=str(tmp_path), **settings)
    return ColdSupervisor(config, admission=lambda receipt: None, poll=lambda: None,
                          worker_target=fixture_worker, preparation_seconds=3, call_seconds=.1)


def test_short_calls_always_check_host_telemetry(tmp_path):
    observations = []
    with supervisor(tmp_path) as runner:
        runner.poll = lambda: observations.append('sample')
        assert runner(1) == dict(index=1, synthetic=True)
        assert observations == ['sample', 'sample']
        with pytest.raises(ValueError):
            runner.save_complete(tmp_path/'boot.json', {})
    assert not runner.process.is_alive()


def test_blocked_native_call_is_killed_and_cannot_resume(tmp_path):
    with supervisor(tmp_path, hang=True) as runner:
        with pytest.raises(TimeoutError):
            runner(1)
        assert runner.terminal and not runner.process.is_alive()
        with pytest.raises(ValueError):
            runner(1)


def test_failed_admission_closes_worker_before_any_query(tmp_path):
    runner = supervisor(tmp_path)
    def reject(receipt):
        raise ValueError('Foreign GPU process')
    runner.admission = reject
    with pytest.raises(ValueError, match='Foreign'):
        with runner:
            pytest.fail('Rejected admission must not query')
    assert runner.terminal and not runner.process.is_alive()


def test_contamination_is_terminal_and_never_publishes_completion(tmp_path):
    with supervisor(tmp_path) as runner:
        def reject():
            raise ValueError('Contamination')
        runner.poll = reject
        with pytest.raises(ValueError, match='Contamination'):
            runner(1)
        with pytest.raises(ValueError):
            runner.save_complete(tmp_path/'boot.json', {})
        assert not (tmp_path/'boot.json').exists()


def test_admission_and_telemetry_cannot_be_omitted(tmp_path):
    with pytest.raises(ValueError):
        ColdSupervisor(dict(root=str(tmp_path)), admission=None, poll=None)
