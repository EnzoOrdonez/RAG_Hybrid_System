import pytest

from scripts.study_operator.i7_gate import GateRun, calendar, summarize

TASKS = ('q001', 'q068', 'q180', 'q016', 'q070', 'q172')


def test_pilot_is_separate_and_only_full_aggregate_decides():
    run = GateRun(TASKS, {'boot_id': 'boot', 'image_id': 'image'})
    slots = calendar(TASKS)
    assert len(slots) == 140
    assert [s['query_id'] for s in slots[:20]] == [s['query_id'] for s in slots[20:40]]
    for index, slot in enumerate(slots, 1):
        run.record(index, dict(status='success', valid=True, elapsed_s=20.,
                              **slot), started=index*25., ended=index*25.+21)
        assert run.decision()['decision'] == ('GO' if index == 140 else 'INCOMPLETE')
    result = run.decision()
    assert result['pilot_count'] == 20 and result['gate_count'] == 120
    assert result['systems']['hybrid']['n'] == 60
    assert result['systems']['no_rag']['p95'] == 20


def test_p95_is_linear_and_slow_rows_are_not_excluded():
    rows = [{'condition': 'hybrid', 'status': 'success', 'valid': True, 'elapsed_s': float(i)}
            for i in range(1, 61)]
    assert summarize(rows)['hybrid']['p95'] == pytest.approx(57.05)


@pytest.mark.parametrize('change', [{'valid': False}, {'status': 'failed'}, {'elapsed_s': float('nan')},
                                  {'identity': {'boot_id': 'other'}}, {'condition': 'wrong'}])
def test_bad_attempt_is_retained_and_cannot_resume(change):
    run = GateRun(TASKS, {'boot_id': 'boot', 'image_id': 'image'})
    row = dict(calendar(TASKS)[0], status='success', valid=True, elapsed_s=2.)
    row.update(change)
    with pytest.raises(ValueError):
        run.record(1, row, started=0, ended=3)
    assert run.rows == [row]
    assert run.decision()['decision'] == 'INCOMPLETE_TERMINAL'
    with pytest.raises(ValueError):
        run.record(2, row, started=4, ended=7)


def test_window_deadline_and_no_go_even_with_zero_errors():
    run = GateRun(TASKS, {'boot_id': 'boot', 'image_id': 'image'})
    for index, slot in enumerate(calendar(TASKS), 1):
        run.record(index, dict(slot, status='success', valid=True, elapsed_s=61.),
                   started=index*65., ended=index*65.+62)
    assert run.decision()['decision'] == 'NO-GO'


def test_schedule_matches_existing_preregistered_windows():
    from scripts.run_study_gate import schedule
    slots = calendar(TASKS)[20:]
    assert [{k: v for k, v in row.items() if k != 'phase'} for row in slots] == schedule(1, TASKS)+schedule(2, TASKS)


def test_driver_uses_call_deadlines_and_requires_final_receipt():
    from scripts.study_operator.i7_gate import collect
    class Host:
        def __init__(self):
            self.t = 0
            self.samples = []
            self.root = None
        def clock(self):
            return self.t
        def poll(self, **kwargs):
            self.samples.append(self.t)
    host = Host()
    class Channel:
        def send(self, request):
            self.request = request
        def receive(self, timeout):
            assert 0 < timeout <= 600
            host.t += 21
            if self.request == {'operation': 'complete'}:
                return {'status': 'GATE_COMPLETE', 'rows_sha256': 'wrong'}
            index = self.request['index']
            return {'status': 'OBSERVATION', 'row': dict(calendar(TASKS)[index-1], index=index,
                status='success', valid=True, elapsed_s=20.)}
    with pytest.raises(ValueError, match='FINAL'):
        collect(Channel(), host, GateRun(TASKS, {'boot_id':'boot','image_id':'image'}))


def test_gate_native_launch_is_bounded_and_not_legacy_cli():
    from scripts.study_operator.i7_gate_host import launch_command
    command = launch_command({'boot_id': 'a'*32, 'config': {'host_code':'/srv/cloudrag/i7/code'}})
    assert '--property=RuntimeMaxSec=30600' in command
    assert 'scripts.study_operator.i7_gate_host' in command
    assert '--property=ExecStopPost=/usr/sbin/shutdown -h now' in command
    assert 'scripts.run_study_gate' not in command


def test_call_and_window_limits_are_independent():
    run = GateRun(TASKS, {'boot_id':'boot','image_id':'image'})
    with pytest.raises(ValueError):
        run.record(1, dict(calendar(TASKS)[0],status='success',valid=True,elapsed_s=601.),
                   started=0,ended=601)
    run = GateRun(TASKS, {'boot_id':'boot','image_id':'image'})
    slots = calendar(TASKS)
    for index in range(1, 22):
        run.record(index,dict(slots[index-1],status='success',valid=True,elapsed_s=1.),
                   started=index*2., ended=index*2.+1)
    with pytest.raises(ValueError):
        run.record(22,dict(slots[21],status='success',valid=True,elapsed_s=1.),
                   started=7300,ended=7301)
