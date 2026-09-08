"""Hard-fault extraction validates capture completeness and preserves unknown attribution."""
from datetime import datetime

import pytest

from scripts import gate_etw as etw
from scripts import measure_interview_gate as gate
from scripts import observe_interview_gate as observe


def event(guid='', opcode=0, data='', extra='', version=2):
    return f'''<Event xmlns="{etw.NS['e']}"><System><Opcode>{opcode}</Opcode><Version>{version}</Version>
    <TimeCreated SystemTime="2026-09-08T00:00:01+00:00"/></System><EventData>{data}</EventData>{extra}
    <ExtendedTracingInfo xmlns="{etw.NS['t']}"><EventGuid>{guid}</EventGuid></ExtendedTracingInfo></Event>'''


def data(**values):
    return ''.join(f'<Data Name="{k}">{v}</Data>' for k, v in values.items())


def test_extracts_hard_fault_and_thread_prefix_without_using_invalid_header_pid(tmp_path):
    header = event(data=data(EventsLost=0, BuffersLost=0))
    thread = event(etw.THREAD, 1, version=3, extra='''<ProcessingErrorData>
        <EventPayload>7B000000C801000000000000</EventPayload></ProcessingErrorData>''')
    fault = event(etw.FAULT, 32, data=data(TThreadId=456, ByteCount=4096, FileObject='0x1', InitialTime=0))
    path = tmp_path / 'trace.xml'
    path.write_text('<Events>' + header + thread + fault + '</Events>', encoding='utf-8')
    result = etw.parse_trace(path)
    assert result['hard_fault_count'] == 1
    assert result['hard_faults'][0]['pid'] == 123
    assert result['thread_identity_prefix_decoded'] == 1
    path.write_text('<Events>' + header + fault + '</Events>', encoding='utf-8')
    assert etw.parse_trace(path)['unattributed'] == 1


@pytest.mark.parametrize('metadata', ['', data(EventsLost=1, BuffersLost=0), data(EventsLost=0, BuffersLost=1)])
def test_absent_or_lost_capture_is_not_zero_faults(tmp_path, metadata):
    path = tmp_path / 'trace.xml'
    path.write_text('<Events>' + event(data=metadata) + '</Events>', encoding='utf-8')
    with pytest.raises(ValueError):
        etw.parse_trace(path)


def test_rate_uses_partial_interval_and_excludes_capture_setup_and_stop():
    start = '2026-09-08T00:00:00+00:00'
    origin = datetime.fromisoformat(start).timestamp()
    faults = [dict(timestamp_s=origin + t, pid=None, bytes_read=4096) for t in (-1, 1, 5.5, 6, 8)]
    rows = etw.intervals(faults, start, 6)
    assert [r['count'] for r in rows] == [1, 1]
    assert [r['hard_faults_per_s'] for r in rows] == [.2, 1]


@pytest.mark.parametrize('fail', [False, True])
def test_trace_wraps_response_and_failure_keeps_actual_answer(tmp_path, monkeypatch, fail):
    calls = []

    class Capture:
        def __init__(self, root):
            pass

        def start(self):
            calls.append('trace_start')

        def finish(self, start, elapsed):
            calls.append(('trace_finish', elapsed))
            assert datetime.fromisoformat(start)
            if fail:
                raise ValueError('lost events')
            return {'hard_fault_count_in_response': 3}

    monkeypatch.setattr(etw, 'Capture', Capture)
    def sensor():
        return dict(monotonic_s=0, ac=True, scheme=observe.BALANCED,
            overlay=observe.BEST_PERFORMANCE, cpu_percent=0, gpu={}, ram_available_bytes=1, processes=[], errors=[])
    observer = observe.Observer(tmp_path / 'telemetry.jsonl', sampler=sensor, capture_enabled=True)

    def work():
        calls.append('work')
        return dict(status='success', answer='preserved')

    row = gate.measure_attempt(tmp_path, dict(system='hybrid', phase='cold', index=0), work, observer=observer)
    assert calls == ['trace_start', 'work', ('trace_finish', row['elapsed_s'])]
    assert row['status'] == 'success' and row['answer'] == 'preserved'
    assert row['conditions_invalid'] is fail
    if fail:
        assert 'hard_fault_capture_invalid' in row['control_reasons']
    else:
        assert row['hard_fault_count_in_response'] == 3


def test_changed_memory_code_or_mode_blocks_cohort(monkeypatch):
    monkeypatch.setenv('CLOUDRAG_MEMORY_TRACE', '1')
    monkeypatch.setattr(gate, 'digest', lambda path: 'same')
    protocol = {'controls': {'memory_trace': True, 'memory_sha256': 'same', 'etw_sha256': 'same', 'profile_sha256': 'same'}}
    gate.check_environment(protocol)
    monkeypatch.setenv('CLOUDRAG_MEMORY_TRACE', '0')
    with pytest.raises(ValueError, match='instrumentation changed'):
        gate.check_environment(protocol)


def test_smoke_saves_only_completed_capture(tmp_path, monkeypatch):
    class Capture:
        def __init__(self, root):
            assert root == tmp_path

        def start(self):
            pass

        def finish(self, at, elapsed):
            assert at and elapsed > 0
            return {'hard_fault_count_in_response': 7}

    monkeypatch.setattr(etw, 'Capture', Capture)
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    etw.smoke(tmp_path)
    assert gate.read_json(tmp_path / 'smoke-result.json')['hard_fault_count_in_response'] == 7


def test_sampler_keeps_graphics_gpu_process_listing(monkeypatch):
    import io
    from scripts import gate_memory
    monkeypatch.setenv('LOCALAPPDATA', 'C:/synthetic')
    monkeypatch.setattr(observe, 'windows_state', lambda: dict(cpu_ticks={'idle': 1, 'total': 2}))
    monkeypatch.setattr(observe, 'windows_processes', lambda: [])
    monkeypatch.setattr(gate_memory, 'system_memory', lambda: {})
    monkeypatch.setattr(gate_memory, 'PagingCounters', lambda: type('Counter', (), {'sample': lambda self: {}})())
    monkeypatch.setattr(observe.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'{"models": []}'))

    def command(args):
        if args == ['nvidia-smi']:
            return 'GPU 0 PID 1234 C+G AnyDesk.exe'
        if args[0] == 'nvidia-smi':
            return ','.join('0' for _ in observe.GPU_FIELDS)
        return 'No running models'

    monkeypatch.setattr(observe, 'command', command)
    row = observe.Sampler()()
    assert row['errors'] == []
    assert '1234 C+G AnyDesk.exe' in row['gpu_process_listing']
