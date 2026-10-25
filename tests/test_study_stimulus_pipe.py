import io
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.study_operator.stimulus_pipe import PrivatePipe


def pipe(raw, *, poll=lambda: None):
    cleaned = []
    process = SimpleNamespace(stdout=io.BytesIO(raw),stdin=io.BytesIO())
    result = PrivatePipe(process, lambda: cleaned.append('owned-container'), poll)
    return result, process, cleaned


def test_private_frames_and_control_are_preserved_without_logging(capsys):
    channel, process, cleaned = pipe(b'{"status":"OBSERVATION","private_text":"synthetic-only"}\n')
    channel.send(dict(index=1))
    assert process.stdin.getvalue() == b'{"index": 1}\n'
    assert channel.receive(1)['private_text'] == 'synthetic-only'
    channel.close()
    channel.close()
    assert cleaned == ['owned-container'] and capsys.readouterr().out == ''


@pytest.mark.parametrize('raw', [b'',b'{}',b'not json\n',b'[]\n',b'X'*2000+b'\n'])
def test_bad_frame_is_terminal_and_cleanup_happens_once(raw):
    channel, _, cleaned = pipe(raw)
    channel.maximum = 1024
    with pytest.raises(ValueError):
        channel.receive(1)
    with pytest.raises(ValueError):
        channel.send(dict(index=1))
    channel.close()
    assert cleaned == ['owned-container']


def test_independent_timer_stops_own_blocked_process_without_a_reply():
    process = subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'],
                               stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    stopped = []
    def cleanup():
        stopped.append(process.pid)
        process.terminate()
        process.wait(timeout=3)
    channel = PrivatePipe(process, cleanup, lambda: None)
    try:
        with pytest.raises(TimeoutError):
            channel.receive(.05)
        assert process.poll() is not None and stopped == [process.pid]
        with pytest.raises(ValueError):
            channel.receive(.05)
    finally:
        channel.close()


def test_telemetry_rejection_stops_before_publishing_a_private_response():
    def reject():
        raise ValueError('Foreign GPU process')
    channel, _, cleaned = pipe(b'{"status":"OBSERVATION"}\n',poll=reject)
    with pytest.raises(ValueError, match='Foreign'):
        channel.receive(1)
    channel.close()
    assert cleaned == ['owned-container']


def test_full_reader_queue_does_not_prevent_owned_shutdown():
    channel, _, cleaned = pipe(b'{}\n'*10)
    channel.close()
    assert not channel.reader.is_alive() and cleaned == ['owned-container']
