import subprocess
import threading

import pytest

from scripts.cloud_entrypoint import streamlit_command
from scripts.study_operator.app_runtime import relays, run_child


def test_relays_keep_running_during_app_and_close_after_error():
    events = []

    class Server:
        def __init__(self, listen, target):
            self.listen, self.target = listen, target
            self.stop = threading.Event()
            events.append(('create', listen, target))

        def serve_forever(self):
            self.stop.wait()

        def shutdown(self):
            events.append(('shutdown', self.listen))
            self.stop.set()

        def server_close(self):
            events.append(('close', self.listen))

    with pytest.raises(RuntimeError):
        with relays('private-ollama-socket', 'private-web-socket', factory=Server):
            assert len(events) == 2
            raise RuntimeError('synthetic app failure')
    assert sum(event[0] == 'shutdown' for event in events) == 2
    assert sum(event[0] == 'close' for event in events) == 2


def test_child_output_discarded_and_failure_reaps_child():
    events = []

    class Child:
        def wait(self, timeout=None):
            if timeout is None:
                raise KeyboardInterrupt()
            if timeout == 10:
                raise subprocess.TimeoutExpired('synthetic child', 10)
            events.append('reaped')
            return 0

        def poll(self):
            return None

        def terminate(self):
            events.append('terminate')

        def kill(self):
            events.append('kill')

    def launch(arguments, **options):
        assert options == dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return Child()

    with pytest.raises(KeyboardInterrupt):
        run_child(['synthetic-child'], launch=launch)
    assert events == ['terminate', 'kill', 'reaped']


def test_public_hostname_retains_loopback_bind_and_disables_telemetry():
    command = streamlit_command({'hostname': 'fixture.sslip.io'})
    assert command[command.index('--server.address')+1] == '127.0.0.1'
    assert command[command.index('--browser.serverAddress')+1] == 'fixture.sslip.io'
    assert command[command.index('--browser.serverPort')+1] == '443'
    assert command[command.index('--browser.gatherUsageStats')+1] == 'false'
