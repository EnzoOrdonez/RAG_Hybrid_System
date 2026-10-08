"""Bounded cold-query worker; live admission and host telemetry are mandatory."""
import ctypes
import os
from pathlib import Path
import signal
import time

from scripts.study_gate_supervisor import Supervisor
from scripts.study_operator.stimulus_collection import live_prepare
from scripts.study_operator.stimulus_evidence import verify_boot
from src.ui.components.session_storage import atomic_json


def worker(connection, config, parent_pid):
    if os.name != 'posix':
        connection.close()
        return  # Never initiate a local Windows model measurement.
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent_pid:
            raise ValueError('Worker parent-death protection unavailable')
        if config['deployment'].get('purpose') != 'technical':
            raise ValueError('Acceptance collection accepts synthetic technical sessions only')
        collector = live_prepare(config['deployment'], config['boot_index'])
        connection.send(dict(ready=True, inventory=collector.inventory, initial=collector.initial))
        while True:
            index = connection.recv()
            if index is None:
                return
            row = collector.call(index)
            connection.send(dict(row=row, proof=collector.proof()))
    except BaseException as exc:
        # Never forward a query, credentials or model exception text.
        connection.send(dict(error=type(exc).__name__, valid=False))
    finally:
        connection.close()


class ColdSupervisor(Supervisor):
    """Host caller must supply cold admission and continuous contamination checks.

    This is a collector, not a GO decision. Deployment-bound host orchestration
    must run admission after preparation without generating a warmup response.
    """
    def __init__(self, config, *, admission, poll, worker_target=worker,
                 call_seconds=600, preparation_seconds=900, boot_seconds=7200):
        if not callable(admission) or not callable(poll) or not 0 < boot_seconds <= 7200:
            raise ValueError('Cold admission, telemetry and bounded boot required')
        super().__init__(config, call_seconds=call_seconds, preparation_seconds=preparation_seconds,
                         poll=poll, worker_target=worker_target)
        self.admission, self.boot_seconds = admission, boot_seconds
        self.initial, self.proof = None, None

    def __enter__(self):
        self.process.start()
        try:
            self.initial = self.receive(self.preparation_seconds)
            if not self.initial.get('ready'):
                raise ValueError('Cold preparation failed')
            self.admission(self.initial)
            self.deadline = time.monotonic()+self.boot_seconds
            return self
        except BaseException:
            self.terminal = True
            self.close()
            raise

    def __call__(self, index):
        if self.terminal or self.deadline is None:
            raise ValueError('Cold worker is not admitted or already terminal')
        try:
            remaining = self.deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Cold boot deadline reached')
            self.poll()  # Check every call, including those shorter than the polling interval.
            self.connection.send(index)
            result = self.receive(min(self.call_seconds, remaining))
            self.poll()
            if result.get('error') or 'row' not in result:
                raise ValueError('Cold call failed; preserve the incomplete boot')
            self.proof = result['proof']
            return result['row']
        except BaseException:
            self.terminal = True
            self.close()
            raise

    def save_complete(self, destination, protocol_config):
        if self.terminal or self.proof is None or self.proof.get('status') != 'COMPLETE':
            raise ValueError('Terminal or unmeasured boot cannot complete')
        verify_boot(self.proof, protocol_config)
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError('Boot evidence already exists; no replay')
        atomic_json(destination, self.proof)
        return dict(status='BOOT_COMPLETE_NOT_ACCEPTANCE', mode=self.proof['mode'])
