"""Persistent inference worker with an independent, bounded caller process."""

import ctypes
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time

from src.ui.components.session_storage import atomic_json


def worker(connection, config, parent_pid):
    # The worker dies when its owning controller dies; never terminate Ollama.
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent_pid:
            raise RuntimeError("Cannot establish parent-death protection")
    try:
        if config.get("synthetic"):
            connection.send({"ready": True})
            while True:
                request = connection.recv()
                if request is None:
                    return
                if request.get("hang"):
                    time.sleep(60)
                connection.send({"elapsed_s": 0.001, "valid": True, "error": None})
        else:
            from src.ui.components.study_pipeline import (
                configure_study_device,
                build_study_pipeline,
            )

            configure_study_device()
            from src.ui.components.interview_preparation import Preparation
            from src.ui.components.study_protocol import verify_draw
            from scripts.run_study_gate import make_app_adapter

            protocol = verify_draw(config["config_dir"])
            preparation = Preparation(
                Path(config["root"]) / "preparation",
                factory=build_study_pipeline,
                systems=("hybrid", "no_rag"),
                nli_systems=("hybrid",),
            )
            scope = config["cohort_id"]
            preparation.prepare(scope)
            adapter = make_app_adapter(
                protocol,
                lambda condition: preparation.pipeline(condition, scope),
                evidence_root=Path(config["root"]) / "requests",
            )
            connection.send({"ready": True})
            while True:
                request = connection.recv()
                if request is None:
                    return
                elapsed, valid, error = adapter(request)
                connection.send(dict(elapsed_s=elapsed, valid=valid, error=error))
    except BaseException as exc:
        connection.send(
            {"error": type(exc).__name__, "valid": False, "elapsed_s": None}
        )
    finally:
        connection.close()


class Supervisor:
    """Parent stays responsive even when inference blocks inside native code."""

    def __init__(self, config, *, call_seconds=600, preparation_seconds=900, poll=None,
                 worker_target=worker):
        self.call_seconds = call_seconds
        self.preparation_seconds = preparation_seconds
        self.poll = poll
        self.config = config
        self.context = multiprocessing.get_context("spawn")
        self.connection, child = self.context.Pipe()
        self.process = self.context.Process(
            target=worker_target, args=(child, config, os.getpid())
        )
        self.deadline = None
        self.terminal = False

    def __enter__(self):
        self.process.start()
        try:
            result = self.receive(self.preparation_seconds)
            if not result.get("ready"):
                raise RuntimeError(
                    "Worker preparation failed: " + str(result.get("error"))
                )
            return self
        except BaseException:
            self.close()
            raise

    def receive(self, seconds):
        expired = threading.Event()

        def stop_at_deadline():
            expired.set()
            if self.process.is_alive():
                self.process.terminate()

        watchdog = threading.Timer(seconds, stop_at_deadline)
        watchdog.daemon = True
        watchdog.start()
        try:
            return self._receive(seconds, expired)
        finally:
            watchdog.cancel()
            watchdog.join()

    def _receive(self, seconds, expired):
        deadline = time.monotonic() + seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or expired.is_set():
                self.terminal = True
                raise TimeoutError("Independent worker deadline reached")
            if self.connection.poll(min(remaining, 1)):
                if expired.is_set():
                    raise TimeoutError("Independent worker deadline reached")
                try:
                    return self.connection.recv()
                except EOFError:
                    if expired.is_set():
                        raise TimeoutError(
                            "Independent worker deadline reached"
                        ) from None
                    raise
            if not self.process.is_alive():
                self.terminal = True
                raise RuntimeError("Inference worker exited")
            if self.poll:
                self.poll()

    def __call__(self, planned):
        if self.terminal:
            raise RuntimeError("Terminal worker cannot resume")
        remaining = (
            self.deadline - time.monotonic() if self.deadline else self.call_seconds
        )
        if remaining <= 0:
            raise TimeoutError("Window deadline reached")
        self.connection.send(planned)
        try:
            result = self.receive(min(self.call_seconds, remaining))
        except BaseException:
            self.terminal = True
            self.close()
            raise
        return result["elapsed_s"], result["valid"], result["error"]

    def close(self):
        if self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
        self.connection.close()
        root = Path(self.config["root"])
        if root.exists():
            atomic_json(
                root / "worker-closed.json",
                dict(
                    pid=self.process.pid,
                    alive=self.process.is_alive(),
                    terminal=self.terminal,
                ),
            )

    def __exit__(self, *args):
        self.close()
