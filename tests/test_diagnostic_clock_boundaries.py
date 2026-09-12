"""Deterministic clock boundaries; no models or measured sleep."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import measure_interview_gate as gate
from scripts import lexical_diagnostic as diagnostic

class Boundaries(unittest.TestCase):
    def test_durable_clock_inclusions_and_exclusions(self):
        for phase, inside in {
            "request": False, "observer_start": False, "event_fsync": True,
            "thread_start": True, "work": True, "thread_join": False,
            "observer_finish": False, "validation": False, "result": False,
        }.items():
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                tick = [0.0]
                in_write = [False]
                real_write, real_fsync = gate.write_new, gate.os.fsync
                def advance(name):
                    if name == phase:
                        tick[0] += 10.0
                def write(path, value):
                    advance("request" if Path(path).name == "request.json" else "result")
                    in_write[0] = True
                    try:
                        return real_write(path, value)
                    finally:
                        in_write[0] = False
                def fsync(fd):
                    if not in_write[0]:
                        advance("event_fsync")
                    return real_fsync(fd)
                class Thread:
                    def __init__(self, **kwargs): pass
                    def start(self): advance("thread_start")
                    def join(self): advance("thread_join")
                class Observer:
                    def start(self): advance("observer_start")
                    def finish(self):
                        advance("observer_finish")
                        return {}
                def work():
                    advance("work")
                    return {"status": "success"}
                with patch.object(gate.time, "perf_counter", lambda: tick[0]), \
                     patch.object(gate, "write_new", write), \
                     patch.object(gate.os, "fsync", fsync), \
                     patch.object(gate.threading, "Thread", Thread):
                    row = gate.measure_attempt(folder, {}, work, observer=Observer(),
                        validate_after=lambda: advance("validation"))
                self.assertEqual(row["elapsed_s"], 10.0 if inside else 0.0)
                self.assertEqual(tick[0], 10.0)

    def test_trace_setup_excluded_selection_serialization_export_included(self):
        tick = [0.0]
        def add(seconds): tick[0] += seconds
        class Trace:
            def __init__(self, pipeline): add(11)
            def __enter__(self):
                add(13)
                return self
            def __exit__(self, *args): add(17)
            def export(self):
                add(7)
                return {"proof": True}
        def dump(**kwargs):
            add(5)
            return {"answer": "Complete technical answer", "confidence": "HIGH"}
        def query(question):
            add(3)
            return SimpleNamespace(model_dump=dump)
        subject = SimpleNamespace(query=query)
        def select():
            add(2)
            return subject
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(gate.time, "perf_counter", lambda: tick[0]), \
             patch.object(diagnostic, "PipelineTrace", Trace):
            row = diagnostic.measure_traced_attempt(folder,
                {"query": {"question": "technical test"}}, subject, before_query=select)
        self.assertEqual(row["elapsed_s"], 2 + 3 + 5 + 7)
        self.assertEqual(tick[0], 11 + 13 + 17 + 2 + 3 + 5 + 7)
        self.assertEqual(row["diagnostic_trace"], {"proof": True})

if __name__ == "__main__":
    unittest.main(verbosity=2)
