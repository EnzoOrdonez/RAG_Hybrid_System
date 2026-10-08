"""Host-owned private pipe; deadlines stop only its retained measurement container."""
import json
import queue
import threading
import time


class PrivatePipe:
    def __init__(self, process, stop_owned, poll, *, maximum_frame_bytes=16*1024*1024):
        if not callable(stop_owned) or not callable(poll) or not 1024 <= maximum_frame_bytes <= 16*1024*1024:
            raise ValueError('Owned cleanup, live telemetry and bounded private frames required')
        self.process, self.stop_owned, self.poll = process, stop_owned, poll
        self.maximum = maximum_frame_bytes
        self.frames = queue.Queue(maxsize=2)
        self.terminal = False
        self.stop_lock = threading.Lock()
        self.stopped = False
        self.reader_closed = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        while True:
            try:
                raw = self.process.stdout.readline(self.maximum+1)
                if not raw or len(raw) > self.maximum or not raw.endswith(b'\n'):
                    raise ValueError('Private frame invalid or EOF')
                frame = json.loads(raw)
                if not isinstance(frame, dict):
                    raise ValueError('Private frame must be an object')
                self._offer((True, frame))
            except BaseException as exc:
                self._offer((False, type(exc).__name__))
                return

    def _offer(self, item):
        while not self.reader_closed.is_set():
            try:
                self.frames.put(item, timeout=.1)
                return
            except queue.Full:
                continue

    def stop(self):
        with self.stop_lock:
            if self.stopped:
                return
            self.stopped = True
        self.stop_owned()  # Fixed owned container action, never a process-name kill.

    def send(self, value):
        if self.terminal or self.stopped:
            raise ValueError('Terminal private pipe cannot resume')
        raw = (json.dumps(value)+'\n').encode()
        if len(raw) > 128:
            raise ValueError('Control request exceeds private protocol bound')
        try:
            self.process.stdin.write(raw)
            self.process.stdin.flush()
        except BaseException:
            self.terminal = True
            self.stop()
            raise ValueError('Private control pipe failed') from None

    def receive(self, seconds):
        if self.terminal or self.stopped or not 0 < seconds <= 900:
            raise ValueError('Finite admitted private receive required')
        expired = threading.Event()

        def timeout():
            expired.set()
            self.stop()

        timer = threading.Timer(seconds, timeout)
        timer.daemon = True
        timer.start()
        deadline = time.monotonic()+seconds
        try:
            self.poll()
            while True:
                left = deadline-time.monotonic()
                if left <= 0 or expired.is_set():
                    raise TimeoutError('Independent cold container deadline reached')
                try:
                    valid, value = self.frames.get(timeout=min(left,1))
                except queue.Empty:
                    self.poll()
                    continue
                if expired.is_set():
                    raise TimeoutError('Independent cold container deadline reached')
                self.poll()
                if not valid:
                    raise ValueError('Private response pipe failed')
                return value
        except BaseException:
            self.terminal = True
            self.stop()
            raise
        finally:
            timer.cancel()
            timer.join(timeout=1)

    def close(self):
        self.terminal = True
        self.reader_closed.set()
        self.stop()
        self.reader.join(timeout=1)
