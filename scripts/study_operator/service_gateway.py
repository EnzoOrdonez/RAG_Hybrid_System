"""Uniform fresh-runner service. Original generation request bytes pass unchanged."""
import argparse
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import socketserver
import subprocess
import threading
import time
import urllib.request
import uuid


OPTIONS = dict(temperature=0.0, num_predict=1024, seed=42, num_ctx=4096)


def save_state(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Backend:
    """Fixed loopback target. Never accepts a URL or headers from the app."""
    def __init__(self, container):
        self.container = container

    def request(self, method, path, body, timeout):
        request = urllib.request.Request('http://127.0.0.1:11434' + path, data=body,
                                         headers={'Content-Type': 'application/json'}, method=method)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read(2 * 1024 * 1024 + 1)
            if len(content) > 2 * 1024 * 1024:
                raise ValueError('Upstream response too large')
            return response.status, content

    def runners(self):
        text = subprocess.check_output(['docker', 'top', self.container, '-eo', 'pid,args'],
                                       text=True, timeout=3)
        return {int(row.split()[0]) for row in text.splitlines()[1:]
                if 'ollama runner' in row or 'ollama_llama_server' in row}


class FreshRunner:
    def __init__(self, backend, *, model, digest, state_path, boot_id, clock=time.monotonic, sleep=time.sleep):
        self.backend, self.model, self.digest = backend, model, digest
        self.state_path, self.boot_id = Path(state_path), boot_id
        self.clock, self.sleep = clock, sleep
        self.lock = threading.Lock()
        self.failed = False
        self.sequence = 0
        self.state('STARTING', None, None)

    def state(self, phase, request_id, deadline, **details):
        self.sequence += 1
        save_state(self.state_path, dict(schema_version=1, mode='fresh_runner', phase=phase,
            request_id=request_id, sequence=self.sequence, boot_id=self.boot_id,
            written_monotonic_s=self.clock(), deadline_monotonic_s=deadline, **details))

    def models(self):
        code, data = self.backend.request('GET', '/api/ps', None, 3)
        if code != 200:
            raise ValueError('Cannot verify model state')
        return json.loads(data)['models']

    def verify_model(self, models):
        return (len(models) == 1 and models[0].get('digest', '').removeprefix('sha256:') == self.digest
                and models[0].get('context_length') == 4096)

    def forward(self, path, body):
        payload = json.loads(body)
        if payload.get('model') != self.model or payload.get('stream', False):
            raise ValueError('Generation policy differs')
        content = bool(payload.get('prompt')) or any(m.get('content') for m in payload.get('messages', []))
        if path == '/api/chat' and not content:
            raise ValueError('Empty chat request')
        if content and payload.get('options') != OPTIONS:
            raise ValueError('Frozen generation options differ')
        if not content and (path != '/api/generate' or set(payload) - {'model', 'keep_alive', 'stream', 'options'}):
            raise ValueError('Administrative operation differs')
        entered = self.clock()
        total_deadline = entered + 600
        if not self.lock.acquire(timeout=600):
            raise TimeoutError('Service queue deadline')
        request_id = uuid.uuid4().hex
        try:
            if self.failed:
                raise RuntimeError('Service requires verified restart')
            if not content:
                # Empty administrative warmup is not an answer. Its bytes also pass unchanged.
                return self.backend.request('POST', path, body, max(.001, total_deadline - self.clock()))
            old = self.backend.runners()
            reset_deadline = min(self.clock() + 10, total_deadline)
            self.state('RESETTING', request_id, reset_deadline, previous_runner_pids=sorted(old))
            code, result = self.backend.request('POST', '/api/generate',
                json.dumps(dict(model=self.model, keep_alive=0)).encode(), max(.001, reset_deadline - self.clock()))
            if code != 200 or not json.loads(result).get('done'):
                raise ValueError('Unload not confirmed')
            while self.models() or self.backend.runners():
                if self.clock() >= reset_deadline:
                    raise TimeoutError('Previous runner did not exit')
                self.sleep(.05)
            if self.clock() >= reset_deadline:
                raise TimeoutError('Runner reset deadline')
            load_deadline = min(self.clock() + 30, total_deadline)
            self.state('LOADING', request_id, load_deadline, previous_runner_pids=sorted(old))
            done = threading.Event()
            output = []

            def invoke():
                try:
                    output.append(self.backend.request('POST', path, body, max(.001, total_deadline - self.clock())))
                except BaseException as exc:
                    output.append(exc)
                finally:
                    done.set()

            thread = threading.Thread(target=invoke, daemon=True)
            thread.start()
            fresh = set()
            while not done.is_set() or not fresh:
                if self.clock() >= (total_deadline if fresh else load_deadline):
                    raise TimeoutError('Fresh runner generation deadline')
                models = self.models()
                runners = self.backend.runners()
                if models:
                    if not self.verify_model(models) or not runners or runners & old:
                        raise ValueError('Fresh runner identity differs')
                    if not fresh:
                        fresh = runners
                        self.state('GENERATING', request_id, total_deadline, runner_pids=sorted(fresh))
                    elif runners != fresh:
                        raise ValueError('Runner changed during generation')
                elif fresh:
                    raise ValueError('Runner disappeared during generation')
                if done.is_set() and output and isinstance(output[0], BaseException):
                    raise output[0]
                self.sleep(.05)
            thread.join(timeout=.1)
            if not output or isinstance(output[0], BaseException):
                raise RuntimeError('Generation not confirmed')
            code, result = output[0]
            if code != 200 or not json.loads(result).get('done') or not self.verify_model(self.models()):
                raise ValueError('Completed generation not verified')
            self.state('RESIDENT', request_id, None, runner_pids=sorted(fresh))
            return code, result
        except BaseException:
            self.failed = True
            self.state('FAILED', request_id, None)
            raise
        finally:
            self.lock.release()


def serve(settings):
    backend = Backend(settings['ollama_container'])
    gateway = FreshRunner(backend, model=settings['model'], digest=settings['digest'],
        state_path=settings['state_path'], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, content):
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            try:
                if self.path not in {'/api/tags', '/api/version', '/api/ps'}:
                    raise ValueError('Route not permitted')
                self.respond(*backend.request('GET', self.path, None, 5))
            except Exception:
                self.respond(503, b'{"error":"SERVICE_STATUS_NOT_VERIFIED"}')

        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if self.path not in {'/api/chat', '/api/generate'} or not 0 < size <= 256 * 1024:
                    raise ValueError('Route or size not permitted')
                self.respond(*gateway.forward(self.path, self.rfile.read(size)))
            except Exception:
                self.respond(503, b'{"error":"SERVICE_GENERATION_NOT_VERIFIED"}')

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            pass

    with Server(settings['socket'], Handler) as server:
        os.chmod(settings['socket'], 0o660)
        server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', required=True)
    args = parser.parse_args()
    serve(json.loads(Path(args.settings).read_text(encoding='utf-8')))
