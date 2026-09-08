"""Real loopback HTTP tests. No Ollama server or model execution."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from src.generation.llm_manager import LLMManager


@pytest.fixture
def endpoint():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                if self.path == '/slow':
                    time.sleep(1.2)
                elif self.path != '/chunks':
                    time.sleep(.3)
                self.send_response(200)
                self.end_headers()
                if self.path == '/chunks':
                    for _ in range(12):
                        self.wfile.write(b'x')
                        self.wfile.flush()
                        time.sleep(.1)
                else:
                    self.wfile.write(json.dumps({'models': [{'model': 'test', 'digest': 'a' * 64}],
                                                'message': {'content': 'ok'}}).encode())
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # expected client disconnection after the timeout

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_read_gap_exceeding_limit_raises_real_readtimeout(endpoint):
    with httpx.Client(timeout=httpx.Timeout(.3, connect=1), trust_env=False) as client:
        with pytest.raises(httpx.ReadTimeout):
            client.get(endpoint + '/slow')


def test_regular_bytes_allow_total_time_above_read_limit(endpoint):
    with httpx.Client(timeout=httpx.Timeout(.5, connect=1), trust_env=False) as client:
        started = time.monotonic()
        response = client.get(endpoint + '/chunks')
        elapsed = time.monotonic() - started
    assert response.content == b'x' * 12
    assert elapsed > .5


def test_real_manager_identity_and_chat_have_independent_request_budgets(endpoint):
    calls = []
    clients = []

    class Client:
        def __init__(self, host, timeout):
            assert timeout.read == timeout.write == timeout.pool == .5
            self.http = httpx.Client(timeout=timeout, trust_env=False)
            clients.append(self.http)

        def list(self):
            calls.append('list')
            return self.http.get(endpoint + '/tags').json()

        def chat(self, **kwargs):
            calls.append('chat')
            return self.http.get(endpoint + '/chat').json()

    manager = LLMManager.__new__(LLMManager)
    manager.enforce_timeout = True
    manager.timeout = .5
    manager.num_ctx = 4096
    manager.model = 'test'
    manager.expected_model_digest = 'a' * 64
    manager._ollama_client = None
    try:
        started = time.monotonic()
        response = manager._ollama_chat(SimpleNamespace(Client=Client), options={})
        assert time.monotonic() - started > manager.timeout
        assert response['message']['content'] == 'ok'
        assert calls == ['list', 'chat']
    finally:
        for client in clients:
            client.close()
