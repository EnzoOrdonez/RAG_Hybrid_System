import json
from types import SimpleNamespace

import pytest

from scripts.study_operator.service_gateway import FreshRunner, OPTIONS


class Backend:
    def __init__(self):
        self.pid = 10
        self.loaded = True
        self.requests = []
        self.digest = 'a' * 64
        self.fail_unload = False
        self.reuse_pid = False

    def runners(self):
        return {self.pid} if self.loaded else set()

    def request(self, method, path, body, timeout):
        self.requests.append((method, path, body))
        assert timeout > 0
        if method == 'GET':
            return 200, json.dumps({'models': [dict(digest=self.digest, context_length=4096)] if self.loaded else []}).encode()
        if json.loads(body).get('keep_alive') == 0:
            if not self.fail_unload:
                self.loaded = False
            return 200, b'{"done":true}'
        self.loaded = True
        if not self.reuse_pid:
            self.pid += 1
        return 200, b'{"done":true,"message":{"content":"live synthetic response"}}'


def gateway(tmp_path, backend, **kwargs):
    return FreshRunner(backend, model='granite4.1:8b', digest='a' * 64,
                       state_path=tmp_path / 'state.json', boot_id='boot-fixture', **kwargs)


def request(text):
    # Deliberate whitespace proves forwarding bytes, not a reconstructed payload.
    return json.dumps(dict(model='granite4.1:8b', messages=[{'role': 'user', 'content': text}],
                           options=OPTIONS, keep_alive='30m', stream=False), indent=4).encode()


def test_every_generation_uniformly_resets_and_forwards_exact_original_bytes(tmp_path):
    backend = Backend()
    service = gateway(tmp_path, backend)
    bodies = [request('synthetic task'), request('arbitrary synthetic free query')]
    for body in bodies:
        assert json.loads(service.forward('/api/chat', body)[1])['done']
    posts = [(path, body) for method, path, body in backend.requests if method == 'POST']
    assert [path for path, body in posts] == ['/api/generate', '/api/chat'] * 2
    assert [body for path, body in posts if path == '/api/chat'] == bodies
    marker = json.loads((tmp_path / 'state.json').read_text())
    assert marker['phase'] == 'RESIDENT' and marker['runner_pids'] == [12]
    assert 'synthetic' not in json.dumps(marker)


@pytest.mark.parametrize('field,value', [('num_predict', 1023), ('temperature', .1), ('seed', 43), ('num_ctx', 8192)])
def test_generation_policy_rejected_before_backend_call(tmp_path, field, value):
    backend = Backend()
    service = gateway(tmp_path, backend)
    payload = json.loads(request('text'))
    payload['options'][field] = value
    with pytest.raises(ValueError, match='Frozen'):
        service.forward('/api/chat', json.dumps(payload).encode())
    assert not backend.requests


def test_unload_deadline_fails_closed_without_serving_generation(tmp_path):
    backend = Backend()
    backend.fail_unload = True
    ticks = [0.0]
    service = gateway(tmp_path, backend, clock=lambda: ticks[0], sleep=lambda _: ticks.__setitem__(0, ticks[0] + 11))
    with pytest.raises(TimeoutError):
        service.forward('/api/chat', request('text'))
    assert not any(path == '/api/chat' for method, path, body in backend.requests)
    assert json.loads((tmp_path / 'state.json').read_text())['phase'] == 'FAILED'
    with pytest.raises(RuntimeError, match='restart'):
        service.forward('/api/chat', request('text'))


def test_reused_pid_or_wrong_digest_does_not_return_a_response(tmp_path):
    for fault in ('reuse_pid', 'digest'):
        backend = Backend()
        if fault == 'reuse_pid':
            backend.reuse_pid = True
        else:
            backend.digest = 'b' * 64
        service = gateway(tmp_path, backend)
        with pytest.raises(ValueError, match='identity'):
            service.forward('/api/chat', request('text'))


def test_empty_administrative_preload_passes_bytes_without_generating_text(tmp_path):
    backend = Backend()
    service = gateway(tmp_path, backend)
    body = b'{"model":"granite4.1:8b", "keep_alive":"30m"}'
    service.forward('/api/generate', body)
    assert backend.requests == [('POST', '/api/generate', body)]


def test_gate_timer_includes_unload_and_generation_wait(tmp_path):
    from scripts.run_study_gate import make_app_adapter

    backend = Backend()
    ticks = [0.0]
    original = backend.request

    def measured(method, path, body, timeout):
        if method == 'POST':
            ticks[0] += 2 if path == '/api/generate' else 3
        return original(method, path, body, timeout)

    backend.request = measured
    service = gateway(tmp_path, backend, clock=lambda: ticks[0], sleep=lambda _: None)

    def query(question):
        service.forward('/api/chat', request(question))
        return SimpleNamespace(answer='synthetic answer', error=None, confidence='HIGH',
                               sources=[], hallucination_report={'method': 'nli'})

    adapter = make_app_adapter({'queries': {'q': {'question': 'synthetic query'}}},
                              lambda _: SimpleNamespace(query=query), clock=lambda: ticks[0])
    assert adapter(dict(query_id='q', condition='no_rag', attempt_id='synthetic')) == (5.0, True, None)
