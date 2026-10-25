"""Diagnostic tracing must preserve requests, results and original exceptions."""
from types import SimpleNamespace

import httpx
import pytest

from scripts import diagnose_interview_timeout as diagnostic
from scripts import measure_interview_gate as gate


def test_trace_preserves_request_result_and_server_metrics(tmp_path):
    calls = []
    response = {"message": {"content": "answer"}, "eval_count": 30, "eval_duration": 900}

    def chat(**kwargs):
        calls.append(kwargs)
        return response

    client = diagnostic.TracedClient(SimpleNamespace(chat=chat), tmp_path)
    request = {"model": "synthetic", "options": {"seed": 42}, "messages": [{"content": "q"}]}
    assert client.chat(**request) is response
    assert calls == [request]
    record = gate.read_json(next(tmp_path.glob("*.json")))
    assert record["request"] == request
    assert record["server"]["eval_count"] == 30
    assert record["method"] == "chat"
    assert record["status"] == "success"


def test_trace_persists_readtimeout_and_reraises_same_exception(tmp_path):
    error = httpx.ReadTimeout("synthetic read timeout")

    def chat(**kwargs):
        raise error

    client = diagnostic.TracedClient(SimpleNamespace(chat=chat), tmp_path)
    with pytest.raises(httpx.ReadTimeout) as raised:
        client.chat(model="synthetic")
    assert raised.value is error
    record = gate.read_json(next(tmp_path.glob("*.json")))
    assert record["exception_type"] == "httpx.ReadTimeout"
    assert record["status"] == "error"
    assert record["elapsed_s"] >= 0


def test_trace_distinguishes_identity_failure_and_never_overwrites(tmp_path):
    error = httpx.ConnectError("synthetic identity failure")

    def listing():
        raise error

    client = diagnostic.TracedClient(SimpleNamespace(list=listing), tmp_path)
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            client.list()
    records = [gate.read_json(p) for p in tmp_path.glob("*.json")]
    assert len(records) == 2
    assert all(r["method"] == "list" for r in records)
    assert all(r["exception_type"] == "httpx.ConnectError" for r in records)
