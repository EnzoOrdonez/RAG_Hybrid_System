"""Residence leases must be evaluated when each live observation was collected."""

from datetime import datetime, timezone
import io
from types import SimpleNamespace

import pytest

from scripts.study_gate_environment import assess


BASE = 1_800_000_000


def history(seconds=1625):
    return [
        dict(
            at=datetime.fromtimestamp(BASE + elapsed, timezone.utc).isoformat(),
            monotonic_s=elapsed,
            cpu_percent=0,
            errors=[],
            processes=[],
            gpu={"utilization.gpu": "0"},
            gpu_pids=[],
            ollama_ps_api={
                "models": [dict(
                    digest="d", context_length=4096,
                    expires_at=datetime.fromtimestamp(
                        BASE + elapsed + 1800, timezone.utc
                    ).isoformat(),
                )]
            },
        )
        for elapsed in range(0, seconds + 1, 5)
    ]


def test_renewed_lease_history_survives_first_lease_age(monkeypatch):
    monkeypatch.setattr("scripts.study_gate_environment.time.time", lambda: BASE + 1625)
    assert assess(history(), {"model_digest": "d"}) == []


def test_short_lease_at_historical_observation_is_still_rejected(monkeypatch):
    monkeypatch.setattr("scripts.study_gate_environment.time.time", lambda: BASE + 1625)
    rows = history()
    rows[0]["ollama_ps_api"]["models"][0]["expires_at"] = datetime.fromtimestamp(
        BASE + 179, timezone.utc
    ).isoformat()
    assert "residency_lease" in assess(rows, {"model_digest": "d"})


def test_current_expired_lease_is_rejected_even_if_sample_was_healthy(monkeypatch):
    monkeypatch.setattr("scripts.study_gate_environment.time.time", lambda: BASE + 1801)
    assert "residency_lease" in assess(history(seconds=0), {"model_digest": "d"})


@pytest.mark.parametrize("stamp", [None, "invalid", "2027-01-01T00:00:00"])
def test_missing_invalid_or_naive_observation_clock_fails_closed(monkeypatch, stamp):
    monkeypatch.setattr("scripts.study_gate_environment.time.time", lambda: BASE)
    rows = history(seconds=0)
    rows[0]["at"] = stamp
    assert "telemetry_error" in assess(rows, {"model_digest": "d"})


def test_linux_sampler_provides_the_aware_observation_clock(monkeypatch):
    from scripts import study_gate_environment as environment

    monkeypatch.setattr(environment, "Path", lambda _: SimpleNamespace(
        read_text=lambda: "cpu 1 1 1 100 0 0 0 0", iterdir=lambda: []
    ))
    monkeypatch.setattr(environment, "command", lambda _: "0")
    monkeypatch.setattr(environment.urllib.request, "urlopen", lambda *args, **kw:
        io.BytesIO(b'{"models": []}'))
    sampled = environment.LinuxSampler()()
    assert datetime.fromisoformat(sampled["at"]).tzinfo is not None
    assert sampled["ollama_ps_api"] == {"models": []}
