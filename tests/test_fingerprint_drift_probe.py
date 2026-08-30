"""Offline tests for the exp19b idle fingerprint-drift probe."""

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import probe_fingerprint_drift as probe  # noqa: E402


def _clock():
    instant = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)
    return lambda: instant


def test_stable_injected_fingerprints_are_logged_as_idle_stable():
    samples = iter([("same", 101), ("same", 102), ("same", 103)])

    report = probe.collect_probe(
        3, 0, capture_fn=lambda: next(samples), now_fn=_clock())

    assert report["transitions_total"] == 2
    assert report["transitions_changed"] == 0
    assert report["verdict"] == "huella estable en idle"
    assert [row["identical_to_previous"] for row in report["captures"]] == [False, True, True]
    assert [row["tokens_out"] for row in report["captures"]] == [101, 102, 103]
    assert all(row["timestamp"] == "2026-08-22T12:00:00+00:00"
               for row in report["captures"])


def test_changed_injected_fingerprints_are_counted_without_a_recommendation():
    samples = iter([("state-a", 90), ("state-b", 91), ("state-b", 92), ("state-c", 93)])

    report = probe.collect_probe(
        4, 0, capture_fn=lambda: next(samples), now_fn=_clock())

    assert report["transitions_total"] == 3
    assert report["transitions_changed"] == 2
    assert report["verdict"] == "huella deriva en idle"
    assert [row["identical_to_previous"] for row in report["captures"]] == [
        False, False, True, False]
    assert "recommend" not in json_keys(report)


def json_keys(value):
    """Flatten JSON object keys so a design recommendation cannot slip into the report."""
    if isinstance(value, dict):
        return {key for key, item in value.items()} | set().union(
            *(json_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(json_keys(item) for item in value), set())
    return set()
