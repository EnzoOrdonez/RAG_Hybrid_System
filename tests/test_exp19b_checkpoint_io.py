"""Atomic checkpoint I/O for the long-running exp19b selector."""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import select_exp19b_evidence as selector  # noqa: E402


def test_atomic_write_retries_one_transient_replace_failure(monkeypatch, tmp_path):
    destination = tmp_path / "selection_ids.partial.json"
    destination.write_text('{"old": true}', encoding="utf-8")
    real_replace = os.replace
    attempts = []
    sleeps = []

    def flaky_replace(source, target):
        attempts.append((Path(source), Path(target)))
        if len(attempts) == 1:
            assert destination.read_text(encoding="utf-8") == '{"old": true}'
            raise OSError(22, "transient Windows lock")
        return real_replace(source, target)

    monkeypatch.setattr(selector.os, "replace", flaky_replace)
    selector.atomic_write_text(
        destination, '{"complete": true}', sleep_fn=sleeps.append)

    assert len(attempts) == 2
    assert sleeps == [selector.IO_RETRY_DELAY_SECONDS]
    assert json.loads(destination.read_text(encoding="utf-8")) == {"complete": True}
    assert not destination.with_name(f".{destination.name}.tmp").exists()


def test_final_checkpoint_is_complete_after_repeated_atomic_writes(tmp_path):
    checkpoint = tmp_path / "selection_ids.partial.json"
    expected = {}

    for i in range(5):
        expected[f"q{i:03d}"] = {"qid": f"q{i:03d}", "claim_rank_ids": [f"c{i}"]}
        selector.atomic_write_text(checkpoint, json.dumps(expected))

    assert json.loads(checkpoint.read_text(encoding="utf-8")) == expected


def test_checkpoint_unlink_retries_one_transient_failure(monkeypatch, tmp_path):
    checkpoint = tmp_path / "selection_ids.partial.json"
    checkpoint.write_text("{}", encoding="utf-8")
    real_unlink = Path.unlink
    attempts = []

    def flaky_unlink(path, *args, **kwargs):
        attempts.append(Path(path))
        if len(attempts) == 1:
            raise OSError(22, "transient Windows lock")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    selector.unlink_with_retry(checkpoint, sleep_fn=lambda _seconds: None)

    assert len(attempts) == 2
    assert not checkpoint.exists()
