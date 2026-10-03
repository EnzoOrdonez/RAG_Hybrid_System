import io
import json
from urllib.error import HTTPError

import pytest

from scripts import cloud_entrypoint as entry, cloud_storage as storage
from src.ui.components.session_storage import SessionStorageError, atomic_json
from src.ui.components.study_protocol import digest
from src.ui.components.study_sessions import StudyStore
from tests.study_helpers import configured


def test_generation_bound_copy_is_verified_and_conflict_never_overwritten(monkeypatch):
    bucket = storage.Bucket("test-private-bucket")
    calls = []

    def request(url, *, data=None, method="GET"):
        calls.append((url, method))
        if method == "POST":
            assert "ifGenerationMatch=0" in url
            raise HTTPError(url, 412, "exists", {}, None)
        if "alt=media" in url:
            assert "generation=123" in url
            return io.BytesIO(b"different")
        return io.BytesIO(b'{"generation":"123"}')

    monkeypatch.setattr(bucket, "request", request)
    with pytest.raises(ValueError, match="differs"):
        bucket.put_verified("sessions/id/full_session.json", b"original")
    assert [method for _, method in calls] == ["POST", "GET", "GET"]


def test_successful_copy_receipt_records_real_generation(monkeypatch):
    bucket = storage.Bucket("test-private-bucket")

    def request(url, *, data=None, method="GET"):
        return io.BytesIO(b'{"generation":"456"}' if method == "POST" else b"content")

    monkeypatch.setattr(bucket, "request", request)
    result = bucket.put_verified("object", b"content")
    assert result["generation"] == "456"
    assert len(result["sha256"]) == 64


def test_backup_failure_stays_pending_and_retry_is_verified(tmp_path):
    atomic_json(tmp_path / "full_session.json", {"stage": "complete"})
    atomic_json(
        tmp_path / "export_manifest.json",
        {"files": {"full_session.json": digest(tmp_path / "full_session.json")}},
    )

    class Bucket:
        name = "test-private-bucket"
        fail = True

        def put_verified(self, name, data):
            if self.fail:
                raise OSError("network unavailable")
            return {"object": name, "generation": "789"}

    bucket = Bucket()
    with pytest.raises(OSError):
        storage.backup_session(tmp_path, bucket, "study")
    assert (
        json.loads((tmp_path / "backup_state.json").read_text())["status"] == "pending"
    )
    bucket.fail = False
    result = storage.backup_session(tmp_path, bucket, "study")
    assert result["status"] == "complete"
    assert storage.backup_session(tmp_path, bucket, "study") == result


def test_missing_cloud_backup_blocks_a_preissued_invitation(tmp_path, monkeypatch):
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / "sessions", protocol)
    store.freeze()
    token = store.issue("P01")
    atomic_json(store.root / ("a" * 32) / "full_session.json", {"stage": "complete"})
    monkeypatch.setenv("CLOUDRAG_BACKUP_BUCKET", "test-private-bucket")
    with pytest.raises(SessionStorageError, match="missing"):
        store.admit(token)


def test_unsealed_existing_study_checkpoint_is_never_reused(tmp_path):
    _, _, protocol = configured(tmp_path)
    store = StudyStore(tmp_path / "sessions", protocol)
    atomic_json(
        store.root / ("a" * 32) / "study_checkpoint.json", {"stage": "complete"}
    )
    with pytest.raises(ValueError, match="historical"):
        store.freeze()
    assert not store.seal.exists()


@pytest.mark.parametrize(
    "fault", ["build", "dirty", "artifact", "seal", "model", "ollama"]
)
def test_startup_rejects_changed_deployment_identity(monkeypatch, fault):
    deployment = dict(
        build_id="expected",
        artifact_manifest="manifest",
        artifact_manifest_sha256="trusted",
        config_dir="config",
        fingerprint="seal",
        model_digest="weights",
        ollama_version="0.22.1",
    )
    monkeypatch.setattr(
        entry.subprocess,
        "check_output",
        lambda args, **kw: (
            ("wrong" if fault == "build" else "expected")
            if "rev-parse" in args
            else (b"M source" if fault == "dirty" else b"")
        ),
    )
    monkeypatch.setattr(
        entry, "verify_manifest", lambda *_: "bad" if fault == "artifact" else "trusted"
    )
    monkeypatch.setattr(
        entry,
        "verify_draw",
        lambda _: dict(
            fingerprint="bad" if fault == "seal" else "seal",
            config={"task_evidence_sha256": "evidence"},
        ),
    )

    def urlopen(url, **kw):
        result = (
            {
                "models": [
                    {
                        "name": "granite4.1:8b",
                        "digest": "bad" if fault == "model" else "weights",
                    }
                ]
            }
            if url.endswith("tags")
            else {"version": "bad" if fault == "ollama" else "0.22.1"}
        )
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(entry.urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError):
        entry.verify(deployment)
