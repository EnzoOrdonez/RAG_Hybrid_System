"""Verified, retryable export backup.  A failed copy never changes the source export."""

import shutil
import uuid
from pathlib import Path

from src.ui.components.session_storage import atomic_json, read_json
from src.ui.components.study_protocol import digest


def backup_export(session_dir, destination, *, same_physical_disk=None):
    session_dir, destination = Path(session_dir).resolve(), Path(destination).resolve()
    same_physical_disk = same_physical_disk or (
        lambda left, right: left.drive == right.drive
    )
    if same_physical_disk(session_dir, destination):
        raise ValueError("Backup destination must be on another physical disk")
    source = session_dir / "full_session.json"
    manifest = session_dir / "export_manifest.json"
    if (
        not source.exists()
        or not manifest.exists()
        or read_json(manifest)["files"].get(source.name) != digest(source)
    ):
        raise ValueError("Export is missing or its hash is invalid")
    target = destination / session_dir.name
    temporary = destination / (".pending-" + uuid.uuid4().hex)
    try:
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(session_dir, temporary)
        copied = temporary / source.name
        if digest(copied) != digest(source):
            raise OSError("Backup hash mismatch")
        if target.exists():
            if digest(target / source.name) != digest(source):
                raise FileExistsError("Existing backup differs; preserve for review")
            shutil.rmtree(temporary)
        else:
            temporary.replace(target)
        state = dict(
            status="complete",
            source=str(session_dir),
            destination=str(target),
            sha256=digest(source),
        )
    except Exception as exc:
        state = dict(
            status="pending",
            source=str(session_dir),
            destination=str(destination),
            error=type(exc).__name__,
        )
        atomic_json(session_dir / "backup_state.json", state)
        raise
    atomic_json(session_dir / "backup_state.json", state)
    return state
