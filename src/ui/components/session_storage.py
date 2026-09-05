"""Durable single-instance interview storage. Never stores invitation plaintext."""

import hashlib
import json
import os
import re
import secrets
import tempfile
import uuid
from pathlib import Path

from filelock import FileLock


class SessionStorageError(RuntimeError):
    """Persisted state cannot be read safely; operator recovery is required."""


class SessionConflict(SessionStorageError):
    """Another tab or an active interview owns this state."""


def validate_participant(value):
    if not isinstance(value, str) or not re.fullmatch(r"P[0-9]{2,6}", value):
        raise ValueError("Participant ID must be P followed by 2–6 digits")
    return value


def session_path(root, session_id):
    if not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
        raise ValueError("Invalid session identifier")
    root = Path(root).resolve()
    path = (root / session_id).resolve()
    if path.parent != root:
        raise ValueError("Session path escapes storage root")
    return path


def atomic_text(path, text):
    """Flush a sibling temporary file, then atomically replace; preserve old on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected a stored object")
        return value
    except (OSError, ValueError) as exc:
        raise SessionStorageError("Stored session is unreadable; contact the coordinator") from exc


class InvitationStore:
    """One active interview across browser sessions, tabs and worker processes.

    Admissions are persistent (no timeout takeover). Operator abandonment is explicit.
    Generation additionally holds a file lock for its full lifetime.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "_admissions.json"
        self.lock = FileLock(str(self.root / "_admissions.lock"), timeout=5)

    def _read(self):
        return read_json(self.path) if self.path.exists() else {"invitations": {}, "active": None}

    def issue(self, participant_id):
        validate_participant(participant_id)
        with self.lock:
            data = self._read()
            if (self.root / participant_id).exists():
                raise ValueError("Legacy participant data exists; use a new participant identifier")
            if any(i["participant_id"] == participant_id for i in data["invitations"].values()):
                raise ValueError("Participant already has an invitation; do not duplicate observations")
            token = secrets.token_urlsafe(32)
            data["invitations"][hashlib.sha256(token.encode()).hexdigest()] = {
                "participant_id": participant_id, "session_id": uuid.uuid4().hex}
            atomic_json(self.path, data)
            return token

    def admit(self, token, experience):
        from src.ui.components.session_manager import EvaluationSession
        with self.lock:
            data = self._read()
            invite = data["invitations"].get(hashlib.sha256(token.encode()).hexdigest())
            if invite is None or invite.get("revoked"):
                raise ValueError("Invitación inválida. Contacta al coordinador.")
            sid = invite["session_id"]
            session = EvaluationSession.load_checkpoint(sid)
            if session and session.state == "complete":
                return session
            active = data.get("active")
            if active and active != sid:
                previous = EvaluationSession.load_checkpoint(active)
                if previous is None or previous.state != "complete":
                    raise SessionConflict("Hay otra entrevista activa. Contacta al coordinador.")
            if session is None:
                session = EvaluationSession(invite["participant_id"], experience, session_id=sid)
                session.save_checkpoint()
            data["active"] = sid
            atomic_json(self.path, data)
            return session

    def assert_active(self, session_id):
        with self.lock:
            data = self._read()
            if data.get("active") != session_id:
                raise SessionConflict("La sesión no está activa. Contacta al coordinador.")

    def abandon(self, session_id):
        """Operator-only release; revoke invitation without deleting interview records."""
        with FileLock(str(self.root / "_inference.lock"), timeout=0), self.lock:
            data = self._read()
            if data.get("active") != session_id:
                raise SessionConflict("Session is not the active interview")
            for invite in data["invitations"].values():
                if invite["session_id"] == session_id:
                    invite["revoked"] = True
            data["active"] = None
            atomic_json(self.path, data)
