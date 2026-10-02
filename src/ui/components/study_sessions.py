"""Versioned two-block study sessions, separate from historical evaluation sessions."""

import copy

from src.evaluation.decline_classifier import CLASSIFIER_VERSION, classify_response
import hashlib
import os
from pathlib import Path
import time
import uuid

from filelock import FileLock

from src.ui.components.session_storage import (
    InvitationStore,
    SessionConflict,
    SessionStorageError,
    atomic_json,
    read_json,
    session_path,
)
from src.ui.components.study_protocol import (
    CELLS,
    LIKERT_IDS,
    PROFILES,
    ROOT,
    digest,
    scores,
    sus_score,
)

STAGES = {
    "familiarization",
    "tasks",
    "free_query",
    "instruments",
    "comparative",
    "blinding",
    "complete",
    "abandoned",
}


class StudyStore(InvitationStore):
    def __init__(self, root, protocol, purpose="study"):
        root = Path(root).resolve()
        if root.is_relative_to(ROOT.parent.parent) or purpose not in (
            "study",
            "pilot",
            "technical",
            "smoke",
            "rehearsal",
        ):
            raise ValueError(
                "Use a separate external study directory and explicit purpose"
            )
        super().__init__(root)
        self.protocol, self.purpose = protocol, purpose
        self.seal = root / "_study_protocol.json"

    def freeze(self):
        expected = dict(
            schema_version=3,
            fingerprint=self.protocol["fingerprint"],
            hashes=self.protocol["hashes"],
            purpose=self.purpose,
        )
        with self.lock:
            if self.seal.exists():
                if read_json(self.seal) != expected:
                    raise ValueError("Frozen study identity changed")
            else:
                if self.path.exists() or any(
                    self.root.glob("*/session_checkpoint.json")
                ):
                    raise ValueError("Do not reuse historical session storage")
                atomic_json(self.seal, expected)

    def check(self):
        if not self.seal.exists() or read_json(self.seal) != dict(
            schema_version=3,
            fingerprint=self.protocol["fingerprint"],
            hashes=self.protocol["hashes"],
            purpose=self.purpose,
        ):
            raise SessionStorageError(
                "Freeze valid study configuration before admitting participants"
            )
        for name, path in self.protocol.get("paths", {}).items():
            if digest(path) != self.protocol["hashes"][name]:
                raise SessionStorageError("Frozen configuration file changed")

    def assignment(self, pid, *, cell=None, profile=None):
        if self.purpose != "study":
            required = {"smoke": "P999", "rehearsal": "P998"}.get(self.purpose)
            if (
                not pid.startswith("P")
                or not pid[1:].isdigit()
                or (required and pid != required)
                or (not required and int(pid[1:]) < 900)
                or cell not in CELLS
                or profile not in PROFILES
            ):
                raise ValueError(
                    "Pilots/technical sessions require fictitious P900+, cell and profile"
                )
            return dict(
                participant_id=pid,
                role="pilot",
                cell=cell,
                profile=profile,
                primary_slot=pid,
            )
        row = self.protocol["assignments"].get(pid)
        if row is None or cell is not None or profile is not None:
            raise ValueError("Participant must have a frozen assignment")
        replacements = (
            read_json(self.root / "_replacements.json")
            if (self.root / "_replacements.json").exists()
            else {}
        )
        if row["role"] == "reserve" and pid not in replacements:
            raise ValueError("Activate reserve explicitly for a matching primary slot")
        if pid in {r["primary"] for r in replacements.values()}:
            raise ValueError("Primary slot already replaced")
        return dict(row, primary_slot=replacements.get(pid, {}).get("primary", pid))

    def check_backups(self):
        """Fail closed before issuing or admitting, including preissued invitations."""
        states = list(self.root.glob("*/backup_state.json"))
        if any(read_json(path).get("status") != "complete" for path in states):
            raise SessionStorageError(
                "Mandatory backup is pending; resolve it before another session"
            )

    def issue(self, participant_id, *, cell=None, profile=None):
        self.check()
        assignment = self.assignment(participant_id, cell=cell, profile=profile)
        with self.lock:
            self.check_backups()
            token = super().issue(participant_id)
            data = self._read()
            data["invitations"][hashlib.sha256(token.encode()).hexdigest()][
                "assignment"
            ] = assignment
            atomic_json(self.path, data)
        return token

    def admit(self, token):
        self.check()
        with self.lock:
            self.check_backups()
            data = self._read()
            invite = data["invitations"].get(hashlib.sha256(token.encode()).hexdigest())
            if not invite or invite.get("revoked"):
                raise ValueError("Invitación inválida. Contacta al coordinador.")
            sid = invite["session_id"]
            session = StudySession.load(self, sid)
            if session and session.data["stage"] in ("complete", "abandoned"):
                return session
            if data.get("active") not in (None, sid):
                previous = StudySession.load(self, data["active"])
                if previous is None or previous.data["stage"] not in (
                    "complete",
                    "abandoned",
                ):
                    raise SessionConflict("Hay otra sesión activa.")
            if session is None:
                session = StudySession(self, sid, invite["assignment"])
                session.save()
            data["active"] = sid
            atomic_json(self.path, data)
            return session

    def abandon(self, sid):
        with FileLock(str(self.root / "_inference.lock"), timeout=0), self.lock:
            data = self._read()
            if data.get("active") != sid:
                raise SessionConflict("Not the active session")
            session = StudySession.load(self, sid)
            if session.pending and session.pending["status"] == "running":
                session.finish(error="interrupted", elapsed_ms=None)
            session.data["stage"] = "abandoned"
            session.data["events"].append(
                dict(kind="session_abandoned", timestamp=time.time())
            )
            session.save()
            for invite in data["invitations"].values():
                if invite["session_id"] == sid:
                    invite["revoked"] = True
            data["active"] = None
            atomic_json(self.path, data)
            session.export()

    def replace(self, primary, reserve):
        self.check()
        a, b = (self.protocol["assignments"].get(p, {}) for p in (primary, reserve))
        if (
            self.purpose != "study"
            or a.get("role") != "primary"
            or b.get("role") != "reserve"
            or any(a[k] != b[k] for k in ("cell", "profile"))
        ):
            raise ValueError("Replacement must preserve primary cell and profile")
        with self.lock:
            admissions = self._read()
            invitations = admissions["invitations"]
            for invite in invitations.values():
                if invite["participant_id"] == primary:
                    session = StudySession.load(self, invite["session_id"])
                    if session and session.data["stage"] != "abandoned":
                        raise ValueError("Abandon primary session before replacement")
            path = self.root / "_replacements.json"
            rows = read_json(path) if path.exists() else {}
            if reserve in rows or primary in {r["primary"] for r in rows.values()}:
                raise ValueError("Replacement already recorded")
            rows[reserve] = dict(primary=primary, timestamp=time.time())
            for invite in invitations.values():
                if invite["participant_id"] == primary:
                    invite["revoked"] = True
            # Revoke first: an interruption must never enable both observations.
            atomic_json(self.path, admissions)
            atomic_json(path, rows)


class StudySession:
    def __init__(self, store, sid, assignment):
        self.store, self.session_id = store, sid
        self.path = session_path(store.root, sid) / "study_checkpoint.json"
        self.data = dict(
            schema_version=3,
            session_id=sid,
            revision=0,
            assignment=assignment,
            purpose=store.purpose,
            protocol_fingerprint=store.protocol["fingerprint"],
            build_id=os.environ.get("CLOUDRAG_BUILD_ID", "unverified"),
            created_at=time.time(),
            stage="familiarization",
            block_index=0,
            task_index=0,
            events=[],
            incidents=[],
            attempts=[],
            instruments=[],
            comparative=None,
            blinding=None,
        )

    @property
    def block(self):
        label, task_set = CELLS[self.data["assignment"]["cell"]][
            self.data["block_index"]
        ]
        return dict(
            label=label,
            task_set=task_set,
            condition=self.store.protocol["config"]["labels"][label],
        )

    @property
    def pending(self):
        role = self.data["stage"]
        return next(
            (
                a
                for a in reversed(self.data["attempts"])
                if a["block_index"] == self.data["block_index"]
                and a["analysis_role"] == role
                and (role != "tasks" or a["task_index"] == self.data["task_index"])
            ),
            None,
        )

    def require(self, stage):
        if self.data["stage"] != stage:
            raise ValueError("Invalid study transition")

    def save(self):
        self.store.check()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + ".lock", timeout=5):
            previous = read_json(self.path) if self.path.exists() else None
            old = previous["revision"] if previous is not None else 0
            if old != self.data["revision"]:
                raise SessionConflict("La sesión cambió en otra pestaña. Recarga.")
            payload = copy.deepcopy(self.data)
            payload["revision"] += 1
            try:
                atomic_json(self.path, payload)
            except (OSError, SessionStorageError):
                # The durable revision remains authoritative when publication fails.
                # Never let a failed form submission or unflushed answer advance it.
                if previous is not None:
                    self.data = previous
                raise
            self.data = payload

    @classmethod
    def load(cls, store, sid):
        store.check()
        path = session_path(store.root, sid) / "study_checkpoint.json"
        if not path.exists():
            return None
        data = read_json(path)
        try:
            obj = cls(store, sid, data["assignment"])
            if (
                set(data) != set(obj.data)
                or data["schema_version"] != 3
                or data["session_id"] != sid
                or data["purpose"] != store.purpose
                or data["protocol_fingerprint"] != store.protocol["fingerprint"]
            ):
                raise ValueError("Schema or protocol mismatch")
            if (
                data["stage"] not in STAGES
                or type(data["block_index"]) is not int
                or data["block_index"] not in (0, 1)
                or type(data["task_index"]) is not int
                or data["task_index"] not in range(3)
            ):
                raise ValueError("Invalid progress")
            if type(data["revision"]) is not int or data["revision"] < 1:
                raise ValueError("Invalid revision")
            invite = next(
                i
                for i in store._read()["invitations"].values()
                if i["session_id"] == sid
            )
            if data["assignment"] != invite["assignment"]:
                raise ValueError("Assignment changed")
            for event in data["events"]:
                expected = (
                    {"kind", "timestamp", "block_index"}
                    if event["kind"] == "familiarization_done"
                    else {"kind", "timestamp"}
                )
                if set(event) != expected or event["kind"] not in (
                    "familiarization_done",
                    "session_abandoned",
                ):
                    raise ValueError("Unexpected event payload")
            for incident in data["incidents"]:
                if (
                    set(incident) != {"kind", "timestamp"}
                    or incident["kind"] != "session_technical_block"
                ):
                    raise ValueError("Incident must have no practice content")
            if len({a["attempt_id"] for a in data["attempts"]}) != len(
                data["attempts"]
            ):
                raise ValueError("Duplicate attempts")
            if any(
                a["analysis_role"] not in ("tasks", "free_query")
                or a["block_index"] not in (0, 1)
                for a in data["attempts"]
            ):
                raise ValueError("Practice must never be persisted")
            obj.data = data
            return obj
        except (ValueError, KeyError, TypeError, StopIteration) as exc:
            raise SessionStorageError(
                "Stored study state is invalid; preserve it for review"
            ) from exc

    def familiarization_done(self):
        self.require("familiarization")
        self.data["events"].append(
            dict(
                kind="familiarization_done",
                block_index=self.data["block_index"],
                timestamp=time.time(),
            )
        )
        self.data["stage"] = "tasks"
        self.save()

    def incident(self):
        self.data["incidents"].append(
            dict(kind="session_technical_block", timestamp=time.time())
        )
        self.save()

    def begin(self, free_question=None):
        if self.data["stage"] not in ("tasks", "free_query") or (
            self.pending and self.pending["status"] != "error"
        ):
            raise SessionConflict("No new query allowed")
        block = self.block
        if self.data["stage"] == "tasks":
            qid = self.store.protocol["config"]["tasks"][block["task_set"]][
                self.data["task_index"]
            ]
            question = self.store.protocol["queries"][qid]["question"]
        else:
            qid, question = None, free_question
            if (
                not isinstance(question, str)
                or not question.strip()
                or len(question) > 6000
            ):
                raise ValueError("Escribe una consulta de hasta 6000 caracteres.")
        self.data["attempts"].append(
            dict(
                attempt_id=uuid.uuid4().hex,
                analysis_role=self.data["stage"],
                block_index=self.data["block_index"],
                task_index=self.data["task_index"],
                **block,
                query_id=qid,
                question=question,
                status="running",
                started_at=time.time(),
                answer=None,
                sources=[],
                error=None,
                elapsed_ms=None,
                shown_at=None,
                finished_at=None,
                decline_class=None,
                decline_classifier_version=None,
            )
        )
        self.save()

    def finish(self, *, answer=None, sources=None, error=None, elapsed_ms=None):
        if not self.pending or self.pending["status"] != "running":
            raise SessionConflict("No running attempt")
        if not error and (not isinstance(answer, str) or not answer.strip()):
            error = "empty_response"
        self.pending.update(
            status="error" if error else "success",
            error=error,
            answer=None if error else answer,
            sources=[] if error else (sources or []),
            elapsed_ms=elapsed_ms,
            finished_at=time.time(),
            decline_class=None if error else classify_response(answer),
            decline_classifier_version=None if error else CLASSIFIER_VERSION,
        )
        self.save()

    def shown(self):
        if not self.pending or self.pending["status"] != "success":
            raise ValueError("No successful response")
        if self.pending["shown_at"] is None:
            self.pending["shown_at"] = time.time()
            self.save()

    def acknowledge(self):
        if (
            not self.pending
            or self.pending["status"] != "success"
            or not self.pending["shown_at"]
        ):
            raise ValueError("Read a successful response before continuing")
        self.pending["status"] = "acknowledged"
        if self.data["stage"] == "free_query":
            self.data["stage"] = "instruments"
        elif self.data["task_index"] == 2:
            self.data["stage"] = "free_query"
        else:
            self.data["task_index"] += 1
        self.save()

    def submit_instruments(self, sus, likert):
        self.require("instruments")
        if set(likert) != set(LIKERT_IDS):
            raise ValueError("Complete all block items")
        scores(list(likert.values()), 10)
        self.data["instruments"].append(
            dict(
                block_index=self.data["block_index"],
                **self.block,
                sus=list(sus),
                sus_score=sus_score(sus),
                likert=dict(likert),
                timestamp=time.time(),
            )
        )
        if self.data["block_index"] == 0:
            self.data.update(block_index=1, task_index=0, stage="familiarization")
        else:
            self.data["stage"] = "comparative"
        self.save()

    def submit_comparative(self, values):
        self.require("comparative")
        if (
            set(values) != {"C1", "C2", "C3", "C4"}
            or any(values[k] not in ("A", "B", "iguales") for k in ("C1", "C2"))
            or values["C3"] not in ("A", "B", "ninguno")
            or not isinstance(values["C4"], str)
        ):
            raise ValueError("Complete comparative items")
        self.data.update(comparative=values, stage="blinding")
        self.save()

    def submit_blinding(self, choice, reason):
        self.require("blinding")
        if choice not in self.store.protocol["config"][
            "blinding_choices"
        ] or not isinstance(reason, str):
            raise ValueError("Complete blinding item")
        self.data.update(
            blinding=dict(choice=choice, reason=reason, timestamp=time.time()),
            stage="complete",
        )
        self.save()
        self.export()

    def export(self):
        if self.data["stage"] not in ("complete", "abandoned"):
            raise ValueError("Only closed sessions may be exported")
        payload = copy.deepcopy(self.data)
        payload["analysis_excluded"] = self.store.purpose in (
            "smoke",
            "rehearsal",
            "technical",
        )
        if self.store.purpose == "smoke":
            payload["gate_marker"] = "SMOKE_NOT_GATE"
            payload["instrument_responses_synthetic"] = True
        payload["protocol_hashes"] = self.store.protocol["hashes"]
        payload["labels"] = self.store.protocol["config"]["labels"]
        target = self.path.parent / "full_session.json"
        if target.exists() and read_json(target) != payload:
            raise SessionConflict("Export already exists with different contents")
        if not target.exists():
            atomic_json(target, payload)
        manifest = self.path.parent / "export_manifest.json"
        content = dict(schema_version=1, files={"full_session.json": digest(target)})
        if manifest.exists() and read_json(manifest) != content:
            raise SessionStorageError("Export integrity mismatch")
        if not manifest.exists():
            atomic_json(manifest, content)
        return target
