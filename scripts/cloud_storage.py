"""Private GCS transfers using the VM identity; no credential files."""

import hashlib
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.parse
import urllib.request


class Bucket:
    def __init__(self, name):
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,220}[a-z0-9]", name):
            raise ValueError("Invalid private bucket name")
        self.name = name

    def request(self, url, *, data=None, method="GET"):
        if os.environ.get("CLOUDRAG_ISOLATED_SERVICE") == "1" or os.environ.get("CLOUDRAG_STUDY_PURPOSE") == "study":
            raise ValueError("The isolated application must use its host backup agent")
        metadata = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(metadata, timeout=5) as response:
            token = json.load(response)["access_token"]
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/octet-stream",
            },
        )
        return urllib.request.urlopen(request, timeout=120)

    def get(self, name, generation):
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            + self.name
            + "/o/"
            + urllib.parse.quote(name, safe="")
            + "?"
            + urllib.parse.urlencode(dict(alt="media", generation=generation))
        )
        with self.request(url) as response:
            return response.read()

    def put_verified(self, name, data):
        url = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            + self.name
            + "/o?"
            + urllib.parse.urlencode(
                dict(uploadType="media", name=name, ifGenerationMatch=0)
            )
        )
        try:
            with self.request(url, data=data, method="POST") as response:
                metadata = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 412:
                raise
            url = (
                "https://storage.googleapis.com/storage/v1/b/"
                + self.name
                + "/o/"
                + urllib.parse.quote(name, safe="")
            )
            with self.request(url) as response:
                metadata = json.load(response)
        copied = self.get(name, metadata["generation"])
        if hashlib.sha256(copied).digest() != hashlib.sha256(data).digest():
            raise ValueError("Private remote copy differs; preserve both")
        return dict(
            object=name,
            generation=metadata["generation"],
            sha256=hashlib.sha256(data).hexdigest(),
        )


def backup_session(session_dir, bucket, prefix):
    from src.ui.components.session_storage import atomic_json, read_json
    from src.ui.components.study_protocol import digest

    session_dir = Path(session_dir)
    manifest = read_json(session_dir / "export_manifest.json")
    if manifest["files"] != {
        "full_session.json": digest(session_dir / "full_session.json")
    }:
        raise ValueError("Export integrity mismatch")
    state_path = session_dir / "backup_state.json"
    if state_path.exists():
        previous = read_json(state_path)
        if previous.get("status") == "complete":
            if previous.get("sha256") != digest(session_dir / "full_session.json"):
                raise ValueError("Export changed after verified backup")
            return previous
    atomic_json(
        state_path,
        dict(status="pending", destination="gs://" + bucket.name + "/" + prefix),
    )
    if os.environ.get("CLOUDRAG_BACKUP_SOCKET"):
        from scripts.study_operator.agent_client import backup_request

        state = backup_request(os.environ["CLOUDRAG_BACKUP_SOCKET"], session_dir, bucket.name, prefix)
        atomic_json(state_path, state)
        return state
    receipts = {}
    for name in ("full_session.json", "export_manifest.json"):
        receipts[name] = bucket.put_verified(
            prefix + "/" + session_dir.name + "/" + name,
            (session_dir / name).read_bytes(),
        )
    state = dict(
        status="complete",
        source=str(session_dir),
        destination="gs://" + bucket.name + "/" + prefix + "/" + session_dir.name,
        sha256=digest(session_dir / "full_session.json"),
        objects=receipts,
    )
    atomic_json(state_path, state)
    return state
