"""Generate and verify an external, sealed inventory against the live runtime."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.components.session_storage import atomic_json, read_json  # noqa: E402
from src.ui.components.study_protocol import ROOT, digest  # noqa: E402


def _external(path):
    path = Path(path).resolve()
    if path.is_relative_to(ROOT.parent.parent):
        raise ValueError("Environment inventory must be outside the checkout")
    return path


def _api(endpoint):
    with urllib.request.urlopen(
        os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434") + endpoint,
        timeout=10,
    ) as response:
        return json.load(response)


def snapshot(config):
    from scripts.study_gate_environment import collect_identity, command
    from src.pipeline.pipeline_config import SURVEY_DEPLOY
    from src.ui.components.study_pipeline import STUDY_NO_RAG

    observed = collect_identity(config)
    models = [m for m in _api("/api/tags")["models"] if m["name"] == "granite4.1:8b"]
    if len(models) != 1 or models[0]["digest"] != config["model_digest"]:
        raise ValueError("Live Ollama digest differs")
    ollama = _api("/api/version")["version"]
    if ollama != config["ollama_version"]:
        raise ValueError("Live Ollama version differs")
    if not re.fullmatch(r"[a-f0-9]{40}", observed["build"]):
        raise ValueError("Full source commit required")
    if not re.fullmatch(r"[a-f0-9]{64}", models[0]["digest"]):
        raise ValueError("Full Ollama digest required")
    image = {"kind": "local", "id": None}
    if config.get("cloud"):
        # Receipt is produced by the host from docker inspect, mounted read-only.
        # It is a host trust boundary, never an assertion inferred inside Docker.
        receipt = read_json(_external(config["runtime_image_receipt"]))
        image = dict(kind="docker", **receipt)
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image["image_id"]):
            raise ValueError("Full host-inspected image ID required")
        if image["container_image_id"] != image["image_id"]:
            raise ValueError("Running container image differs")
        if os.environ.get("CLOUDRAG_IMAGE_ID") != image["image_id"]:
            raise ValueError("Runtime image receipt differs from launch environment")
    artifacts = read_json(config["artifact_manifest"])
    vendor = read_json(config["vendor_manifest"]) if config.get("cloud") else None
    recipes = dict(hybrid=SURVEY_DEPLOY.model_dump(), no_rag=STUDY_NO_RAG.model_dump())
    packages = sorted((d.metadata["Name"], d.version) for d in distributions())
    return dict(
        schema_version=1,
        observed=observed,
        source=dict(
            commit=observed["build"],
            tree=command(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"]),
        ),
        protocol=dict(
            fingerprint=observed["fingerprint"], config_dir=config["config_dir"]
        ),
        recipes=recipes,
        dependencies=packages,
        locks={
            name: digest(ROOT / name)
            for name in ("requirements-lock.txt", "requirements-app.txt", "Dockerfile")
        },
        vendor=vendor,
        image=image,
        ollama=dict(
            version=ollama, model=models[0]["name"], digest=models[0]["digest"]
        ),
        artifacts=artifacts,
        locations={
            key: config[key]
            for key in (
                "artifact_manifest",
                "config_dir",
                "session_root",
                "preregistration",
            )
        },
    )


def generate(config, output):
    output = _external(output)
    if output.exists():
        raise FileExistsError("Never overwrite an environment identity")
    inventory = snapshot(config)
    inventory["generated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(output, inventory)
    return dict(
        config,
        environment_identity=str(output),
        environment_identity_sha256=digest(output),
    )


def load(config):
    path = _external(config["environment_identity"])
    if digest(path) != config["environment_identity_sha256"]:
        raise ValueError("Environment inventory seal changed")
    inventory = read_json(path)
    if inventory.get("schema_version") != 1:
        raise ValueError("Unsupported environment identity")
    return inventory


def verify(config):
    inventory = load(config)
    actual = snapshot(config)
    expected = {k: v for k, v in inventory.items() if k != "generated_utc"}
    # JSON serialization normalizes tuples in package inventories.
    if json.loads(json.dumps(actual)) != expected:
        raise ValueError("Live runtime differs from generated environment identity")
    return dict(
        inventory["observed"],
        environment_identity_sha256=config["environment_identity_sha256"],
    )


def report(config):
    """Report identity from its sealed inventory, never hand-entered evidence hashes."""
    return dict(inventory=load(config), sha256=config["environment_identity_sha256"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("generate", "verify", "report"))
    parser.add_argument("--settings", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_json(args.settings)
    if args.operation == "generate":
        effective = generate(config, args.output)
        destination = Path(args.output).with_name("settings-effective.json")
        if destination.exists():
            raise FileExistsError("Never replace effective settings")
        atomic_json(destination, effective)
        print(destination)
    else:
        result = verify(config) if args.operation == "verify" else report(config)
        output = _external(args.output)
        if output.exists():
            raise FileExistsError("Never overwrite identity evidence")
        atomic_json(output, result)
        print(output)


if __name__ == "__main__":
    main()
