"""Verify the deployed identities before any app or gate operation."""

import argparse
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ui.components.session_storage import atomic_json  # noqa: E402
from src.ui.components.study_protocol import ROOT, verify_draw  # noqa: E402
from src.utils.deployment_artifacts import verify_manifest  # noqa: E402


def configure(deployment):
    if deployment.get("environment_identity"):
        from scripts.environment_identity import load

        inventory = load(deployment)
        deployment = dict(
            deployment,
            build_id=inventory["source"]["commit"],
            model_digest=inventory["ollama"]["digest"],
        )
    config_dir = Path(deployment["config_dir"])
    os.environ.update(
        CLOUDRAG_BUILD_ID=deployment["build_id"],
        CLOUDRAG_MODEL_DIGEST=deployment["model_digest"],
        CLOUDRAG_ARTIFACT_MANIFEST=deployment["artifact_manifest"],
        CLOUDRAG_STUDY_CONFIG=str(config_dir / "study.json"),
        CLOUDRAG_STUDY_ASSIGNMENTS=str(config_dir / "assignments.csv"),
        CLOUDRAG_STUDY_SESSION_DIR=deployment["session_root"],
        CLOUDRAG_STUDY_PURPOSE=deployment.get("purpose", "study"),
        CLOUDRAG_MODE="participant",
        CLOUDRAG_DEMO_GPU=deployment.get("device_gpu", "0"),
        CLOUDRAG_BACKUP_BUCKET=deployment["bucket"],
        CLOUDRAG_BACKUP_PREFIX=deployment["backup_prefix"],
    )
    if deployment.get('service_mode'):
        os.environ['CLOUDRAG_ISOLATED_APP'] = '1'
        os.environ['CLOUDRAG_ISOLATED_SERVICE'] = '1'
        os.environ['CLOUDRAG_BACKUP_SOCKET'] = deployment['backup_socket']


def freeze_settings(deployment, path):
    from src.pipeline.pipeline_config import SURVEY_DEPLOY
    from src.ui.components.study_pipeline import STUDY_NO_RAG
    from scripts.environment_identity import generate

    path = Path(path)
    if path.exists():
        raise FileExistsError("Never replace deployment admission settings")
    recipe = hashlib.sha256(
        json.dumps(
            dict(hybrid=SURVEY_DEPLOY.model_dump(), no_rag=STUDY_NO_RAG.model_dump()),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    packages = sorted((d.metadata["Name"], d.version) for d in distributions())
    settings = dict(
        deployment,
        recipe_sha256=recipe,
        packages_sha256=hashlib.sha256(json.dumps(packages).encode()).hexdigest(),
        cloud=True,
        preregistration=deployment.get('preregistration', str(
            ROOT / "docs/STUDY_GATE_PREREGISTRATION_AMENDMENT_2026-10-02.md"
        )),
    )
    settings = generate(settings, path.with_name("environment_identity.json"))
    atomic_json(path, settings)
    atomic_json(path.with_name(path.stem + "-packages.json"), dict(packages=packages))
    return settings


def verify(deployment, *, generating=False):
    build = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=15
    ).strip()
    if (
        build != deployment["build_id"]
        or subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], timeout=15
        ).strip()
    ):
        raise ValueError("Source commit or tracked checkout changed")
    if (
        verify_manifest(ROOT, deployment["artifact_manifest"])
        != deployment["artifact_manifest_sha256"]
    ):
        raise ValueError("Artifact trust anchor changed")
    protocol = verify_draw(deployment["config_dir"])
    if protocol["fingerprint"] != deployment["fingerprint"] or not protocol[
        "config"
    ].get("task_evidence_sha256"):
        raise ValueError("Reviewed study configuration required")
    with urllib.request.urlopen(
        "http://127.0.0.1:11434/api/tags", timeout=10
    ) as response:
        models = json.load(response)["models"]
    if not any(
        m["name"] == "granite4.1:8b" and m["digest"] == deployment["model_digest"]
        for m in models
    ):
        raise ValueError("Exact Ollama model unavailable")
    with urllib.request.urlopen(
        "http://127.0.0.1:11434/api/version", timeout=10
    ) as response:
        if json.load(response)["version"] != deployment["ollama_version"]:
            raise ValueError("Ollama version differs")
    if not generating:
        from scripts.environment_identity import verify as verify_identity

        verify_identity(deployment)
    return protocol


def streamlit_command(deployment):
    return [sys.executable, '-m', 'streamlit', 'run', 'src/ui/app.py',
            '--server.address', '127.0.0.1', '--server.port', '8501',
            '--server.headless', 'true', '--server.fileWatcherType', 'none',
            '--browser.gatherUsageStats', 'false',
            *(['--browser.serverAddress', deployment['hostname'], '--browser.serverPort', '443']
              if deployment.get('hostname') else [])]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("verify", "freeze", "serve", "test", "gate", "backup", "invite"),
    )
    parser.add_argument("--deployment", default="/srv/cloudrag/deployment.json")
    parser.add_argument("--request")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    deployment = json.loads(Path(args.deployment).read_text())
    configure(deployment)
    protocol = verify(deployment, generating=args.operation == "freeze")
    from src.ui.components.study_sessions import StudyStore

    store = StudyStore(
        deployment["session_root"], protocol, deployment.get("purpose", "study")
    )
    store.freeze()
    if args.operation == "freeze":
        print(json.dumps(freeze_settings(deployment, args.output)))
    elif args.operation == "serve":
        os.execv(sys.executable, streamlit_command(deployment))
    elif args.operation == "test":
        os.execv(sys.executable, [sys.executable, "-m", "pytest", "-ra"])
    elif args.operation == "gate":
        from scripts.run_study_gate import main as gate

        gate(["--settings", args.request, "--output", args.output])
    elif args.operation == "backup":
        from scripts.cloud_storage import Bucket, backup_session

        receipts = []
        for export in sorted(
            Path(deployment["session_root"]).glob("*/full_session.json")
        ):
            receipts.append(
                backup_session(
                    export.parent,
                    Bucket(deployment["bucket"]),
                    deployment["backup_prefix"],
                )
            )
        print(json.dumps(receipts))
    elif args.operation == "invite":
        request = json.load(sys.stdin) if args.request == '-' else json.loads(Path(args.request).read_text())
        store.register_invitation_hash(
            request["token_sha256"], request["participant_id"],
            cell=request.get("cell"), profile=request.get("profile"),
        )
        print(json.dumps(dict(status="invitation_hash_registered")))
    else:
        print(
            json.dumps(
                dict(
                    status="verified",
                    build=deployment["build_id"],
                    fingerprint=protocol["fingerprint"],
                )
            )
        )


if __name__ == "__main__":
    main()
