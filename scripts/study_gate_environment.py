"""Read-only deployment identity and live Windows/Linux admission checks."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

from src.ui.components.session_storage import atomic_json, read_json
from src.ui.components.study_protocol import ROOT, digest, verify_draw
from src.utils.deployment_artifacts import verify_manifest


def command(args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def identity(config):
    protocol = verify_draw(config["config_dir"])
    build = command(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if build != config["build_id"] or command(
        ["git", "-C", str(ROOT), "status", "--porcelain"]
    ):
        raise ValueError("Build identity differs or worktree is dirty")
    artifact = verify_manifest(ROOT, config["artifact_manifest"])
    if artifact != config["artifact_manifest_sha256"]:
        raise ValueError("Trusted artifact manifest changed")
    if protocol["fingerprint"] != config["fingerprint"]:
        raise ValueError("Sealed protocol changed")
    session_root = Path(config["session_root"])
    if not session_root.is_dir():
        raise ValueError("Session root is missing")
    for state in session_root.glob("*/backup_state.json"):
        if read_json(state).get("status") != "complete":
            raise ValueError("Mandatory session backup pending")
    from src.pipeline.pipeline_config import SURVEY_DEPLOY
    from src.ui.components.study_pipeline import STUDY_NO_RAG

    recipes = {
        "hybrid": SURVEY_DEPLOY.model_dump(),
        "no_rag": STUDY_NO_RAG.model_dump(),
    }
    recipe_hash = hashlib.sha256(
        json.dumps(recipes, sort_keys=True).encode()
    ).hexdigest()
    if recipe_hash != config["recipe_sha256"]:
        raise ValueError("Pipeline recipe changed")
    packages = sorted((d.metadata["Name"], d.version) for d in distributions())
    software = hashlib.sha256(json.dumps(packages).encode()).hexdigest()
    if software != config["packages_sha256"]:
        raise ValueError("Installed dependency inventory changed")
    result = dict(
        build=build,
        fingerprint=protocol["fingerprint"],
        artifacts=artifact,
        model_digest=config["model_digest"],
        recipe=recipe_hash,
        packages=software,
        device=os.environ.get("CLOUDRAG_DEMO_GPU", "0"),
        platform=platform.platform(),
        preregistration=digest(config["preregistration"]),
        gpu=command(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    )
    if config.get("cloud"):
        for field in ("id", "zone", "machine-type"):
            request = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/" + field,
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(request, timeout=3) as response:
                result[field] = response.read().decode()
        if (
            result["zone"].split("/")[-1] != config["zone"]
            or result["machine-type"].split("/")[-1] != config["machine_type"]
        ):
            raise ValueError("Cloud hardware differs from preregistration")
    return result


class LinuxSampler:
    def __init__(self):
        self.previous = None
        self.processes = {}
        self.previous_at = None

    def __call__(self):
        now = time.monotonic()
        ticks = [
            int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
        ]
        total, idle = sum(ticks), ticks[3] + ticks[4]
        cpu = None
        if self.previous:
            delta = total - self.previous[0]
            cpu = 100 * (1 - (idle - self.previous[1]) / delta) if delta else 0
        rows = []
        current = {}
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                stat = (proc / "stat").read_text()
                end = stat.rfind(")")
                fields = stat[end + 2 :].split()
                used = (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
                key = (int(proc.name), fields[19])
                load = (
                    100
                    * (used - self.processes[key])
                    / (now - self.previous_at)
                    / os.cpu_count()
                    if key in self.processes
                    else 0
                )
                current[key] = used
                rows.append(
                    dict(
                        pid=key[0],
                        name=stat[stat.find("(") + 1 : end],
                        cpu_percent=load,
                    )
                )
            except FileNotFoundError:
                continue
        self.previous = (total, idle)
        self.processes, self.previous_at = current, now
        gpu = command(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        with urllib.request.urlopen(
            os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434") + "/api/ps",
            timeout=3,
        ) as response:
            resident = json.load(response)
        return dict(
            monotonic_s=now,
            cpu_percent=cpu,
            gpu={"utilization.gpu": gpu},
            processes=rows,
            ollama_ps_api=resident,
            errors=[],
        )


def assess(rows, config, *, admission=False, allowed_pids=()):
    reasons = set()
    if not rows:
        return ["telemetry_missing"]
    previous = None
    busy_previous = set()
    for row in rows:
        if row.get("errors"):
            reasons.add("telemetry_error")
        if previous is not None and row["monotonic_s"] - previous > 15:
            reasons.add("telemetry_gap")
        previous = row["monotonic_s"]
        models = row.get("ollama_ps_api", {}).get("models", [])
        if (
            len(models) != 1
            or models[0].get("digest", "").removeprefix("sha256:")
            != config["model_digest"]
        ):
            reasons.add("model_residency")
        elif (
            models[0].get("context_length", 4096) != 4096
            or datetime.fromisoformat(models[0]["expires_at"]).timestamp()
            < time.time() + 180
        ):
            reasons.add("residency_lease")
        busy = set()
        process_names = {p["pid"]: p["name"].lower() for p in row.get("processes", [])}
        for pid in row.get("gpu_pids", []):
            if pid not in set(allowed_pids) | {
                os.getpid()
            } and "ollama" not in process_names.get(pid, ""):
                reasons.add("foreign_gpu_process")
        for process in row.get("processes", []):
            name = process["name"].lower().removesuffix(".exe")
            if (
                name
                in (
                    "chrome",
                    "chromium",
                    "brave",
                    "msedge",
                    "firefox",
                    "opera",
                    "steam",
                )
                or "overlay" in name
            ):
                reasons.add("prohibited_process")
            exempt = process["pid"] in set(allowed_pids) | {os.getpid()} or name in (
                "idle",
                "system",
                "registry",
                "memory compression",
                "ollama",
                "ollama app",
            )
            if not exempt and (process.get("cpu_percent") or 0) >= 10:
                busy.add(process["pid"])
        if busy & busy_previous:
            reasons.add("external_cpu_load")
        busy_previous = busy
    if admission:
        if rows[-1]["monotonic_s"] - rows[0]["monotonic_s"] < 60:
            reasons.add("idle_window_incomplete")
        cpus = [r["cpu_percent"] for r in rows[1:] if r.get("cpu_percent") is not None]
        if len(cpus) != len(rows) - 1 or not cpus or sum(cpus) / len(cpus) >= 10:
            reasons.add("idle_cpu")
        try:
            gpus = [float(r["gpu"]["utilization.gpu"]) for r in rows]
            if sum(gpus) / len(gpus) >= 10:
                reasons.add("idle_gpu")
        except (KeyError, ValueError):
            reasons.add("gpu_telemetry_missing")
    return sorted(reasons)


class Environment:
    def __init__(self, config, root, allowed_pids=()):
        self.config, self.root, self.allowed_pids = config, Path(root), allowed_pids
        if os.name == "nt":
            from scripts.observe_interview_gate import Sampler

            self.sampler = Sampler()
        else:
            self.sampler = LinuxSampler()
        self.rows = []
        self.last = 0

    def sample(self):
        if time.monotonic() - self.last < 5:
            return
        row = self.sampler()
        listing = command(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        )
        row["gpu_pids"] = [
            int(pid.strip()) for pid in listing.splitlines() if pid.strip()
        ]
        self.rows.append(row)
        self.last = time.monotonic()
        atomic_json(self.root / "telemetry" / f"{len(self.rows):06d}.json", row)
        errors = assess(self.rows, self.config, allowed_pids=self.allowed_pids)
        if errors:
            raise ValueError("Live environment invalid: " + ",".join(errors))

    def preflight(self):
        before = identity(self.config)
        self.rows = []
        self.last = 0
        started = time.monotonic()
        while True:
            self.sample()
            if self.rows[-1]["monotonic_s"] - self.rows[0]["monotonic_s"] >= 60:
                break
            if time.monotonic() - started > 90:
                raise TimeoutError("Admission observation deadline")
            time.sleep(1)
        errors = assess(
            self.rows, self.config, admission=True, allowed_pids=self.allowed_pids
        )
        if os.name == "nt":
            from scripts.observe_interview_gate import assess as windows_assess

            errors.extend(
                windows_assess(
                    self.rows, admission=True, allowed_pids=self.allowed_pids
                )
            )
        after = identity(self.config)
        if errors or before != after:
            raise ValueError(
                "Preflight rejected: " + ",".join(errors or ["identity_changed"])
            )
        return dict(
            valid=True,
            synthetic=False,
            identity=after,
            observed_seconds=self.rows[-1]["monotonic_s"] - self.rows[0]["monotonic_s"],
            at=datetime.now(timezone.utc).isoformat(),
        )
