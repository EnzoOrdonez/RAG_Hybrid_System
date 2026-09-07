"""Durable, resumable operational cohort. All generated evidence stays outside git.

plan/report are read-only. init starts an independent cohort; --source explicitly
selects legacy import. Each physical attempt has a durable journal and terminal record.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

from filelock import FileLock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
SYSTEMS = ("hybrid", "lexical", "semantic")
PHASES = ("cold", "warm")


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, data):
    """Publish a complete file atomically, refusing replacement on Windows/Linux."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)  # exclusive atomic publication, same filesystem
    finally:
        temp.unlink(missing_ok=True)  # only this invocation's unpublished temporary


def read_events(path):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    events = []
    for i, line in enumerate(lines):
        if not line.endswith(b"\n") and i == len(lines) - 1:
            break  # a torn tail is preserved, not interpreted as an event
        events.append(json.loads(line))
    return events


def coordinator_lock(root):
    return FileLock(str(Path(root) / "run.lock"), timeout=0)


def measure_attempt(root, metadata, work, interval=1.0, validate_after=None, observer=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "inference.lock"), timeout=0):
        attempt = dict(metadata, attempt_id=uuid.uuid4().hex, started_at=now(),
                       worker_pid=os.getpid())
        directory = root / "attempts" / attempt["attempt_id"]
        directory.mkdir(parents=True)
        write_new(directory / "request.json", attempt)
        journal = directory / "events.jsonl"
        if observer is not None:
            observer.start()
        started = time.perf_counter()
        stop = threading.Event()
        write_lock = threading.Lock()
        heartbeat_errors = []

        def event(kind):
            record = dict(event=kind, at=now(), elapsed_lower_bound_s=time.perf_counter() - started)
            if kind == "start":
                record["attempt"] = attempt
            with write_lock, journal.open("ab") as stream:
                stream.write((json.dumps(record, allow_nan=False) + "\n").encode())
                stream.flush()
                os.fsync(stream.fileno())

        def heartbeat():
            while not stop.wait(interval):
                try:
                    event("heartbeat")
                except Exception as exc:
                    heartbeat_errors.append(type(exc).__name__)
                    return

        event("start")
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            try:
                payload = work()
            except Exception as exc:
                payload = dict(status="error", error=f"{type(exc).__name__}: {exc}")
            elapsed = time.perf_counter() - started
        finally:
            stop.set()
            thread.join()
            control = observer.finish() if observer is not None else {}
        if heartbeat_errors:
            raise OSError("Heartbeat persistence failed: " + ",".join(heartbeat_errors))
        if not isinstance(payload, dict) or payload.get("status") not in ("success", "error"):
            raise ValueError("Work must return a success/error payload")
        if validate_after is not None:
            try:
                validate_after()  # identity checks are outside the response clock
            except Exception as exc:
                payload = dict(payload, status="error", environment_invalid=True,
                               error=f"Environment validation: {type(exc).__name__}: {exc}")
        result = dict(payload, **attempt, **control, elapsed_s=elapsed, finished_at=now(),
                      journal_sha256=digest(journal))
        write_new(directory / "result.json", result)
        return result


def recover(root):
    """Only a free OS-backed inference lock permits declaring unfinished work dead."""
    root = Path(root)
    recovered = []
    with FileLock(str(root / "inference.lock"), timeout=0):
        for request in sorted(root.glob("attempts/*/request.json")):
            journal = request.with_name("events.jsonl")
            result_path = request.with_name("result.json")
            if result_path.exists():
                continue
            events = read_events(journal) if journal.exists() else []
            if events and events[0].get("event") != "start":
                raise ValueError(f"Invalid durable start: {journal}")
            result = dict(read_json(request), status="aborted", elapsed_s=None,
                          elapsed_lower_bound_s=events[-1]["elapsed_lower_bound_s"] if events else None,
                          error="interrupted; originating cause undetermined", recovered_at=now(),
                          finished_at=None, journal_sha256=digest(journal) if journal.exists() else None)
            write_new(result_path, result)
            recovered.append(result)
    return recovered


def local_records(root):
    return [read_json(path) for path in sorted(Path(root).glob("attempts/*/result.json"))]


def completed_slots(rows):
    slots = [(r["system"], r["phase"], r["index"]) for r in rows
             if not r.get("warmup") and (r["status"] in ("success", "error")
                                        or r.get("consumes_slot", False))]
    if len(slots) != len(set(slots)):
        raise ValueError("Duplicate completed logical slot")
    return set(slots)


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid successful duration")
    position = (len(values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def selected_systems(protocol):
    systems = tuple(protocol.get("systems", SYSTEMS))
    if "systems" in protocol and (not systems or len(set(systems)) != len(systems)
                                  or any(s not in SYSTEMS for s in systems)):
        raise ValueError("Invalid cohort systems")
    return systems


def summarize(rows, systems=None):
    cells = []
    for system in SYSTEMS if systems is None else systems:
        for phase in PHASES:
            selected = [r for r in rows if r["system"] == system and r["phase"] == phase
                        and not r.get("warmup")]
            durations = [r["elapsed_s"] for r in selected if r["status"] == "success"
                         and not r.get("conditions_invalid") and not r.get("environment_invalid")]
            failures = sum(r["status"] != "success" for r in selected)
            invalid = sum(bool(r.get("conditions_invalid") or r.get("environment_invalid")) for r in selected)
            cells.append(dict(system=system, phase=phase, attempts=len(selected),
                              completed_slots=len(completed_slots(selected)), successes=len(durations),
                              failures=failures, conditions_invalid=invalid,
                              valid_attempts=len(selected) - invalid,
                              aborted=sum(r["status"] == "aborted" for r in selected),
                              failure_rate=failures / len(selected) if selected else None,
                              p50_s=percentile(durations, .5), p95_s=percentile(durations, .95)))
    warmups = [r for r in rows if r.get("warmup")]
    return dict(cells=cells, percentile_population="successful complete responses with valid controls only",
                warmups=dict(attempts=len(warmups), failures=sum(r["status"] != "success" for r in warmups)),
                passed=bool(cells) and all(c["completed_slots"] == 20 and c["failures"] == 0
                           and c["conditions_invalid"] == 0
                           and c["p95_s"] is not None and c["p95_s"] <= 60 for c in cells))


def legacy_records(source):
    source = Path(source)
    protocol = read_json(source / "protocol.json")
    if protocol["seed"] != 42 or protocol["llm_cache"] or len(protocol["queries"]) != 20:
        raise ValueError("Legacy protocol is incompatible")
    rows = []
    for system in SYSTEMS:
        for phase in PHASES:
            for path in sorted(source.glob(f"{system}-{phase}-*.json")):
                row = read_json(path)
                if row["query"] != protocol["queries"][max(0, row["index"])]:
                    raise ValueError("Legacy query does not match protocol")
                row.update(status="success" if row["success"] else "error",
                           evidence_path=str(path), evidence_sha256=digest(path))
                rows.append(row)
    completed_slots(rows)
    closed = source / "lexical-cold-10.worker.log"
    if closed.exists() and "window-CLOSE event" in closed.read_text(encoding="utf-8-sig"):
        rows.append(dict(system="lexical", phase="cold", index=10, warmup=False,
                         status="aborted", elapsed_s=None, elapsed_lower_bound_s=None,
                         error="window-CLOSE event; originating cause undetermined",
                         evidence_path=str(closed), evidence_sha256=digest(closed)))
    return rows


def pending(rows, systems=None):
    completed = completed_slots(rows)
    return [(s, p, i) for s in (SYSTEMS if systems is None else systems) for p in PHASES for i in range(20)
            if (s, p, i) not in completed]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def validate_app_identity(original_build):
    # The new recorder changes the runner commit, never the app under measurement.
    if git("diff", original_build, "--", "src") or git("ls-files", "--others", "--exclude-standard", "src"):
        raise ValueError("Application source changed; use a separately approved cohort")
    if os.environ.get("CLOUDRAG_BUILD_ID") != git("rev-parse", "HEAD"):
        raise ValueError("CLOUDRAG_BUILD_ID must equal current HEAD")


def api(path, body=None):
    request = urllib.request.Request(os.environ["OLLAMA_HOST"].rstrip("/") + path,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def check_model():
    models = api("/api/tags")["models"]
    from src.pipeline.pipeline_config import SURVEY_DEPLOY
    model = SURVEY_DEPLOY.llm_model
    if next(m["digest"] for m in models if m["name"] == model) != os.environ["CLOUDRAG_MODEL_DIGEST"]:
        raise ValueError("Provisioned model digest changed; stop for operator")
    return model


def preflight(protocol):
    required = dict(CLOUDRAG_MODE="participant", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                    PYTHONHASHSEED="42")
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise ValueError("Participant/offline/seed environment mismatch")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("Auxiliary models must use CPU")
    from src.utils.deployment_artifacts import verify_manifest
    if verify_manifest(PROJECT, os.environ["CLOUDRAG_ARTIFACT_MANIFEST"]) != protocol["manifest_sha256"]:
        raise ValueError("Artifact identity changed")
    validate_app_identity(protocol["build_id"])
    check_model()
    if api("/api/version") != protocol["server"]:
        raise ValueError("Ollama server version changed; separate cohort required")
    check_environment(protocol)


def environment_identity():
    """Stable runtime identity; no model execution and no utilization samples."""
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits"], text=True).strip()
    hardware = json.loads(subprocess.check_output([
        "powershell", "-NoProfile", "-Command",
        "@{cpu=@(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors);"
        "ram=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory} | ConvertTo-Json -Depth 4"],
        text=True))
    model = check_model()
    return dict(gpu=gpu, hardware=hardware, server=api("/api/version"),
                model=model, model_digest=os.environ["CLOUDRAG_MODEL_DIGEST"],
                commit=git("rev-parse", "HEAD"), runner_sha256=digest(__file__),
                python=sys.version, packages={d.metadata["Name"]: d.version
                                             for d in importlib.metadata.distributions()},
                source_sha256={p.relative_to(PROJECT).as_posix(): digest(p)
                               for p in sorted((PROJECT / "src").rglob("*.py"))},
                lock_sha256=digest(PROJECT / "requirements-app.txt"))


def check_environment(protocol):
    if "environment" in protocol and environment_identity() != protocol["environment"]:
        raise ValueError("Cohort environment changed; stop and request a separate cohort")


def initialize_fresh(root, protocol):
    root = Path(root).resolve()
    common_git = Path(git("rev-parse", "--git-common-dir")).resolve()
    if root.is_relative_to(common_git.parent):
        raise ValueError("Evidence output must be outside checkout")
    manifest = dict(mode="fresh", abort_consumes_slot=True,
                    total_attempts=len(selected_systems(protocol)) * 40, protocol=protocol)
    path = root / "source-manifest.json"
    if path.exists():
        existing = read_json(path)
        if {k: v for k, v in existing.items() if k != "created_at"} != manifest:
            raise ValueError("Cohort identity differs; existing evidence preserved")
        return existing
    if list(root.glob("attempts/*/request.json")):
        raise ValueError("Cohort identity missing from nonempty attempt directory")
    manifest["created_at"] = now()
    write_new(path, manifest)
    return manifest


def fresh_protocol(systems=None, controlled=False):
    from src.ui.components.session_manager import _get_evaluation_queries
    queries = _get_evaluation_queries()
    selected = [queries[i * 30 // 20] for i in range(20)]
    environment = environment_identity()
    if any(line.split(",")[2].strip() != "616.64" for line in environment["gpu"].splitlines()):
        raise ValueError("New cohort requires authorized driver 616.64")
    protocol = dict(purpose="TECHNICAL SYNTHETIC VALIDATION ONLY", queries=selected,
                    seed=42, llm_cache=False, timeout_s=60, environment=environment,
                    build_id=git("rev-parse", "HEAD"), server=environment["server"],
                    model_digest=environment["model_digest"],
                    manifest_sha256=digest(os.environ["CLOUDRAG_ARTIFACT_MANIFEST"]),
                    queries_sha256=digest(PROJECT / "data/evaluation/test_queries.json"),
                    cold="fresh process, Granite unloaded; OS file cache retained",
                    warm="persistent process, successful unmeasured warmup",
                    percentile_population="successful complete responses only")
    if systems is not None:
        protocol["systems"] = list(systems)
        selected_systems(protocol)
    if controlled:
        from scripts import observe_interview_gate as observe
        protocol["controls"] = dict(version=1, sample_interval_s=5, idle_window_s=60,
                                    power_scheme=observe.BALANCED, overlay=observe.BEST_PERFORMANCE,
                                    ac_required=True, idle_cpu_gpu_below_percent=10,
                                    pause_on_invalid=True, replace_attempts=False,
                                    observer_sha256=digest(observe.__file__))
    preflight(protocol)
    return protocol


def initialize(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    common_git = Path(git("rev-parse", "--git-common-dir")).resolve()
    if root.is_relative_to(common_git.parent) or root.is_relative_to(source):
        raise ValueError("Evidence output must be outside checkout and legacy run")
    inventory = {str(p.relative_to(source)): digest(p) for p in sorted(source.iterdir()) if p.is_file()}
    manifest = dict(source=str(source), files=inventory, protocol=read_json(source / "protocol.json"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "source-manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Legacy evidence inventory changed")
    else:
        write_new(path, manifest)
    return manifest


def all_records(root):
    manifest = read_json(Path(root) / "source-manifest.json")
    if manifest.get("mode") == "fresh":
        rows = local_records(root)
        if any(not r.get("consumes_slot") for r in rows):
            raise ValueError("Fresh attempt missing fixed-slot policy")
        completed_slots(rows)
        systems = selected_systems(manifest["protocol"])
        if any(("systems" in manifest["protocol"] and r["system"] not in systems) or r["phase"] not in PHASES
               or (r["index"] != -1 if r.get("warmup") else not 0 <= r["index"] < 20) for r in rows):
            raise ValueError("Attempt outside immutable cohort slots")
        return rows
    for name, expected in manifest["files"].items():
        if digest(Path(manifest["source"]) / name) != expected:
            raise ValueError("Legacy evidence hash changed")
    return legacy_records(manifest["source"]) + local_records(root)


def worker(root, system, phase, indices):
    manifest = read_json(root / "source-manifest.json")
    protocol = manifest["protocol"]
    if system not in selected_systems(protocol):
        raise ValueError("Worker system outside cohort")
    pipeline = None
    sequence = [-1] + indices if phase == "warm" else indices
    for index in sequence:
        check_model()
        check_environment(protocol)
        metadata = dict(system=system, phase=phase, index=index, warmup=index == -1,
                        query=protocol["queries"][max(0, index)], build_id=os.environ["CLOUDRAG_BUILD_ID"],
                        application_baseline_build_id=protocol["build_id"],
                        runner_sha256=digest(__file__), timeout_s=60,
                        consumes_slot=manifest.get("abort_consumes_slot", False),
                        environment_manifest_sha256=digest(root / "source-manifest.json"))
        observer = None
        if protocol.get("controls"):
            from scripts import observe_interview_gate as observe
            if digest(observe.__file__) != protocol["controls"]["observer_sha256"]:
                raise ValueError("Observer changed during cohort")
            trace_id = uuid.uuid4().hex
            observer = observe.Observer(root / "telemetry" / f"{trace_id}.jsonl",
                                         allowed_pids={os.getpid(), os.getppid()})
            metadata["http_trace_path"] = str(root / "http" / trace_id)
            metadata["control_journal_path"] = str(observer.path)

        def query():
            nonlocal pipeline
            if pipeline is None:
                import torch
                if torch.version.cuda is not None:
                    raise ValueError("CPU-only auxiliary runtime required")
                from src.ui.components.index_loader import load_hybrid_index, load_pipeline
                pipeline = load_pipeline(system, _hybrid_index=load_hybrid_index())
            llm = pipeline.llm
            if llm.cache_enabled or llm.seed != 42 or llm.timeout != 60 or llm.max_retries != 1:
                raise ValueError("Measured recipe differs from original cohort")
            if observer is not None:
                import httpx
                import ollama
                from scripts.diagnose_interview_timeout import TracedClient
                if llm._ollama_client is None:
                    llm._ollama_client = TracedClient(ollama.Client(host=os.environ["OLLAMA_HOST"],
                        timeout=httpx.Timeout(llm.timeout, connect=5)), metadata["http_trace_path"])
                else:
                    llm._ollama_client.output = Path(metadata["http_trace_path"])
            response = pipeline.query(metadata["query"]["question"]).model_dump(mode="json")
            report = response.get("hallucination_report")
            error = response.get("error")
            if not error and (not response["answer"].strip() or response["confidence"] == "ERROR"):
                error = "empty_response"
            if not error and report and report["method"] in ("keyword_fallback", "mixed"):
                error = "verification_unavailable"
            return dict(status="error" if error else "success", error=error, response=response,
                        configuration=pipeline.config.model_dump(mode="json"), seed=llm.seed,
                        cache_enabled=llm.cache_enabled, num_ctx=llm.num_ctx, model_digest=llm.model_digest,
                        artifact_manifest_sha256=pipeline.hybrid_index.deployment_manifest_sha256)

        row = measure_attempt(root, metadata, query, validate_after=lambda: check_environment(protocol),
                              observer=observer)
        print(json.dumps({k: row[k] for k in ("system", "phase", "index", "status", "elapsed_s")}), flush=True)
        check_model()
        if row.get("environment_invalid") or row.get("conditions_invalid"):
            raise RuntimeError("Environment changed during attempt; excluded from percentiles")
        if index == -1 and row["status"] != "success":
            raise RuntimeError("Warmup failed; warm condition not established")


def run(source, root):
    if source is not None:
        initialize(source, root)
    with coordinator_lock(root):
        recover(root)
        protocol = read_json(root / "source-manifest.json")["protocol"]
        systems = selected_systems(protocol)
        existing = all_records(root)
        recorded_digests = {r["model_digest"] for r in existing if r.get("model_digest")}
        if protocol.get("model_digest"):
            recorded_digests.add(protocol["model_digest"])
        if recorded_digests != {os.environ.get("CLOUDRAG_MODEL_DIGEST")}:
            raise ValueError("Model digest differs from recorded cohort")
        preflight(protocol)
        if protocol.get("controls"):
            from scripts.observe_interview_gate import admission
            admission(root / "admissions" / uuid.uuid4().hex)
        write_new(root / "invocations" / f"{uuid.uuid4().hex}.json",
                  dict(at=now(), coordinator_pid=os.getpid(), build_id=git("rev-parse", "HEAD"),
                       runner_sha256=digest(__file__), source_manifest_sha256=digest(root / "source-manifest.json"),
                       pending_slots=pending(existing, systems), interrupted_cohort=bool(existing), heartbeat_interval_s=1.0))
        for system in systems:
            for phase in PHASES:
                indices = [i for s, p, i in pending(all_records(root), systems) if (s, p) == (system, phase)]
                batches = [[i] for i in indices] if phase == "cold" else ([indices] if indices else [])
                for batch in batches:
                    with FileLock(str(root / "inference.lock"), timeout=0):
                        model = check_model()
                        if phase == "cold":
                            if any(m["name"] != model for m in api("/api/ps")["models"]):
                                raise RuntimeError("Concurrent Ollama model detected")
                            unloaded = api("/api/generate", {"model": model, "keep_alive": 0})
                            after = api("/api/ps")
                            if after["models"]:
                                raise RuntimeError("Model did not unload")
                            write_new(root / "unloads" / f"{uuid.uuid4().hex}.json",
                                      dict(at=now(), system=system, index=batch[0], unload=unloaded, ps_after=after))
                    command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--output", str(root),
                               "--system", system, "--phase", phase, "--indices", *map(str, batch)]
                    with (root / f"worker-{uuid.uuid4().hex}.log").open("x", encoding="utf-8") as log:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    report = summarize(all_records(root), systems)
                    write_new(root / "reports" / f"{uuid.uuid4().hex}.json", dict(at=now(), **report))
                    print(json.dumps(report), flush=True)
                    if result.returncode:
                        raise RuntimeError("Worker stopped; inspect journal and log before resuming")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("init", "plan", "run", "report", "_worker"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, help="init only; immutable system subset")
    parser.add_argument("--controlled", action="store_true", help="init only; require observed idle admission")
    args = parser.parse_args()
    if args.operation == "plan" and args.source is None and args.output is None:
        parser.error("--source or --output is required")
    if args.operation in ("init", "run", "report", "_worker") and args.output is None:
        parser.error("--output is required")
    if args.operation == "init" and args.source is not None:
        parser.error("init cannot import legacy evidence")
    if args.operation != "init" and (args.systems is not None or args.controlled):
        parser.error("cohort controls/selection may only be set at init")
    if args.operation == "_worker" and (not args.system or not args.phase or not args.indices
            or len(set(args.indices)) != len(args.indices) or any(i < 0 or i > 19 for i in args.indices)):
        parser.error("worker requires system, phase and distinct indices 0..19")
    if args.operation == "init":
        print(json.dumps(initialize_fresh(args.output, fresh_protocol(args.systems, args.controlled)), indent=2))
    elif args.operation == "plan":
        rows = legacy_records(args.source) if not args.output or not args.output.exists() else all_records(args.output)
        systems = selected_systems(read_json(args.output / "source-manifest.json")["protocol"]) if (
            args.output and args.output.exists()) else SYSTEMS
        print(json.dumps(dict(pending=pending(rows, systems), report=summarize(rows, systems)), indent=2))
    elif args.operation == "report":
        systems = selected_systems(read_json(args.output / "source-manifest.json")["protocol"])
        print(json.dumps(summarize(all_records(args.output), systems), indent=2))
    elif args.operation == "run":
        run(args.source, args.output)
    else:
        worker(args.output, args.system, args.phase, args.indices)


if __name__ == "__main__":
    main()
