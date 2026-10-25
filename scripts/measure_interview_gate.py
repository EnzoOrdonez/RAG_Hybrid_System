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
            observer.response_started_at = now()
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
        elapsed = None
        try:
            try:
                payload = work()
            except Exception as exc:
                payload = dict(status="error", error=f"{type(exc).__name__}: {exc}")
            elapsed = time.perf_counter() - started
        finally:
            stop.set()
            thread.join()
            if observer is not None:
                observer.response_elapsed_s = elapsed if elapsed is not None else time.perf_counter() - started
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


def recipe(protocol):
    version = protocol.get('protocol_version', 1)
    if version == 1:
        return dict(read=60, connect=5, write=60, pool=60, keep_alive=None)
    expected = dict(read=180, connect=5, write=60, pool=60)
    if (version != 2 or protocol.get('http_timeouts') != expected
            or protocol.get('keep_alive') != '30m'
            or protocol.get('warm_preparation') != 'all_three_in_process'
            or protocol.get('acceptance') != 'prepared_warm_three_systems_v1'):
        raise ValueError('Unregistered deployment recipe')
    return dict(expected, keep_alive='30m')


def selected_conditions(protocol, system=None, phase=None):
    systems = selected_systems(protocol)
    if system is None and phase is None:
        return [(s, p) for s in systems for p in PHASES]
    if system not in systems or phase not in PHASES:
        raise ValueError('Condition selection requires registered system and phase')
    return [(system, phase)]


def window_has_margin(seconds=600):
    deadline = os.environ.get('CLOUDRAG_GATE_DEADLINE')
    return not deadline or (datetime.fromisoformat(deadline) - datetime.now(timezone.utc)).total_seconds() >= seconds


def summarize(rows, systems=None, protocol=None):
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
    report = dict(cells=cells, percentile_population="successful complete responses with valid controls only",
                warmups=dict(attempts=len(warmups), failures=sum(r["status"] != "success" for r in warmups)),
                passed=bool(cells) and all(c["completed_slots"] == 20 and c["failures"] == 0
                           and c["conditions_invalid"] == 0
                           and c["p95_s"] is not None and c["p95_s"] <= 60 for c in cells))
    if protocol is not None:
        recipe(protocol)
    if protocol and protocol.get('protocol_version') == 2:
        warm = [c for c in cells if c['phase'] == 'warm']
        report['acceptance'] = protocol['acceptance']
        report['passed'] = (len(cells) == 6 and {c['system'] for c in cells} == set(SYSTEMS)
            and all(c['completed_slots'] == 20 for c in cells)
            and all(c['successes'] == 20 and c['failures'] == c['conditions_invalid'] == 0
                    and c['p95_s'] is not None and c['p95_s'] <= 60 for c in warm))
        report['warmups']['description'] = 'Each all_system_preparation attempt contains three separate warmup queries and explicit NLI probes'
    return report


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
    if protocol.get('controls', {}).get('memory_trace'):
        from scripts import gate_etw, gate_memory
        if os.environ.get('CLOUDRAG_MEMORY_TRACE') != '1' or any(
            digest(path) != protocol['controls'][key] for key, path in (
                ('memory_sha256', gate_memory.__file__), ('etw_sha256', gate_etw.__file__),
                ('profile_sha256', gate_memory.PROFILE))):
            raise ValueError('Memory instrumentation changed during cohort')


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
                    seed=42, llm_cache=False, timeout_s=180, environment=environment,
                    protocol_version=2, http_timeouts=dict(read=180, connect=5, write=60, pool=60),
                    keep_alive='30m', warm_preparation='all_three_in_process',
                    acceptance='prepared_warm_three_systems_v1',
                    preregistration_sha256=digest(PROJECT / 'docs/WARM_GATE_PREREGISTRATION.md'),
                    build_id=git("rev-parse", "HEAD"), server=environment["server"],
                    model_digest=environment["model_digest"],
                    manifest_sha256=digest(os.environ["CLOUDRAG_ARTIFACT_MANIFEST"]),
                    artifact_manifest_path=str(Path(os.environ['CLOUDRAG_ARTIFACT_MANIFEST']).resolve()),
                    queries_sha256=digest(PROJECT / "data/evaluation/test_queries.json"),
                    cold="fresh process, Granite unloaded; OS file cache retained",
                    warm="persistent process, all three pipelines prepared; Granite residency checked per query",
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
        protocol['controls']['memory_trace'] = os.environ.get('CLOUDRAG_MEMORY_TRACE') == '1'
        if protocol['controls']['memory_trace']:
            from scripts import gate_etw, gate_memory
            protocol['controls']['memory_sha256'] = digest(gate_memory.__file__)
            protocol['controls']['etw_sha256'] = digest(gate_etw.__file__)
            protocol['controls']['profile_sha256'] = digest(gate_memory.PROFILE)
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
    policy = recipe(protocol)
    pipeline = None
    preparation = None
    scope = uuid.uuid4().hex
    if protocol.get('protocol_version') == 2 and phase == 'warm':
        from src.ui.components.interview_preparation import Preparation
        preparation = Preparation(root / 'preparation', factory=lambda key: attach_trace(load(key)))

    def load(key):
        import torch
        if torch.version.cuda is not None:
            raise ValueError('CPU-only auxiliary runtime required')
        from src.ui.components.index_loader import load_hybrid_index, load_pipeline
        return load_pipeline(key, _hybrid_index=load_hybrid_index())

    def attach_trace(candidate):
        llm = candidate.llm
        if (llm.cache_enabled or llm.seed != 42 or llm.timeout != policy['write'] or llm.max_retries != 1
                or (getattr(llm, 'read_timeout', None) or llm.timeout) != policy['read']
                or getattr(llm, 'default_keep_alive', None) != policy['keep_alive']):
            raise ValueError('Measured recipe differs from registered cohort')
        if observer is not None:
            import httpx
            import ollama
            from scripts.diagnose_interview_timeout import TracedClient
            if llm._ollama_client is None:
                llm._ollama_client = TracedClient(ollama.Client(host=os.environ['OLLAMA_HOST'],
                    timeout=httpx.Timeout(policy['write'], read=policy['read'], connect=policy['connect'])),
                    metadata['http_trace_path'])
            else:
                llm._ollama_client.output = Path(metadata['http_trace_path'])
        return candidate

    sequence = [-1] + indices if phase == "warm" else indices
    for index in sequence:
        if not window_has_margin(900 if preparation is not None and index == -1 else 600):
            write_new(root / 'pauses' / f'{uuid.uuid4().hex}.json',
                      dict(at=now(), reason='window_margin', system=system, phase=phase, next_index=index))
            return
        if preparation is not None and index != -1 and not preparation.ready(scope):
            raise RuntimeError('Warm preparation lost before starting the next position')
        check_model()
        check_environment(protocol)
        metadata = dict(system=system, phase=phase, index=index, warmup=index == -1,
                        query=protocol["queries"][max(0, index)], build_id=os.environ["CLOUDRAG_BUILD_ID"],
                        application_baseline_build_id=protocol["build_id"],
                        runner_sha256=digest(__file__), timeout_s=policy['read'],
                        http_timeouts={k: policy[k] for k in ('read', 'connect', 'write', 'pool')},
                        keep_alive=policy['keep_alive'],
                        consumes_slot=manifest.get("abort_consumes_slot", False),
                        environment_manifest_sha256=digest(root / "source-manifest.json"))
        if preparation is not None and index == -1:
            metadata['warmup_kind'] = 'all_system_preparation'
        observer = None
        if protocol.get("controls"):
            from scripts import observe_interview_gate as observe
            if digest(observe.__file__) != protocol["controls"]["observer_sha256"]:
                raise ValueError("Observer changed during cohort")
            if bool(protocol['controls'].get('memory_trace')) != (os.environ.get('CLOUDRAG_MEMORY_TRACE') == '1'):
                raise ValueError('Memory trace mode differs from cohort')
            trace_id = uuid.uuid4().hex
            observer = observe.Observer(root / "telemetry" / f"{trace_id}.jsonl",
                                         allowed_pids={os.getpid(), os.getppid()})
            metadata["http_trace_path"] = str(root / "http" / trace_id)
            metadata["control_journal_path"] = str(observer.path)

        def query():
            nonlocal pipeline
            if preparation is not None:
                if index == -1:
                    receipt = preparation.prepare(scope)
                    return dict(status='success', preparation_receipt=receipt,
                                model_digest=protocol['model_digest'])
                pipeline = preparation.pipeline(system, scope)
            elif pipeline is None:
                pipeline = load(system)
            attach_trace(pipeline)
            llm = pipeline.llm
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
                        preparation_id=getattr(pipeline, 'interview_preparation_id', None),
                        preparation_check=getattr(pipeline, 'interview_preparation_check', None),
                        artifact_manifest_sha256=pipeline.hybrid_index.deployment_manifest_sha256)

        row = measure_attempt(root, metadata, query, validate_after=lambda: check_environment(protocol),
                              observer=observer)
        print(json.dumps({k: row[k] for k in ("system", "phase", "index", "status", "elapsed_s")}), flush=True)
        check_model()
        if row.get("environment_invalid") or row.get("conditions_invalid"):
            raise RuntimeError("Environment changed during attempt; excluded from percentiles")
        if index == -1 and row["status"] != "success":
            raise RuntimeError("Warmup failed; warm condition not established")


def run(source, root, system=None, phase=None):
    if source is not None:
        initialize(source, root)
    with coordinator_lock(root):
        recover(root)
        protocol = read_json(root / "source-manifest.json")["protocol"]
        systems = selected_systems(protocol)
        conditions = selected_conditions(protocol, system, phase)
        recipe(protocol)
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
                if (system, phase) not in conditions:
                    continue
                indices = [i for s, p, i in pending(all_records(root), systems) if (s, p) == (system, phase)]
                batches = [[i] for i in indices] if phase == "cold" else ([indices] if indices else [])
                for batch in batches:
                    if not window_has_margin(900 if phase == 'warm' else 600):
                        write_new(root / 'pauses' / f'{uuid.uuid4().hex}.json',
                                  dict(at=now(), reason='window_margin', system=system, phase=phase))
                        return
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
                    report = summarize(all_records(root), systems, protocol=protocol)
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
    if args.operation not in ('run', '_worker') and (args.system or args.phase):
        parser.error('system/phase apply only to run or worker')
    if args.operation == 'run' and bool(args.system) != bool(args.phase):
        parser.error('select both system and phase for a bounded window')
    if args.operation == "_worker" and (not args.system or not args.phase or not args.indices
            or len(set(args.indices)) != len(args.indices) or any(i < 0 or i > 19 for i in args.indices)):
        parser.error("worker requires system, phase and distinct indices 0..19")
    if args.operation == "init":
        print(json.dumps(initialize_fresh(args.output, fresh_protocol(args.systems, args.controlled)), indent=2))
    elif args.operation == "plan":
        rows = legacy_records(args.source) if not args.output or not args.output.exists() else all_records(args.output)
        systems = selected_systems(read_json(args.output / "source-manifest.json")["protocol"]) if (
            args.output and args.output.exists()) else SYSTEMS
        protocol = read_json(args.output / 'source-manifest.json')['protocol'] if args.output and args.output.exists() else None
        print(json.dumps(dict(pending=pending(rows, systems), report=summarize(rows, systems, protocol=protocol)), indent=2))
    elif args.operation == "report":
        systems = selected_systems(read_json(args.output / "source-manifest.json")["protocol"])
        protocol = read_json(args.output / 'source-manifest.json')['protocol']
        print(json.dumps(summarize(all_records(args.output), systems, protocol=protocol), indent=2))
    elif args.operation == "run":
        run(args.source, args.output, args.system, args.phase)
    else:
        worker(args.output, args.system, args.phase, args.indices)


if __name__ == "__main__":
    main()
