"""Auditable two-window gate; only a verified real aggregate may decide GO."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.ui.components.session_storage import atomic_json
from src.ui.components.study_protocol import digest

TASKS = ("q001", "q064", "q171", "q010", "q070", "q172")
WINDOW_SECONDS = 120 * 60


def schedule(window, tasks=TASKS):
    if window not in (1, 2) or len(tasks) != 6 or len(set(tasks)) != 6:
        raise ValueError("Require window 1/2 and six distinct sealed tasks")
    return [
        dict(window=window, repetition=r, query_id=q, condition=c)
        for r in (range(1, 6) if window == 1 else range(6, 11))
        for pos, q in enumerate(tasks)
        for c in (("hybrid", "no_rag") if (r + pos) % 2 else ("no_rag", "hybrid"))
    ]


def summarize(rows):
    systems = {}
    for condition in ("hybrid", "no_rag"):
        group = [r for r in rows if r["condition"] == condition]
        values = [
            r["elapsed_s"] for r in group if r["status"] == "success" and r["valid"]
        ]
        systems[condition] = dict(
            n=len(values),
            failures=sum(r["status"] != "success" for r in group),
            invalid=sum(not r["valid"] for r in group),
            p50=float(np.percentile(values, 50)) if values else None,
            p95=float(np.percentile(values, 95)) if values else None,
        )
    return systems


def seal(root):
    files = {
        p.relative_to(root).as_posix(): digest(p)
        for p in sorted(root.rglob("*.json"))
        if p.name != "manifest.json"
    }
    atomic_json(root / "manifest.json", dict(files=files))


def verify_window(root, *, expected_window=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    inventory = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*.json")
        if p.name != "manifest.json"
    }
    if set(manifest["files"]) != inventory:
        raise ValueError("Window file inventory changed")
    for name, sha in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != sha:
            raise ValueError("Window hash mismatch")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    planned = json.loads((root / "schedule.json").read_text(encoding="utf-8"))
    rows = json.loads((root / "attempts.json").read_text(encoding="utf-8"))
    if expected_window and summary["window"] != expected_window:
        raise ValueError("Wrong preceding window")
    if summary["status"] != "complete" or len(rows) != 60 or len(planned) != 60:
        raise ValueError("Window is incomplete or terminal")
    if planned != schedule(summary["window"], summary["tasks"]):
        raise ValueError("Window calendar differs from sealed tasks")
    for index, (row, slot) in enumerate(zip(rows, planned), 1):
        if (
            any(row.get(k) != v for k, v in slot.items())
            or row["status"] != "success"
            or not row["valid"]
            or not isinstance(row["elapsed_s"], (int, float))
            or not math.isfinite(row["elapsed_s"])
            or row["elapsed_s"] < 0
        ):
            raise ValueError("Window schedule or validity mismatch")
        if (
            json.loads((root / "attempts" / f"{index:03d}-result.json").read_text())
            != row
        ):
            raise ValueError("Published attempt mismatch")
    if summarize(rows) != summary["systems"]:
        raise ValueError("Window statistics differ from attempts")
    return summary


def run(
    output,
    *,
    window,
    dry_run=False,
    adapter=None,
    preflight=None,
    previous=None,
    tasks=TASKS,
    monotonic=time.monotonic,
    operator_zoom_active=False,
    screen_share_declared=False,
    backup_pending=False,
):
    """Legacy Zoom arguments have no authority under the cloud amendment."""
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("Evidence directory must be new")
    first = (
        verify_window(previous, expected_window=1) if window == 2 and previous else None
    )
    if window == 2 and first is None:
        raise ValueError("Window 2 requires intact window 1")
    if not dry_run and (adapter is None or preflight is None):
        raise ValueError("Real supervisor and live preflight required")
    root.mkdir(parents=True)
    plan = schedule(window, tasks)
    atomic_json(root / "schedule.json", plan)
    rows = []
    started_at = datetime.now(timezone.utc).isoformat()
    status = "PREFLIGHT_FAILED"
    try:
        admission = (
            preflight()
            if preflight
            else dict(
                valid=not backup_pending, synthetic=True, identity={"mode": "synthetic"}
            )
        )
        atomic_json(root / "preflight.json", admission)
        if not admission["valid"] or bool(admission["synthetic"]) != dry_run:
            raise ValueError("Preflight rejected")
        if first:
            if first["tasks"] != list(tasks):
                raise ValueError("Sealed tasks changed between windows")
            prior = json.loads((Path(previous) / "preflight.json").read_text())
            if prior["identity"] != admission["identity"] or first["mode"] != (
                "SYNTHETIC_ONLY" if dry_run else "REAL"
            ):
                raise ValueError("Identity or mode changed between windows")
            atomic_json(
                root / "previous.json",
                {"manifest_sha256": digest(Path(previous) / "manifest.json")},
            )
        started = monotonic()
        if hasattr(adapter, "deadline"):
            adapter.deadline = time.monotonic() + WINDOW_SECONDS
        status = "complete"
        for index, slot in enumerate(plan, 1):
            if monotonic() - started >= WINDOW_SECONDS:
                status = "WINDOW_EXPIRED"
                break
            request = dict(slot, attempt_id=f"w{window}-{index:03d}")
            atomic_json(root / "attempts" / f"{index:03d}-started.json", request)
            try:
                elapsed, valid, error = (
                    adapter(request) if adapter else (0.001, True, None)
                )
                if error is None and (
                    not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed)
                    or elapsed < 0
                ):
                    raise ValueError("Invalid elapsed time")
                row = dict(
                    slot,
                    status="success" if error is None else "error",
                    valid=bool(valid),
                    error=error,
                    elapsed_s=elapsed if error is None else None,
                )
            except BaseException as exc:
                row = dict(
                    slot,
                    status="aborted",
                    valid=False,
                    error=type(exc).__name__,
                    elapsed_s=None,
                )
            rows.append(row)
            atomic_json(root / "attempts" / f"{index:03d}-result.json", row)
            atomic_json(root / "attempts.json", rows)
            print(
                f"{index}/60 {slot['condition']} ETA={(60 - index) * (monotonic() - started) / index:.1f}s",
                flush=True,
            )
            if row["status"] != "success" or not row["valid"]:
                status = "INVALID_TERMINAL"
                break
    except BaseException as exc:
        if status == "complete":
            status = "INTERRUPTED_TERMINAL"
        atomic_json(
            root / "terminal.json", dict(error=type(exc).__name__, status=status)
        )
        raise
    finally:
        atomic_json(root / "attempts.json", rows)
        summary = dict(
            mode="SYNTHETIC_ONLY" if dry_run else "REAL",
            window=window,
            tasks=list(tasks),
            status=status,
            started_at=started_at,
            systems=summarize(rows),
            go_decision="NOT_A_GO_DECISION",
        )
        atomic_json(root / "summary.json", summary)
        seal(root)
    return summary


def aggregate(root):
    root = Path(root)
    first = verify_window(root / "window-1", expected_window=1)
    second = verify_window(root / "window-2", expected_window=2)
    link = json.loads((root / "window-2" / "previous.json").read_text())
    if (
        link["manifest_sha256"] != digest(root / "window-1" / "manifest.json")
        or first["mode"] != second["mode"]
    ):
        raise ValueError("Aggregate linkage mismatch")
    preflights = [
        json.loads((root / w / "preflight.json").read_text())
        for w in ("window-1", "window-2")
    ]
    if preflights[0]["identity"] != preflights[1]["identity"]:
        raise ValueError("Aggregate identity mismatch")
    rows = []
    for window in ("window-1", "window-2"):
        rows.extend(json.loads((root / window / "attempts.json").read_text()))
    systems = summarize(rows)
    eligible = all(
        v["n"] == 60 and v["failures"] == 0 and v["invalid"] == 0 and v["p95"] <= 60
        for v in systems.values()
    )
    result = dict(
        mode=first["mode"],
        status="complete",
        systems=systems,
        windows=[first, second],
        go_decision="SYNTHETIC_NOT_GO"
        if first["mode"] == "SYNTHETIC_ONLY"
        else ("GO" if eligible else "NO_GO"),
        manifests={
            w: digest(root / w / "manifest.json") for w in ("window-1", "window-2")
        },
    )
    atomic_json(root / "aggregate.json", result)
    return result


def run_cohort(
    output,
    *,
    dry_run=False,
    adapter=None,
    preflight_factory=None,
    tasks=TASKS,
    operator_zoom_active=False,
    screen_share_declared=False,
    backup_pending=False,
):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError("Cohort evidence directory must be new")
    root.mkdir(parents=True)
    for window in (1, 2):
        child = root / f"window-{window}"
        result = run(
            child,
            window=window,
            dry_run=dry_run,
            adapter=adapter,
            tasks=tasks,
            preflight=preflight_factory(child) if preflight_factory else None,
            previous=root / "window-1" if window == 2 else None,
            backup_pending=backup_pending,
        )
        if result["status"] != "complete":
            result = dict(
                mode="SYNTHETIC_ONLY" if dry_run else "REAL",
                status=f"WINDOW_{window}_TERMINAL",
                go_decision="SYNTHETIC_NOT_GO" if dry_run else "NO_GO",
            )
            atomic_json(root / "aggregate.json", result)
            return result
    return aggregate(root)


def make_app_adapter(
    protocol, pipeline_factory, *, clock=time.perf_counter, evidence_root=None
):
    from src.ui.components.study_service import QueryTimer, execute_query

    if evidence_root is not None and "config" in protocol:
        from src.ui.components import study_service
        from src.ui.components.study_protocol import CELLS
        from src.ui.components.study_sessions import StudyStore

        store = StudyStore(
            Path(evidence_root) / "sessions", protocol, purpose="technical"
        )
        store.freeze()
        previous = [None]
        sequence = [0]

        def session_adapter(planned):
            receipt = Path(evidence_root) / (planned["attempt_id"] + ".json")
            if receipt.exists():
                raise FileExistsError("Attempt already exists; no replay")
            if previous[0] is not None:
                store.abandon(previous[0])
            task_set = next(
                k
                for k, v in protocol["config"]["tasks"].items()
                if planned["query_id"] in v
            )
            cell = next(
                k
                for k, blocks in CELLS.items()
                if blocks[0][1] == task_set
                and protocol["config"]["labels"][blocks[0][0]] == planned["condition"]
            )
            sequence[0] += 1
            token = store.issue(
                f"P{900000 + sequence[0]}", cell=cell, profile="without_experience"
            )
            session = store.admit(token)
            previous[0] = session.session_id
            session.familiarization_done()
            session.data["task_index"] = protocol["config"]["tasks"][task_set].index(
                planned["query_id"]
            )
            session.save()
            atomic_json(
                receipt,
                dict(
                    planned,
                    session_id=session.session_id,
                    purpose="technical",
                    status="prepared",
                ),
            )
            captured = []
            study_service.answer(
                session, pipeline_factory, clock=clock, capture=captured.append
            )
            attempt = session.pending
            atomic_json(
                Path(evidence_root) / (planned["attempt_id"] + "-response.json"),
                attempt,
            )
            if captured and hasattr(captured[0], "model_dump"):
                atomic_json(
                    Path(evidence_root) / (planned["attempt_id"] + "-pipeline.json"),
                    captured[0].model_dump(mode="json"),
                )
            error = attempt["error"]
            return (
                attempt["elapsed_ms"] / 1000 if error is None else None,
                error is None,
                error,
            )

        return session_adapter

    def adapter(planned):
        question = protocol["queries"][planned["query_id"]]["question"]
        timer = QueryTimer(clock)
        if evidence_root:
            path = Path(evidence_root) / (planned["attempt_id"] + ".json")
            if path.exists():
                raise FileExistsError("Attempt already exists; no replay")
            atomic_json(path, dict(planned, question=question, status="running"))
        try:
            captured = []
            payload, _ = execute_query(
                planned["condition"],
                question,
                pipeline_factory,
                clock=clock,
                capture=captured.append,
            )
            elapsed = timer.elapsed_ms() / 1000
            if evidence_root:
                atomic_json(
                    Path(evidence_root) / (planned["attempt_id"] + "-response.json"),
                    payload,
                )
                if hasattr(captured[0], "model_dump"):
                    atomic_json(
                        Path(evidence_root)
                        / (planned["attempt_id"] + "-pipeline.json"),
                        captured[0].model_dump(mode="json"),
                    )
            return elapsed, True, None
        except Exception as exc:
            return None, False, type(exc).__name__

    return adapter


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--settings")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(run_cohort(args.output, dry_run=True), indent=2))
        return
    if not args.settings:
        parser.error("Real mode requires sealed --settings")
    from scripts.study_gate_environment import Environment, identity
    from scripts.study_gate_supervisor import Supervisor
    from src.ui.components.study_protocol import verify_draw
    import os

    config = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError("Cohort directory already exists")
    os.environ.update(
        CLOUDRAG_BUILD_ID=config["build_id"],
        CLOUDRAG_MODEL_DIGEST=config["model_digest"],
        CLOUDRAG_ARTIFACT_MANIFEST=config["artifact_manifest"],
        CLOUDRAG_DEMO_GPU=config.get("device_gpu", "0"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONHASHSEED="42",
        OLLAMA_HOST="http://127.0.0.1:11434",
    )
    if os.name == "nt":
        from scripts.gate_job import enter

        enter()
    protocol = verify_draw(config["config_dir"])
    identity(config)  # Refuse invalid deployment before loading or warming models.
    tasks = protocol["config"]["tasks"]["T1"] + protocol["config"]["tasks"]["T2"]
    # Worker preparation evidence is separate, so cohort exclusive-create remains valid.
    worker_root = root.with_name(root.name + "-worker")
    worker_root.mkdir()
    config.update(root=str(worker_root), cohort_id=root.name)
    from filelock import FileLock

    with (
        FileLock(str(Path(config["session_root"]) / "_inference.lock"), timeout=0),
        Supervisor(config) as supervisor,
    ):

        def preflight_factory(window_root):
            env = Environment(
                config, window_root, allowed_pids=(supervisor.process.pid,)
            )
            supervisor.poll = env.sample
            return env.preflight

        result = run_cohort(
            root, adapter=supervisor, preflight_factory=preflight_factory, tasks=tasks
        )
    seal(worker_root)
    result["worker_manifest_sha256"] = digest(worker_root / "manifest.json")
    result["settings_sha256"] = digest(args.settings)
    atomic_json(root / "aggregate.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
