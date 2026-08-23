"""exp19b — the whole run as one supervised pipeline, for a terminal that stays open.

Seccion de Claude Code — 2026-08-21 14:05 (hora local).

WHY THIS EXISTS. The run is ~6.6 h of generation and it MUST live inside one warmed Ollama
session: measured on 2026-08-21, restarting the server changes the answer to a byte-identical
prompt (q001 vs the checkpoint: jaccard-5gram 0.0705), so a run split across a restart puts two
generator states inside one arm and the difference gets read as a selector effect. Four
consecutive background processes were killed in the agent environment, so the run has to be
started from a terminal a human keeps open, and it has to supervise itself.

WHAT IT GUARDS, and why that is the whole point. On 2026-08-22 the warmup fingerprint was
replaced as a gate: it distinguished three Ollama load states, yet two full drafts separated by
4.7 hours and load-state changes produced 194/194 bit-identical answers. The proxy therefore had
structural false positives. Immediately before `regen`, the pipeline now replays five archived
draft queries through the real draft prompt/generation path and requires 5/5 byte identity. The
warmup fingerprint remains in the log as diagnostic data, never as a decision rule.

The stage list is data, not prose, so tests can assert its order and that the gate sits where it
has to sit. The subprocess runner is injected for the same reason.

RESUMPTION. Without `--start-from`, every full run starts at a fresh draft (`--no-cache` and
`--no-resume`). `--start-from` is only for an explicitly resumed run whose prior artifacts
already exist; starting at regen still re-runs the mandatory direct replay gate first.

Usage (normally via scripts/launch_exp19b_full.ps1):
  python scripts/run_exp19b_pipeline.py [--dry-run] [--max-queries N] [--start-from STAGE]
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 PYTHONUTF8=1
Exit: 0 all stages passed · 2 a stage failed · 3 RUNTIME_STATE_CHANGED
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EXP_ID = "exp19b_anchored_selector"
EXP_DIR = PROJECT_ROOT / "experiments/results" / EXP_ID
LOG_DIR = PROJECT_ROOT / "logs"
VERIFIERS = ["small", "base", "hhem"]
DRAFT_REPLAY_N = 5
RESUME_ARTIFACTS = ("results.json", "draft_claims.json", "selection_ids.json")

EXIT_OK, EXIT_STAGE_FAILED, EXIT_STATE_CHANGED = 0, 2, 3
STATE_CHANGED_MARKER = "RUNTIME_STATE_CHANGED"

_spec = importlib.util.spec_from_file_location(
    "exp19b_gen", PROJECT_ROOT / "scripts/run_exp19b_generation.py")
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
session_fingerprint = _gen.session_fingerprint


class StageCommand(list):
    """List-compatible argv carrying environment overrides for the real subprocess runner."""

    def __init__(self, argv, env=None):
        super().__init__(argv)
        self.env = dict(env or {})


def build_stages(py=None, exp_dir=None, max_queries=None):
    """Build the stage contract.

    Invariant: draft -> [extract, select on CPU] -> direct draft replay gate -> regen; the GPU is
    untouched until regen finishes. Order and per-stage environment are both testable data.
    """
    py = py or sys.executable
    exp_dir = str(exp_dir or EXP_DIR)
    mq = ["--max-queries", str(max_queries)] if max_queries else []
    S = str(PROJECT_ROOT / "scripts")

    stages = [
        # --no-resume is not optional for either derived stage: a fresh draft invalidates every
        # claim-conditioned selection checkpoint, even if that checkpoint is otherwise intact.
        ("draft", [py, f"{S}/run_exp19b_generation.py", "--stage", "draft", "--no-cache",
                   "--no-resume", *mq]),
        ("extract", StageCommand(
            [py, f"{S}/extract_exp19b_claims.py"], env={"CUDA_VISIBLE_DEVICES": ""})),
        ("select", StageCommand(
            [py, f"{S}/select_exp19b_evidence.py", "--no-resume", *mq],
            env={"CUDA_VISIBLE_DEVICES": ""})),
        # THE GATE. Replay the first five archived draft qids; a mismatch blocks the second arm.
        ("draft_replay_check", None),
        ("regen", [py, f"{S}/run_exp19b_generation.py", "--stage", "regen", "--no-cache", *mq]),
    ]
    stages += [(f"pass_n_{v}", [py, f"{S}/run_exp15_ablation.py", "--exp-id", EXP_ID,
                                "--pass", "N", "--verifier", v]) for v in ("small", "base")]
    stages.append(("grounding_hhem", [py, f"{S}/rescore_grounding_tierA.py",
                                      "--exp-dir", exp_dir]))
    stages += [(f"arm_stats_{v}", [py, f"{S}/compute_tierA_arm_stats.py", "--exp-dir", exp_dir,
                                   "--verifier", v]) for v in VERIFIERS]
    stages += [(f"diagnosis_{v}", [py, f"{S}/compute_exp18_diagnosis.py", "--exp-dir", exp_dir,
                                   "--verifier", v]) for v in VERIFIERS]
    stages.append(("guards", [py, f"{S}/compute_exp16_guards.py", "--exp-dir", exp_dir]))
    stages.append(("primary_tost", [py, f"{S}/compute_exp19b_stats.py", "--exp-dir", exp_dir]))
    stages.append(("verify_offline", [py, f"{S}/verify_summer_offline.py"]))
    return stages


def gate_index(stages):
    return [n for n, _ in stages].index("draft_replay_check")


def stages_from(stages, start_from=None, exp_dir=EXP_DIR):
    """Slice an explicitly resumed run while preserving the mandatory pre-regen replay gate."""
    if start_from is None:
        return stages
    names = [name for name, _command in stages]
    if start_from not in names:
        raise ValueError(f"unknown --start-from stage {start_from!r}; choose from {names}")

    requested_index = names.index(start_from)
    replay_index = names.index("draft_replay_check")
    if requested_index >= replay_index:
        exp_dir = Path(exp_dir)
        missing = [name for name in RESUME_ARTIFACTS if not (exp_dir / name).exists()]
        if missing:
            raise ValueError(
                f"cannot --start-from {start_from}: missing required artifacts: "
                + ", ".join(missing))

    # Regen may never bypass the direct replay gate. Asking to resume at regen means that the
    # earlier draft/extract/select stages are complete, then replay is rechecked before regen.
    if start_from == "regen":
        requested_index = replay_index
    return stages[requested_index:]


def draft_replay_check(archived_rows, generate_fn, n=DRAFT_REPLAY_N):
    """Compare the first fixed checkpoint rows with freshly generated draft answers."""
    rows = list(archived_rows)[:n]
    if len(rows) != n:
        detail = {"checked": len(rows), "identical": 0, "mismatched_qids": [],
                  "why": f"checkpoint has fewer than the required {n} replay rows"}
        return False, (f"{STATE_CHANGED_MARKER}: draft replay unavailable "
                       f"({len(rows)}/{n}); nothing was scored"), detail

    mismatched = []
    for row in rows:
        qid = row["query_id"]
        if generate_fn(qid) != (row.get("answer") or ""):
            mismatched.append(qid)
    identical = n - len(mismatched)
    detail = {"checked": n, "identical": identical, "mismatched_qids": mismatched}
    if mismatched:
        return False, (f"{STATE_CHANGED_MARKER}: draft replay {identical}/{n} bit-identical; "
                       f"mismatches={mismatched}; nothing was scored"), detail
    return True, f"draft replay {n}/{n} bit-identical", detail


def capture_fingerprint(generate_fn):
    """Fingerprint = hash of a warmup answer. Injected generator keeps this testable."""
    return session_fingerprint(generate_fn())


def _draft_runtime(exp_dir=EXP_DIR):
    """Load the draft plan and prompt machinery directly from the generation runner."""
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import prompt_templates as PT

    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT", "build_context",
          "get_template")}
    args = type("DraftReplayArgs", (), {"max_queries": None})()
    qids, context, questions, ids_doc = _gen._load_plan("draft", Path(exp_dir), args)
    chunk_map = json.loads(_gen.CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    index = _gen.ChunkMapIndex(chunk_map)
    processor = QueryProcessor()
    query_types = {qid: processor.process(questions[qid]).query_type for qid in qids}
    drift = [qid for qid in qids
             if ids_doc[qid].get("routing_query_type") not in (None, query_types[qid])]
    if drift:
        raise RuntimeError(f"query_type drift before draft replay: {drift[:3]}")
    return qids, context, questions, query_types, index, P


def _draft_prompt(runtime, qid):
    _qids, context, questions, query_types, index, templates = runtime
    return _gen.rgm.build_prompt(
        "hibrido", questions[qid], context[qid], index, query_types[qid], templates)[:2]


def _live_warmup():
    """One informational warmup generation on the real draft path; never a gate."""
    from src.generation.llm_manager import LLMManager

    runtime = _draft_runtime()
    q0 = runtime[0][0]
    pr, sp = _draft_prompt(runtime, q0)
    llm = LLMManager(provider="ollama", model=_gen.MODEL_TAG, cache_enabled=False, seed=42)
    return llm.generate(prompt=pr, system_prompt=sp, temperature=0.0,
                        config_name="pipeline_fingerprint").text


def _live_draft_replay(exp_dir=EXP_DIR):
    """Replay the fixed first five archived draft rows through the real draft path."""
    from src.generation.llm_manager import LLMManager

    label = _gen.rgm.model_label(_gen.MODEL_TAG)
    checkpoint = Path(exp_dir) / f"checkpoint__{label}__baseline_repro.json"
    archived_rows = json.loads(checkpoint.read_text(encoding="utf-8"))["results"]
    runtime = _draft_runtime(exp_dir)
    llm = LLMManager(provider="ollama", model=_gen.MODEL_TAG,
                     cache_enabled=False, seed=_gen.SEED)

    def generate(qid):
        prompt, system_prompt = _draft_prompt(runtime, qid)
        return llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.0,
                            config_name="draft_replay_check").text

    return draft_replay_check(archived_rows, generate)


def run_pipeline(stages, runner=None, fingerprint_fn=None, replay_check_fn=None, log=print):
    """Execute stages in order, stopping at the first failure. Returns (exit_code, report)."""
    def subprocess_runner(argv):
        env = os.environ.copy()
        env.update(getattr(argv, "env", {}))
        return subprocess.run(list(argv), env=env).returncode

    runner = runner or subprocess_runner
    fingerprint_fn = fingerprint_fn or (lambda: capture_fingerprint(_live_warmup))
    replay_check_fn = replay_check_fn or _live_draft_replay

    report = []
    for name, argv in stages:
        t0 = datetime.now()
        if name == "draft_replay_check":
            fingerprint = fingerprint_fn()
            log(f"[info] warmup fingerprint before replay (not a gate): {fingerprint}")
            ok, msg, detail = replay_check_fn()
            log(f"[gate] {msg}")
            report.append({"stage": name, "ok": ok, "detail": msg, "replay": detail,
                           "warmup_fingerprint_info_only": fingerprint})
            if not ok:
                return EXIT_STATE_CHANGED, report
            continue

        log(f"[stage] {name} starting {t0:%H:%M:%S}")
        rc = runner(argv)
        secs = (datetime.now() - t0).total_seconds()
        report.append({"stage": name, "ok": rc == 0, "returncode": rc, "seconds": round(secs)})
        log(f"[stage] {name} -> rc={rc} in {secs / 60:.1f} min")
        if rc != 0:
            log(f"[abort] {name} failed with rc={rc}; nothing after it ran")
            return EXIT_STAGE_FAILED, report

        # Retained as informative diagnostics only. Direct answer replay is the actual gate.
        if name == "draft":
            fingerprint = fingerprint_fn()
            log(f"[info] warmup fingerprint after draft (not a gate): {fingerprint}")
    return EXIT_OK, report


def render_summary(code, report):
    lines = ["", "=" * 68, "exp19b pipeline summary", "=" * 68]
    for r in report:
        mark = "OK  " if r.get("ok") else "FAIL"
        extra = f"{r['seconds'] // 60} min" if "seconds" in r else r.get("detail", "")
        lines.append(f"  [{mark}] {r['stage']:<22} {extra}")
    verdict = {EXIT_OK: "ALL STAGES PASSED",
               EXIT_STAGE_FAILED: "STOPPED: a stage failed",
               EXIT_STATE_CHANGED: f"STOPPED: {STATE_CHANGED_MARKER} — nothing was scored"}
    lines += ["-" * 68, verdict.get(code, f"exit {code}"), "=" * 68, ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the stages and exit 0")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--start-from", default=None,
                    help="explicitly resume at STAGE; completed artifacts must already exist")
    args = ap.parse_args()

    stages = build_stages(max_queries=args.max_queries)
    try:
        stages = stages_from(stages, args.start_from, EXP_DIR)
    except ValueError as exc:
        ap.error(str(exc))
    if args.dry_run:
        for i, (name, argv) in enumerate(stages):
            print(f"{i:>2}. {name:<22} {'<direct draft replay gate>' if argv is None else ' '.join(argv[1:])}")
        return EXIT_OK

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"exp19b_full_{date.today().isoformat()}.log"
    handle = log_path.open("a", encoding="utf-8")

    def log(msg):
        stamped = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(stamped, flush=True)
        handle.write(stamped + "\n")
        handle.flush()

    log(f"exp19b pipeline start — log {log_path}")
    code, report = run_pipeline(stages, log=log)
    summary = render_summary(code, report)
    print(summary)
    handle.write(summary)
    handle.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
