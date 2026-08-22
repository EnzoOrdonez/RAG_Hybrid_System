"""exp19b — the whole run as one supervised pipeline, for a terminal that stays open.

Seccion de Claude Code — 2026-08-21 14:05 (hora local).

WHY THIS EXISTS. The run is ~6.6 h of generation and it MUST live inside one warmed Ollama
session: measured on 2026-08-21, restarting the server changes the answer to a byte-identical
prompt (q001 vs the checkpoint: jaccard-5gram 0.0705), so a run split across a restart puts two
generator states inside one arm and the difference gets read as a selector effect. Four
consecutive background processes were killed in the agent environment, so the run has to be
started from a terminal a human keeps open, and it has to supervise itself.

WHAT IT GUARDS, and why that is the whole point. Between `draft` and `regen` there is a CPU gap
(claim extraction, then ~40 min of cross-encoder re-ranking). That gap is exactly where a
machine sleeps, a driver resets, or someone restarts Ollama. So the session fingerprint is taken
at the start and RE-TAKEN immediately before `regen`. If it moved, the pipeline stops with
RUNTIME_STATE_CHANGED and **scores nothing**: a pipeline that scores two arms generated in
different generator states produces a number that looks valid and is not.

The stage list is data, not prose, so tests can assert its order and that the gate sits where it
has to sit. The subprocess runner is injected for the same reason.

Usage (normally via scripts/launch_exp19b_full.ps1):
  python scripts/run_exp19b_pipeline.py [--dry-run] [--max-queries N]
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

    Invariant: draft -> [extract, select on CPU] -> fingerprint gate -> regen; the GPU is
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
        # THE GATE. Everything above ran before a long CPU stretch; everything below writes the
        # second arm. If the generator moved in between, nothing below may run.
        ("fingerprint_recheck", None),
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
    return [n for n, _ in stages].index("fingerprint_recheck")


def fingerprint_gate(before, after):
    """(ok, message). Pure, so the rule is testable without a server."""
    if before == after:
        return True, f"generator state unchanged ({before})"
    return False, (f"{STATE_CHANGED_MARKER}: {before} -> {after}. The generator moved between "
                   f"the draft and the regeneration, so the two arms would come from different "
                   f"states and the difference would be read as a selector effect. Nothing was "
                   f"scored. Re-run the whole pipeline in one uninterrupted session.")


def capture_fingerprint(generate_fn):
    """Fingerprint = hash of a warmup answer. Injected generator keeps this testable."""
    return session_fingerprint(generate_fn())


def _live_warmup():
    """One warmup generation on the first query's real prompt, exactly as the runner does."""
    from src.generation.llm_manager import LLMManager
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import prompt_templates as PT
    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT", "build_context",
          "get_template")}
    ids = json.loads((PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
                      / "retrieval_ids.json").read_text(encoding="utf-8"))["ids"]
    chunk_map = json.loads((PROJECT_ROOT / "data/indices"
                            / "chunk_map_bge-large_adaptive_500.json").read_text(
                                encoding="utf-8"))
    q0 = next(iter(ids))
    qtype = QueryProcessor().process(ids[q0]["question"]).query_type
    pr, sp, _ = _gen.rgm.build_prompt("hibrido", ids[q0]["question"],
                                      ids[q0]["baseline_repro_ids"],
                                      _gen.ChunkMapIndex(chunk_map), qtype, P)
    llm = LLMManager(provider="ollama", model=_gen.MODEL_TAG, cache_enabled=False, seed=42)
    return llm.generate(prompt=pr, system_prompt=sp, temperature=0.0,
                        config_name="pipeline_fingerprint").text


def run_pipeline(stages, runner=None, fingerprint_fn=None, log=print):
    """Execute stages in order, stopping at the first failure. Returns (exit_code, report)."""
    def subprocess_runner(argv):
        env = os.environ.copy()
        env.update(getattr(argv, "env", {}))
        return subprocess.run(list(argv), env=env).returncode

    runner = runner or subprocess_runner
    fingerprint_fn = fingerprint_fn or (lambda: capture_fingerprint(_live_warmup))

    report, fp_start = [], None
    for name, argv in stages:
        t0 = datetime.now()
        if name == "fingerprint_recheck":
            fp_now = fingerprint_fn()
            ok, msg = fingerprint_gate(fp_start, fp_now)
            log(f"[gate] {msg}")
            report.append({"stage": name, "ok": ok, "detail": msg})
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

        # The starting fingerprint is taken right after the draft: the draft's own warmup has
        # already put the server in the state the arm was generated in.
        if name == "draft":
            fp_start = fingerprint_fn()
            log(f"[gate] session fingerprint after draft: {fp_start}")
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
    args = ap.parse_args()

    stages = build_stages(max_queries=args.max_queries)
    if args.dry_run:
        for i, (name, argv) in enumerate(stages):
            print(f"{i:>2}. {name:<22} {'<fingerprint gate>' if argv is None else ' '.join(argv[1:])}")
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
