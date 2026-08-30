"""exp21 — is a hosted granite equivalent to the local one? (deployment gate, pre-registered)

Seccion de Claude Code — 2026-08-21 08:00 (hora local). Harness only: NOT RUN, and it cannot be
run by accident (the endpoint is required and has no default).

WHY THIS GATE EXISTS. The SUS/Likert survey cannot run on the local box: TTFT is 12.8-15.2 s
because the prefill crosses the CPU/GPU boundary 114 times with only 30 of 41 layers on the GPU.
Renting a 4090 puts all 41 layers on the GPU. But changing where the model runs is changing the
system the participants evaluate, so it needs a gate: `docs/CLOUD_DEPLOYMENT_SURVEY.md` pre-registers
equivalence within +/- 0.081 on all three verifiers, or the survey stays local at k=5.

NOT BIT-IDENTITY. Local runs 30/41 layers, hosted runs 41/41; the arithmetic differs and the
answers will differ. The criterion is DECLARED EQUIVALENCE (TOST), never equality. A run that
demanded byte-identical answers would fail for a reason that has nothing to do with the survey.

SECURITY: the endpoint comes from the environment and from nowhere else.
    EXP21_OLLAMA_HOST     required, e.g. http://<host>:<port>   -- no default, ever
    EXP21_OLLAMA_TOKEN    optional bearer token
Neither is written to any artifact, log line or error message. The host is recorded in results.json
only as a SHA-256 prefix, so a run is auditable without publishing where it ran.

WHAT IT DOES
  1. Digest gate, BEFORE generating. `/api/show` on both sides; the hosted model's digest,
     quantization and parameter count must match the local one. A different weight file makes
     every downstream number meaningless, and it is the cheapest thing to check.
  2. 194 queries over exp18's FROZEN baseline contexts (retrieval_ids.json, read-only), with the
     canonical prompt path (rgm.build_prompt) -- identical prompts on both sides by construction.
  3. 3x determinism probe per arm (H5 pattern, relaxed gate: warn, never abort, as exp18).
  4. Only JSON crosses the wire. Scoring happens LOCALLY afterwards with the three verifiers, so
     no verifier ever runs on rented hardware and the instrument stays fixed.

WHICH LOCAL BASELINE -- READ BEFORE RUNNING. The pre-registration says "TOST against the local
granite" without saying whether that means exp18's stored answers or a fresh local run. On
2026-08-21 that stopped being a detail: regenerating exp18's own baseline locally reproduced its
prompts byte for byte (tokens_in identical on 3/3) and still produced different answers
(jaccard-5gram 0.086 / 0.204) -- cross-session runtime drift, ledger entry 25. Pairing a hosted run
against exp18's stored answers would therefore measure hosting PLUS drift and call the sum
"hosting". So this harness defaults to generating the local arm in the SAME window
(`--local-source fresh`); `--local-source exp18` is available and prints a warning. Enzo should
fix which one the pre-registration means before authorising the spend.

Usage:
  EXP21_OLLAMA_HOST=... python scripts/run_exp21_hosted_equivalence.py --stage local
  EXP21_OLLAMA_HOST=... python scripts/run_exp21_hosted_equivalence.py --stage hosted
  (then the local scoring pipeline, exactly as exp19b: pass N small/base, HHEM, arm_stats,
   guards, diagnosis, and compute_exp19b_stats.py --exp-dir <this dir> --arm hosted)
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""
import argparse
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import ensure_hashseed_at_startup, set_all_seeds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp21")

SEED = 42
CHECKPOINT_EVERY = 10
EXP_ID = "exp21_hosted_equivalence"
EXP_DIR = PROJECT_ROOT / "experiments/results" / EXP_ID
EXP18_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
CHUNK_MAP_PATH = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
MODEL_TAG = "granite4.1:8b"
LOCAL_HOST = "http://localhost:11434"

HOST_ENV = "EXP21_OLLAMA_HOST"
TOKEN_ENV = "EXP21_OLLAMA_TOKEN"

# Anchor first: verify_summer_offline.py finds the anchor as the arm named `baseline*`, and
# compute_exp18_diagnosis.py looks it up by name.
ARM_FOR_STAGE = {"local": "baseline_local", "hosted": "hosted"}

# Fields of /api/show that must agree. `digest` is the weight file; the other two catch a model
# that was re-quantised or re-exported under the same tag.
DIGEST_FIELDS = ["digest", "quantization_level", "parameter_size"]

_spec = importlib.util.spec_from_file_location(
    "rgm", PROJECT_ROOT / "scripts/run_generation_matrix.py")
rgm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgm)


def host_fingerprint(host):
    """SHA-256 prefix of the endpoint: auditable, and it publishes nothing."""
    return hashlib.sha256((host or "").encode("utf-8")).hexdigest()[:16]


class OllamaEndpoint:
    """Minimal Ollama client. Injectable so the tests can drive a mock instead of a network."""

    def __init__(self, host, token=None, timeout=600):
        self.host = host.rstrip("/")
        self._token = token
        self.timeout = timeout

    def _headers(self):
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def show(self, model):
        import requests
        r = requests.post(f"{self.host}/api/show", json={"model": model},
                          headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json()

    def chat(self, model, prompt, system_prompt, seed, temperature=0.0):
        import requests
        payload = {"model": model, "stream": False,
                   "options": {"seed": seed, "temperature": temperature},
                   "messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}]}
        r = requests.post(f"{self.host}/api/chat", json=payload,
                          headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()


def model_identity(show_payload):
    """The subset of /api/show that identifies the weights, flattened."""
    details = show_payload.get("details") or {}
    merged = {**show_payload, **details}
    return {f: merged.get(f) for f in DIGEST_FIELDS}


def digest_gate(local_show, hosted_show):
    """(ok, report). Compared BEFORE generating: different weights make everything meaningless."""
    a, b = model_identity(local_show), model_identity(hosted_show)
    mismatched = [f for f in DIGEST_FIELDS if a.get(f) != b.get(f)]
    return (not mismatched), {
        "local": a, "hosted": b, "mismatched_fields": mismatched,
        "passed": not mismatched,
        "why": ("the hosted model must be the same weight file at the same quantization; a "
                "different one makes every equivalence number below meaningless, and this is "
                "the cheapest possible check"),
    }


def answer_of(chat_payload):
    return ((chat_payload or {}).get("message") or {}).get("content") or ""


def resolve_endpoint(stage, args):
    """The hosted endpoint comes from the environment and nowhere else."""
    if stage == "local":
        return OllamaEndpoint(LOCAL_HOST), LOCAL_HOST
    host = os.environ.get(HOST_ENV)
    if not host:
        sys.exit(f"{HOST_ENV} is not set. This harness has NO default endpoint on purpose: a "
                 f"hardcoded host is how a URL or a token ends up committed. Export "
                 f"{HOST_ENV} (and {TOKEN_ENV} if the provider needs one) and re-run.")
    return OllamaEndpoint(host, os.environ.get(TOKEN_ENV), timeout=args.timeout), host


def main(endpoint_factory=resolve_endpoint):
    ensure_hashseed_at_startup(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["local", "hosted"])
    ap.add_argument("--local-source", default="fresh", choices=["fresh", "exp18"],
                    help="fresh: generate the local arm now (default; avoids the cross-session "
                         "drift measured on 2026-08-21). exp18: reuse exp18's stored answers.")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--skip-digest-gate", action="store_true",
                    help="record the mismatch and continue anyway; say so in the ledger")
    args = ap.parse_args()
    set_all_seeds(SEED)

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    arm = ARM_FOR_STAGE[args.stage]
    ep, host = endpoint_factory(args.stage, args)

    if args.stage == "local" and args.local_source == "exp18":
        logger.warning("--local-source exp18 pairs the hosted run against answers generated in "
                       "ANOTHER session. Cross-session drift was measured at jaccard-5gram "
                       "0.086-0.204 on 2026-08-21 (ledger 25); the contrast would then mix "
                       "hosting with drift.")

    ids_doc = json.loads((EXP18_DIR / "retrieval_ids.json").read_text(encoding="utf-8"))["ids"]
    qids = list(ids_doc)[: args.max_queries] if args.max_queries else list(ids_doc)
    questions = {q: ids_doc[q]["question"] for q in qids}
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))

    class _Index:
        def get_chunk(self, cid):
            return chunk_map.get(cid)

    index = _Index()
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import prompt_templates as PT
    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT", "build_context", "get_template")}
    qp = QueryProcessor()
    qtype = {q: qp.process(questions[q]).query_type for q in qids}

    gate = None
    if args.stage == "hosted":
        local_ep, _ = endpoint_factory("local", args)
        gate_ok, gate = digest_gate(local_ep.show(MODEL_TAG), ep.show(MODEL_TAG))
        logger.info("digest gate: %s (fields checked: %s)", "PASS" if gate_ok else "FAIL",
                    DIGEST_FIELDS)
        if not gate_ok and not args.skip_digest_gate:
            (EXP_DIR / "digest_gate.json").write_text(
                json.dumps(gate, indent=1), encoding="utf-8")
            sys.exit(f"DIGEST GATE FAILED on {gate['mismatched_fields']} — the hosted model is "
                     f"not the local one. Refusing to generate; report written.")
        (EXP_DIR / "digest_gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")

    label = rgm.model_label(MODEL_TAG)
    q0 = qids[0]
    pr0, sp0, _ = rgm.build_prompt("hibrido", questions[q0],
                                   ids_doc[q0]["baseline_repro_ids"], index, qtype[q0], P)
    outs = [answer_of(ep.chat(MODEL_TAG, pr0, sp0, SEED)) for _ in range(3)]
    det_ok = all(o == outs[0] for o in outs)
    probe = {arm: {"determinism_3x_identical": det_ok, "answer_lens_3x": [len(o) for o in outs]}}
    logger.info("[%s] probe: determinism=%s", arm, det_ok)
    if not det_ok:
        logger.warning("[%s] determinism probe NOT bit-identical (relaxed gate, as exp18)", arm)

    config_name = f"{arm} | {label}"
    cpath = EXP_DIR / f"checkpoint__{label}__{arm}.json"
    results, done = [], set()
    if not args.no_resume and cpath.exists():
        ck = json.loads(cpath.read_text(encoding="utf-8"))
        results, done = ck["results"], set(ck["completed_ids"])
        logger.info("[%s] resume: %d done", config_name, len(done))

    todo = [q for q in qids if q not in done]
    for i, qid in enumerate(todo):
        ctx_ids = ids_doc[qid]["baseline_repro_ids"]
        prompt, sysp, _ = rgm.build_prompt("hibrido", questions[qid], ctx_ids,
                                           index, qtype[qid], P)
        t = time.perf_counter()
        payload = ep.chat(MODEL_TAG, prompt, sysp, SEED)
        results.append({
            "query_id": qid, "config_name": config_name, "scenario": arm, "model": MODEL_TAG,
            "question": questions[qid], "answer": answer_of(payload),
            "retrieved_ids": ctx_ids, "query_type": qtype[qid],
            "hallucination_metrics": {"method": "pending_nli"},
            "tokens": {"input": payload.get("prompt_eval_count"),
                       "output": payload.get("eval_count")},
            "latency": {"generation_ms": round((time.perf_counter() - t) * 1000, 1)},
            "from_cache": False, "error": None,
            "timestamp": datetime.now().isoformat(),
        })
        done.add(qid)
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(todo):
            cpath.write_text(json.dumps(
                {"config_name": config_name, "completed_ids": sorted(done),
                 "results": results}, ensure_ascii=False), encoding="utf-8")
            logger.info("[%s] %d/%d", config_name, len(done), len(qids))

    write_results(EXP_DIR, label, probe, qids, gate, host, args)


def write_results(exp_dir, label, probe, qids, gate, host, args):
    configs = {}
    for arm in ARM_FOR_STAGE.values():
        cpath = exp_dir / f"checkpoint__{label}__{arm}.json"
        if not cpath.exists():
            continue
        ck = json.loads(cpath.read_text(encoding="utf-8"))
        rs = ck["results"]
        configs[ck["config_name"]] = {
            "total_queries": len(rs), "errors": sum(1 for r in rs if r.get("error")),
            "scenario": arm, "model": MODEL_TAG, "results": rs}

    rj = exp_dir / "results.json"
    prior = {}
    if rj.exists():
        try:
            prior = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            prior = {}
    doc = {
        "experiment_id": EXP_ID,
        "name": "exp21 hosted-vs-local equivalence gate for the survey deployment",
        "timestamp": datetime.now().isoformat(), "seed": SEED, "temperature": 0.0,
        "context_source": "exp18 retrieval_ids.json baseline top-5 (frozen, read-only)",
        "num_queries": len(qids), "model": MODEL_TAG,
        "probe_report": {**prior.get("probe_report", {}), **probe},
        "digest_gate": gate or prior.get("digest_gate"),
        # The endpoint itself is never recorded. The fingerprint makes two runs comparable
        # without publishing where they ran.
        "endpoint_fingerprint": {ARM_FOR_STAGE[args.stage]: host_fingerprint(host)},
        "local_source": args.local_source,
        "equivalence_note": ("NOT bit-identity: local runs 30/41 layers on GPU, hosted 41/41. The "
                             "pre-registered criterion is TOST equivalence within +/-0.081 on all "
                             "three verifiers, with the HHEM anchor inside 0.40-0.55."),
        "bh_family_note": ("1 hosted-vs-baseline_local contrast per verifier; a family of one "
                           "means BH is the identity and p_BH == p_raw, declared not implied."),
        "scoring_note": ("only JSON crosses the wire; NLI-small, NLI-base and HHEM all run "
                         "locally afterwards, so no verifier ever executes on rented hardware"),
        "configs": configs,
    }
    rj.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", rj)
    return doc


if __name__ == "__main__":
    main()
