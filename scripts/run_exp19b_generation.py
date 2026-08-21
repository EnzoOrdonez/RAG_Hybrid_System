"""exp19b — generation for the anchoring-guided selector (draft arm + regenerated arm).

Seccion de Claude Code — 2026-08-21 14:20 (hora local).

THE EXPERIMENT. exp18 established that the k=50 pool supports 58.3 % of the claims the
baseline actually wrote while the 5 chunks picked by topical relevance support only 45.5 %,
and exp19a showed (offline, zero generation) that re-ranking the pool by (claim, chunk)
recovers 23.4 % of that margin without any faithfulness verifier in the loop. exp19b is the
generative arm that turns the mechanism into an answer:

    stage draft   generate with exp18's baseline top-5           -> arm `baseline_repro`
    stage X       extract the draft's claims                     (extract_exp19b_claims.py)
    stage S       re-rank the pool by (claim, chunk), take 5     (select_exp19b_evidence.py)
    stage regen   generate again from that new top-5             -> arm `claim_selected`

WHY THE DRAFT IS ALSO THE BASELINE. The pre-registration pairs the treatment against exp18's
baseline arm within-session. Generating the draft here rather than reading exp18's stored
answers makes that pairing true by construction: both arms come out of the same Ollama
session, the same warmup, the same seed, with the cache off. exp18's stored baseline answer
is still compared byte-for-byte and REPORTED (`draft_vs_exp18_identical`), but it is a sanity
check on the environment, never evidence and never a gate -- exp18 itself only ran a relaxed
determinism gate, so a mismatch is information, not a failure.

WHAT IS FORBIDDEN HERE. Nothing under experiments/results/exp3..exp19a is opened for writing.
This runner reads exp18's `retrieval_ids.json` and its baseline checkpoint, and writes only
under experiments/results/exp19b_anchored_selector/.

CACHE. Both stages run with --no-cache, so no answer can be served from an earlier session
(H5 cross-session drift). The per-query checkpoints still resume, which is the only kind of
reuse this design allows.

Usage:
  python scripts/run_exp19b_generation.py --stage draft --no-cache [--max-queries N] [--smoke]
  python scripts/run_exp19b_generation.py --stage regen --no-cache [--max-queries N] [--smoke]
Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""
import argparse
import importlib.util
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import ensure_hashseed_at_startup, set_all_seeds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp19b")

SEED = 42
CHECKPOINT_EVERY = 10
EXP_ID = "exp19b_anchored_selector"
EXP_DIR = PROJECT_ROOT / "experiments/results" / EXP_ID
EXP18_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
CHUNK_MAP_PATH = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
MODEL_TAG = "granite4.1:8b"

# The anchor arm is called `baseline_repro` on purpose: verify_summer_offline.py discovers the
# anchor as the arm named `baseline*`, and compute_exp18_diagnosis.py hardcodes that name.
# Renaming it would leave exp19b silently unverified.
ARM_FOR_STAGE = {"draft": "baseline_repro", "regen": "claim_selected"}

_spec = importlib.util.spec_from_file_location(
    "rgm", PROJECT_ROOT / "scripts/run_generation_matrix.py")
rgm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgm)


class ChunkMapIndex:
    """Same shim exp18 uses: prompt-identical chunk dicts without loading FAISS."""

    def __init__(self, chunk_map):
        self._m = chunk_map

    def get_chunk(self, cid):
        return self._m.get(cid)


def out_dir(smoke=False):
    """Smoke artifacts live INSIDE the experiment dir, in `_smoke/`.

    tests/test_scored_arms_complete.py and verify_summer_offline.py both discover experiments
    by walking experiments/results/ one level deep and matching artifact SHAPE. A 3-query
    smoke has no faithfulness rows, so writing it as a sibling directory would turn the suite
    red for a reason nobody is allowed to fix -- and a test that is red for untouchable
    reasons is a test that gets loosened.
    """
    return (EXP_DIR / "_smoke") if smoke else EXP_DIR


def build_results_doc(exp_dir, label, probe_report, qids, model_tag=MODEL_TAG):
    """Fold the per-arm checkpoints into the standard arm-schema results.json.

    `scenario` on every config is what makes the existing scorers work unchanged via
    --exp-dir / --exp-id: run_exp15_ablation.py --pass N, rescore_grounding_tierA.py,
    compute_tierA_arm_stats.py, compute_exp16_guards.py, compute_exp18_diagnosis.py.
    """
    exp_dir = Path(exp_dir)
    configs = {}
    for arm in ARM_FOR_STAGE.values():
        cpath = exp_dir / f"checkpoint__{label}__{arm}.json"
        if not cpath.exists():
            continue
        ck = json.loads(cpath.read_text(encoding="utf-8"))
        rs = ck["results"]
        configs[ck["config_name"]] = {
            "total_queries": len(rs), "errors": sum(1 for r in rs if r.get("error")),
            "scenario": arm, "model": model_tag, "results": rs}
    trunc = {c["scenario"]: {
        "n_at_context_limit": len([r["query_id"] for r in c["results"]
                                   if (r.get("tokens") or {}).get("input", 0) >= 4096]),
        "qids": [r["query_id"] for r in c["results"]
                 if (r.get("tokens") or {}).get("input", 0) >= 4096]}
        for c in configs.values()}
    return {
        "experiment_id": EXP_ID,
        "name": "exp19b anchoring-guided selector (draft -> claims -> claim rerank -> regen)",
        "timestamp": datetime.now().isoformat(), "seed": SEED, "temperature": 0.0,
        "context_source": ("baseline_repro = exp18 retrieval_ids.json baseline top-5; "
                           "claim_selected = exp19b selection_ids.json (claim-conditioned "
                           "rerank of the same k=50 pool)"),
        "queries": "exp18 retrieval_ids.json (194 queries)",
        "num_queries": len(qids), "model": model_tag, "probe_report": probe_report,
        "observed_truncation": trunc,
        "all_arms_bit_deterministic": (
            all(p.get("determinism_3x_identical") for p in probe_report.values())
            if probe_report else None),
        "reproducibility_note": ("Both arms generated in the same session with the LLM cache "
                                 "off, so the paired claim_selected-vs-baseline_repro contrast "
                                 "is within-session. Single-sample generations; retrieval is "
                                 "deterministic and frozen from exp18's pool."),
        "bh_family_note": ("1 arm-vs-baseline_repro contrast (claim_selected) per verifier. "
                           "A family of one means BH is the identity and p_BH == p_raw; that "
                           "is declared, not implied. The three verifiers are triangulation, "
                           "not a family (phase standard since exp17)."),
        "configs": configs,
    }


def carry_forward(doc, prior_doc):
    """Keep the fields only one stage can produce when a later stage rewrites results.json.

    The smoke run caught this: `--stage regen` rebuilds results.json from the checkpoints, and
    `draft_vs_exp18_identical` is computed only by `--stage draft`, so the finished artifact
    came out with that field set to None. The sanity check had run and been logged -- it just
    stopped existing where anyone would later read it, which is the worst of both worlds.

    `probe_report` already merged across stages; this generalises the same rule.
    """
    for field in ("draft_vs_exp18_identical",):
        if doc.get(field) is None and prior_doc.get(field) is not None:
            doc[field] = prior_doc[field]
    return doc


def _load_plan(stage, exp_dir, args):
    """(qids, per-qid context ids, questions) for the stage. Read-only on exp18."""
    ids_doc = json.loads((EXP18_DIR / "retrieval_ids.json").read_text(encoding="utf-8"))["ids"]
    all_qids = list(ids_doc)
    questions = {q: ids_doc[q]["question"] for q in all_qids}
    if stage == "draft":
        ctx = {q: ids_doc[q]["baseline_repro_ids"] for q in all_qids}
        qids = all_qids
    else:
        sel_path = exp_dir / "selection_ids.json"
        if not sel_path.exists():
            sys.exit(f"{sel_path} missing — run extract_exp19b_claims.py and then "
                     f"select_exp19b_evidence.py before --stage regen")
        sel = json.loads(sel_path.read_text(encoding="utf-8"))["per_query"]
        ctx = {q: sel[q]["claim_rank_ids"] for q in sel}
        # Regenerate exactly the queries the selector produced a selection for, in exp18 order.
        qids = [q for q in all_qids if q in ctx]
    if args.max_queries:
        qids = qids[: args.max_queries]
    return qids, ctx, questions, ids_doc


def main():
    ensure_hashseed_at_startup(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["draft", "regen"])
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="write under <exp dir>/_smoke so a partial run is invisible to the "
                         "shape-based discovery in tests and verifiers")
    args = ap.parse_args()
    set_all_seeds(SEED)

    exp_dir = out_dir(smoke=args.smoke)
    exp_dir.mkdir(parents=True, exist_ok=True)
    arm = ARM_FOR_STAGE[args.stage]

    from src.generation.llm_manager import LLMManager
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import prompt_templates as PT
    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT",
          "build_context", "get_template")}

    qids, ctx, questions, ids_doc = _load_plan(args.stage, exp_dir, args)
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    index = ChunkMapIndex(chunk_map)
    label = rgm.model_label(MODEL_TAG)

    # query_type drives BOTH the prompt template and the context layout, so it is recomputed
    # with the same QueryProcessor exp18 used rather than read from the frozen file -- and then
    # cross-checked against the frozen value. A silent drift here would change the prompt while
    # every artifact still claimed the exp18 path.
    qp = QueryProcessor()
    qtype = {q: qp.process(questions[q]).query_type for q in qids}
    drift = [q for q in qids if ids_doc[q].get("routing_query_type") not in (None, qtype[q])]
    if drift:
        sys.exit(f"query_type drift vs exp18 on {len(drift)} queries (e.g. {drift[:3]}): the "
                 f"prompt path is no longer the one exp18 measured. Refusing to generate.")

    llm = LLMManager(provider="ollama", model=MODEL_TAG,
                     cache_enabled=not args.no_cache, seed=SEED)

    q0 = qids[0]
    pr0, sp0, _ = rgm.build_prompt("hibrido", questions[q0], ctx[q0], index, qtype[q0], P)
    llm_nc = LLMManager(provider="ollama", model=MODEL_TAG, cache_enabled=False, seed=SEED)
    llm_nc.generate(prompt=pr0, system_prompt=sp0, temperature=0.0, config_name="warmup")
    logger.info("warmup generation done")

    outs = [llm_nc.generate(prompt=pr0, system_prompt=sp0, temperature=0.0,
                            config_name=f"detprobe|{arm}|{label}").text for _ in range(3)]
    det_ok = all(o == outs[0] for o in outs)
    probe = {arm: {"determinism_3x_identical": det_ok, "answer_lens_3x": [len(o) for o in outs]}}
    logger.info("[%s] probe: determinism=%s", arm, det_ok)
    if not det_ok:
        logger.warning("[%s] determinism probe NOT bit-identical (relaxed gate, as exp18)", arm)

    config_name = f"{arm} | {label}"
    cpath = exp_dir / f"checkpoint__{label}__{arm}.json"
    results, done = [], set()
    if not args.no_resume and cpath.exists():
        ck = json.loads(cpath.read_text(encoding="utf-8"))
        results, done = ck["results"], set(ck["completed_ids"])
        logger.info("[%s] resume: %d done", config_name, len(done))
    todo = [q for q in qids if q not in done]

    for i, qid in enumerate(todo):
        prompt, sysp, _ = rgm.build_prompt("hibrido", questions[qid], ctx[qid],
                                           index, qtype[qid], P)
        t = time.perf_counter()
        resp = llm.generate(prompt=prompt, system_prompt=sysp, temperature=0.0,
                            config_name=config_name)
        gen_ms = resp.latency_ms or (time.perf_counter() - t) * 1000
        results.append({
            "query_id": qid, "config_name": config_name, "scenario": arm, "model": MODEL_TAG,
            "question": questions[qid], "answer": resp.text, "retrieved_ids": ctx[qid],
            "query_type": qtype[qid],
            "hallucination_metrics": {"method": "pending_nli"},
            "tokens": {"input": resp.tokens_input, "output": resp.tokens_output},
            "latency": {"generation_ms": round(gen_ms, 1)},
            "from_cache": resp.from_cache, "error": resp.error,
            "timestamp": datetime.now().isoformat(),
        })
        done.add(qid)
        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(todo):
            cpath.write_text(json.dumps(
                {"config_name": config_name, "completed_ids": sorted(done),
                 "results": results}, ensure_ascii=False), encoding="utf-8")
            logger.info("[%s] %d/%d", config_name, len(done), len(qids))

    rj = exp_dir / "results.json"
    prior_doc = {}
    if rj.exists():
        try:
            prior_doc = json.loads(rj.read_text(encoding="utf-8"))
        except Exception:
            prior_doc = {}
    doc = build_results_doc(exp_dir, label, {**prior_doc.get("probe_report", {}), **probe}, qids)

    if args.stage == "draft":
        doc["draft_vs_exp18_identical"] = _compare_with_exp18(results)
        logger.info("draft vs exp18 baseline: %s", doc["draft_vs_exp18_identical"])
    carry_forward(doc, prior_doc)

    rj.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", rj)


def _compare_with_exp18(results):
    """Byte-comparison of the fresh draft against exp18's stored baseline answer.

    REPORTED, never a gate. exp18 ran its own determinism probe under a relaxed gate, so a
    mismatch here says the environment moved, not that this run is invalid -- and pretending
    otherwise would make an environment hiccup look like an experimental result.
    """
    cpath = EXP18_DIR / "checkpoint__granite4.1-8b__baseline_repro.json"
    if not cpath.exists():
        return {"checked": 0, "note": "exp18 baseline checkpoint not found"}
    prior = {r["query_id"]: (r.get("answer") or "")
             for r in json.loads(cpath.read_text(encoding="utf-8"))["results"]}
    both = [r for r in results if r["query_id"] in prior]
    same = [r["query_id"] for r in both if (r.get("answer") or "") == prior[r["query_id"]]]
    return {"checked": len(both), "identical": len(same),
            "rate": round(len(same) / len(both), 4) if both else None,
            "note": ("sanity check on the environment, NOT evidence and NOT a gate: exp18's "
                     "own determinism gate was relaxed too")}


if __name__ == "__main__":
    main()
