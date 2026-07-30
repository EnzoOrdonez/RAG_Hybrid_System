"""exp18 — generation over the four evidence arms (ceiling diagnostic).

Reads the id lists from build_exp18_evidence_arms.py and generates one answer per
(arm, query) with the canonical prompt path (rgm.build_prompt), granite temp0 seed42,
mirroring the Tier A / exp17 H5 discipline: warmup, 3x determinism probe per arm
(relaxed gate: warn, never abort), per-arm checkpoint, --no-cache so every arm is
generated in THIS session and the paired contrast is within-session.

The question and its query_type always stay the query's OWN, including in
`evidence_swapped` -- the arm asks whether the generator follows the evidence it was
handed for the question it was asked, so changing the question too would test nothing.

Writes results.json in the standard schema, so the existing scorers all work via
--exp-dir: run_exp15_ablation.py --pass N, rescore_grounding_tierA.py,
compute_tierA_arm_stats.py --baseline-arm baseline_repro, compute_exp16_guards.py.

Usage: python scripts/run_exp18_ceiling.py [--no-cache] [--arms a,b] [--max-queries N]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
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
logger = logging.getLogger("exp18")
SEED = 42
CHECKPOINT_EVERY = 10
EXP_DIR = PROJECT_ROOT / "experiments/results/exp18_evidence_ceiling"
CHUNK_MAP_PATH = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
# baseline first: the anchor must exist before any contrast is meaningful
ARMS = ["baseline_repro", "oracle_evidence", "evidence_swapped", "final_top_k_10"]

# Per-arm query scale. Not uniform, and deliberately so: with the paired-difference SD
# measured in Tier A (0.318), n=57 only detects 0.118 in faithfulness -- larger than the
# biggest effect this phase ever found (exp17 HHEM +0.081). An arm whose NULL is going to
# be believed therefore cannot run at 60.
#   oracle_evidence   all  the only arm whose null drives a decision (the cloud spend), so
#                          it carries the pre-registered TOST and needs the power for it
#   final_top_k_10    all  analysed SPLIT by truncation; at n=60 the split was 24/36 and
#                          neither stratum could say anything
#   evidence_swapped  60   expects a large effect by construction; 60 is plenty
#   baseline_repro    all  anchor for the two full-scale arms (60-subset ⊂ 194)
ARM_SCALE = {"baseline_repro": "all", "oracle_evidence": "all",
             "final_top_k_10": "all", "evidence_swapped": "subset"}

_spec = importlib.util.spec_from_file_location(
    "rgm", PROJECT_ROOT / "scripts/run_generation_matrix.py")
rgm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgm)


class ChunkMapIndex:
    def __init__(self, chunk_map):
        self._m = chunk_map

    def get_chunk(self, cid):
        return self._m.get(cid)


def main():
    ensure_hashseed_at_startup(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--arms", default=None, help="comma-separated subset of the 4 arms")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    set_all_seeds(SEED)

    from src.generation.llm_manager import LLMManager
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import prompt_templates as PT
    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT",
          "build_context", "get_template")}

    doc = json.loads((EXP_DIR / "retrieval_ids.json").read_text(encoding="utf-8"))
    ids_doc = doc["ids"]
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(ARMS)
    bad = [a for a in arms if a not in ARMS]
    if bad:
        sys.exit(f"unknown arm(s) {bad}; known: {ARMS}")
    all_qids = list(ids_doc)
    sub_qids = [q for q in all_qids if ids_doc[q].get("in_summer_subset")]
    if not sub_qids:
        sys.exit("retrieval_ids.json has no `in_summer_subset` flags — rebuild with the "
                 "current build_exp18_evidence_arms.py")

    def qids_for(arm):
        qs = all_qids if ARM_SCALE[arm] == "all" else sub_qids
        return qs[: args.max_queries] if args.max_queries else qs

    qids = all_qids[: args.max_queries] if args.max_queries else all_qids
    missing = [a for a in arms
               if any(ids_doc[q].get(f"{a}_ids") is None for q in qids_for(a))]
    if missing:
        sys.exit(f"arm(s) {missing} have no ids in retrieval_ids.json — rebuild with "
                 f"build_exp18_evidence_arms.py (did you pass --no-oracle?)")
    logger.info("scale: %s", {a: len(qids_for(a)) for a in arms})

    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    index = ChunkMapIndex(chunk_map)
    tag = "granite4.1:8b"
    label = rgm.model_label(tag)
    qp = QueryProcessor()
    questions = {q: ids_doc[q]["question"] for q in qids}
    qtype = {q: qp.process(questions[q]).query_type for q in qids}

    llm = LLMManager(provider="ollama", model=tag, cache_enabled=not args.no_cache, seed=SEED)
    probe_report = {}

    if qids:
        q0 = qids[0]
        pr0, sp0, _ = rgm.build_prompt("hibrido", questions[q0],
                                       ids_doc[q0]["baseline_repro_ids"], index, qtype[q0], P)
        LLMManager(provider="ollama", model=tag, cache_enabled=False, seed=SEED).generate(
            prompt=pr0, system_prompt=sp0, temperature=0.0, config_name="warmup")
        logger.info("warmup generation done")

    for arm in arms:
        config_name = f"{arm} | {label}"
        arm_qids = qids_for(arm)
        q0 = arm_qids[0]
        pr, sp, _ = rgm.build_prompt("hibrido", questions[q0], ids_doc[q0][f"{arm}_ids"],
                                     index, qtype[q0], P)
        llm_nc = LLMManager(provider="ollama", model=tag, cache_enabled=False, seed=SEED)
        outs = [llm_nc.generate(prompt=pr, system_prompt=sp, temperature=0.0,
                                config_name=f"detprobe|{arm}|{label}").text for _ in range(3)]
        det_ok = all(o == outs[0] for o in outs)
        probe_report[arm] = {"determinism_3x_identical": det_ok,
                             "answer_lens_3x": [len(o) for o in outs]}
        logger.info("[%s] probe: determinism=%s", arm, det_ok)
        if not det_ok:
            logger.warning("[%s] determinism probe NOT bit-identical (relaxed gate)", arm)

        cpath = EXP_DIR / f"checkpoint__{label}__{arm}.json"
        results, done = [], set()
        if not args.no_resume and cpath.exists():
            ck = json.loads(cpath.read_text(encoding="utf-8"))
            results, done = ck["results"], set(ck["completed_ids"])
            logger.info("[%s] resume: %d done", config_name, len(done))
        todo = [q for q in arm_qids if q not in done]
        for i, qid in enumerate(todo):
            arm_ids = ids_doc[qid][f"{arm}_ids"]
            prompt, sysp, _ = rgm.build_prompt("hibrido", questions[qid], arm_ids,
                                               index, qtype[qid], P)
            t = time.perf_counter()
            resp = llm.generate(prompt=prompt, system_prompt=sysp, temperature=0.0,
                                config_name=config_name)
            gen_ms = resp.latency_ms or (time.perf_counter() - t) * 1000
            results.append({
                "query_id": qid, "config_name": config_name, "scenario": arm, "model": tag,
                "question": questions[qid], "answer": resp.text,
                "retrieved_ids": arm_ids,
                "query_type": qtype[qid],
                "swap_partner": ids_doc[qid]["swap_partner"] if arm == "evidence_swapped" else None,
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
                logger.info("[%s] %d/%d", config_name, len(done), len(arm_qids))

    configs = {}
    for arm in ARMS:
        cpath = EXP_DIR / f"checkpoint__{label}__{arm}.json"
        if not cpath.exists():
            continue
        ck = json.loads(cpath.read_text(encoding="utf-8"))
        rs = ck["results"]
        configs[ck["config_name"]] = {
            "total_queries": len(rs), "errors": sum(1 for r in rs if r.get("error")),
            "scenario": arm, "model": tag, "results": rs}

    rj = EXP_DIR / "results.json"
    prior = {}
    if rj.exists():
        try:
            prior = json.loads(rj.read_text(encoding="utf-8")).get("probe_report", {})
        except Exception:
            prior = {}
    merged = {**prior, **probe_report}
    # observed truncation, now that real token counts exist (the build only estimated)
    trunc = {}
    for cname, c in configs.items():
        over = [r["query_id"] for r in c["results"]
                if (r.get("tokens") or {}).get("input", 0) >= 4096]
        trunc[c["scenario"]] = {"n_at_context_limit": len(over), "qids": over}
    rj.write_text(json.dumps({
        "experiment_id": "exp18_evidence_ceiling",
        "name": "exp18 evidence-ceiling diagnostic (selection / attention / quantity)",
        "timestamp": datetime.now().isoformat(), "seed": SEED, "temperature": 0.0,
        "context_source": "exp18 retrieval_ids.json (same hybrid pool; 4 selections)",
        "queries": doc.get("source", "test_queries.json"),
        "num_queries": len(qids),
        "arm_scale": {a: {"scope": ARM_SCALE[a], "n": len(qids_for(a))} for a in ARMS},
        "model": tag, "probe_report": merged,
        "observed_truncation": trunc,
        "all_arms_bit_deterministic": (
            all(p.get("determinism_3x_identical") for p in merged.values()) if merged else None),
        "reproducibility_note": ("Single-sample generations; retrieval deterministic. Paired "
                                 "arm-vs-baseline_repro valid (same session/queries). The "
                                 "final_top_k_10 arm is partly truncated by the 4096 window "
                                 "and must be read split by observed_truncation."),
        "bh_family_note": ("3 arm-vs-baseline_repro contrasts (oracle_evidence, "
                           "evidence_swapped, final_top_k_10) per verifier."),
        "configs": configs,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", rj)


if __name__ == "__main__":
    main()
