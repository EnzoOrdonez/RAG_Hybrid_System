"""exp17 — generation over the provider-balanced vs baseline cross-cloud retrieval.

Two arms (baseline, balanced) share the 25 cross-cloud queries; each uses its top-5 id list
from build_balanced_retrieval_exp17.py (retrieval_ids.json). Generation replicates the exp12
canonical path (rgm.build_prompt -> cross_cloud template with context_by_provider, granite
temp0 seed42) and the H5 discipline of the Tier A harness: warmup + 3x determinism probe
(relaxed gate: warn, don't abort) + per-arm checkpoint. --no-cache so both arms are generated
fresh in THIS session (within-session paired; retrieval itself is deterministic).

Writes experiments/results/exp17_crosscloud_balanced/results.json in the standard schema, so
the existing scorers (run_exp15_ablation --pass N, rescore_grounding_tierA, compute_tierA_arm_stats,
compute_exp16_guards) all work via --exp-id / --exp-dir.

Usage: python scripts/run_exp17_crosscloud.py [--no-cache] [--max-queries N] [--no-resume]
Env:   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42
"""
import argparse
import importlib.util
import json
import logging
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.reproducibility import ensure_hashseed_at_startup, set_all_seeds  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp17")
SEED = 42
CHECKPOINT_EVERY = 10
EXP_DIR = PROJECT_ROOT / "experiments/results/exp17_crosscloud_balanced"
SUBSET = PROJECT_ROOT / "data/evaluation/cross_cloud_subset.json"
CHUNK_MAP_PATH = PROJECT_ROOT / "data/indices/chunk_map_bge-large_adaptive_500.json"
ARMS = ["baseline", "balanced"]

_spec = importlib.util.spec_from_file_location("rgm", PROJECT_ROOT / "scripts/run_generation_matrix.py")
rgm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rgm)


class ChunkMapIndex:
    def __init__(self, chunk_map):
        self._m = chunk_map

    def get_chunk(self, cid):
        return self._m.get(cid)


def ckpt_path(label, arm):
    return EXP_DIR / f"checkpoint__{label}__{arm}.json"


def main():
    ensure_hashseed_at_startup(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
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

    ids_doc = json.loads((EXP_DIR / "retrieval_ids.json").read_text(encoding="utf-8"))["ids"]
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    questions = {it["query_id"]: it["question"] for it in subset}
    qids = [it["query_id"] for it in subset]
    if args.max_queries:
        qids = qids[: args.max_queries]
    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    index = ChunkMapIndex(chunk_map)
    tag = "granite4.1:8b"
    label = rgm.model_label(tag)
    qp = QueryProcessor()
    qtype = {q: qp.process(questions[q]).query_type for q in qids}

    def arm_ids(arm, qid):
        return ids_doc[qid][f"{arm}_ids"]

    llm = LLMManager(provider="ollama", model=tag, cache_enabled=not args.no_cache, seed=SEED)
    probe_report = {}

    if qids:
        q0 = qids[0]
        pr0, sp0, _ = rgm.build_prompt("hibrido", questions[q0], arm_ids("baseline", q0),
                                       index, qtype[q0], P)
        LLMManager(provider="ollama", model=tag, cache_enabled=False, seed=SEED).generate(
            prompt=pr0, system_prompt=sp0, temperature=0.0, config_name="warmup")
        logger.info("warmup generation done")

    for arm in ARMS:
        config_name = f"{arm} | {label}"
        q0 = qids[0]
        pr, sp, _ = rgm.build_prompt("hibrido", questions[q0], arm_ids(arm, q0),
                                     index, qtype[q0], P)
        llm_nc = LLMManager(provider="ollama", model=tag, cache_enabled=False, seed=SEED)
        outs = [llm_nc.generate(prompt=pr, system_prompt=sp, temperature=0.0,
                                config_name=f"detprobe|{arm}|{label}").text for _ in range(3)]
        det_ok = all(o == outs[0] for o in outs)
        probe_report[arm] = {"determinism_3x_identical": det_ok,
                             "answer_lens_3x": [len(o) for o in outs]}
        logger.info("[%s] probe: determinism=%s", arm, det_ok)
        if not det_ok:
            logger.warning("[%s] determinism probe NOT bit-identical (relaxed gate, single-sample)", arm)

        cpath = ckpt_path(label, arm)
        results, done = [], set()
        if not args.no_resume and cpath.exists():
            ck = json.loads(cpath.read_text(encoding="utf-8"))
            results = ck["results"]; done = set(ck["completed_ids"])
            logger.info("[%s] resume: %d done", config_name, len(done))
        todo = [q for q in qids if q not in done]
        for i, qid in enumerate(todo):
            prompt, sysp, _ = rgm.build_prompt("hibrido", questions[qid], arm_ids(arm, qid),
                                               index, qtype[qid], P)
            t = time.perf_counter()
            resp = llm.generate(prompt=prompt, system_prompt=sysp, temperature=0.0,
                                config_name=config_name)
            gen_ms = resp.latency_ms or (time.perf_counter() - t) * 1000
            results.append({
                "query_id": qid, "config_name": config_name, "scenario": arm, "model": tag,
                "question": questions[qid], "answer": resp.text,
                "retrieved_ids": arm_ids(arm, qid),
                "hallucination_metrics": {"method": "pending_nli"},
                "tokens": {"input": resp.tokens_input, "output": resp.tokens_output},
                "latency": {"generation_ms": round(gen_ms, 1)},
                "from_cache": resp.from_cache, "error": resp.error,
                "timestamp": datetime.now().isoformat(),
            })
            done.add(qid)
            if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(todo):
                cpath.write_text(json.dumps(
                    {"config_name": config_name, "completed_ids": sorted(done), "results": results},
                    ensure_ascii=False), encoding="utf-8")
                logger.info("[%s] %d/%d", config_name, len(done), len(qids))

    configs = {}
    for arm in ARMS:
        cpath = ckpt_path(label, arm)
        if not cpath.exists():
            continue
        ck = json.loads(cpath.read_text(encoding="utf-8"))
        rs = ck["results"]
        configs[ck["config_name"]] = {
            "total_queries": len(rs), "errors": sum(1 for r in rs if r.get("error")),
            "scenario": arm, "model": tag, "results": rs}
    prior = {}
    rj = EXP_DIR / "results.json"
    if rj.exists():
        try:
            prior = json.loads(rj.read_text(encoding="utf-8")).get("probe_report", {})
        except Exception:
            prior = {}
    merged = {**prior, **probe_report}
    payload = {
        "experiment_id": "exp17_crosscloud_balanced",
        "name": "exp17 provider-balanced vs baseline retrieval (cross-cloud comparative)",
        "timestamp": datetime.now().isoformat(), "seed": SEED, "temperature": 0.0,
        "context_source": "exp17 retrieval_ids.json (same hybrid pool; baseline top-5 vs provider-balanced top-5)",
        "queries": str(SUBSET.relative_to(PROJECT_ROOT)), "num_queries": len(qids),
        "model": tag, "probe_report": merged,
        "all_arms_bit_deterministic": all(p.get("determinism_3x_identical") for p in merged.values()) if merged else None,
        "reproducibility_note": ("Single-sample generations; retrieval deterministic. Bit-determinism per "
                                 "arm in probe_report; paired arm-vs-baseline valid (same session/queries)."),
        "configs": configs,
    }
    rj.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", rj)


if __name__ == "__main__":
    main()
