"""
Generation matrix runner (exp12) — Nota 3.

Scenarios x models x queries. Generation + NLI faithfulness only; retrieval is
NOT re-run. The retrieved contexts are the canonical exp11 lists (D12 reranker),
loaded once per scenario and reused across all models — this is both the
mandated optimization (retrieval is deterministic and identical across models)
and a guarantee that exp12 contexts == exp11 (no retrieval<->generation leak).

Scenarios (4):
  sin_rag  -> no retrieval, NO_RAG prompt (Control 0)
  lexico   -> exp11 "RAG Lexico (BM25)" retrieved_ids
  denso    -> exp11 "RAG Semantico (Dense)" retrieved_ids
  hibrido  -> exp11 "RAG Hibrido Propuesto" retrieved_ids (post-rerank, full text)

Prompt building replicates RAGPipeline.query exactly (build_context / get_template
/ SYSTEM_PROMPT, and NO_RAG_PROMPT for sin_rag). Generation is greedy
(temperature=0). Per-query unified schema persists answer, hallucination_metrics
(faithfulness/claims/method/is_honest_decline), tokens, per-stage latency, and a
compute cost-proxy. Checkpoint/resume per (model, scenario).

Smoke (gate before the full matrix):
  python scripts/run_generation_matrix.py --exp-id exp12_smoke \
      --models granite4.1:8b,gemma4:e4b,gemma4:12b,qwen3.5:9b,mistral3:8b \
      --max-queries 3 --determinism-check

Full matrix (post-gate, per stage with resume):
  python scripts/run_generation_matrix.py --exp-id exp12_matrix \
      --models <chosen> [--scenarios sin_rag,lexico,denso,hibrido]

Env: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 (Ollama needs no HF).
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import ensure_hashseed_at_startup, set_all_seeds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gen_matrix")

SEED = 42
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
EXP11_PATH = PROJECT_ROOT / "experiments" / "results" / "exp11_retrieval194_fullrerank" / "results.json"

# scenario_key -> exp11 config name (None = no retrieval)
SCENARIO_SOURCE = {
    "sin_rag": None,
    "lexico": "RAG Lexico (BM25)",
    "denso": "RAG Semantico (Dense)",
    "hibrido": "RAG Hibrido Propuesto",
}
CHECKPOINT_EVERY = 10


def model_label(tag: str) -> str:
    return tag.replace("/", "-").replace(":", "-")


def gpu_mem_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def load_exp11_contexts():
    """scenario_key -> {query_id -> retrieved_ids}; plus query_id -> question."""
    data = json.loads(EXP11_PATH.read_text(encoding="utf-8"))
    ctx = {}
    questions = {}
    for scen, src in SCENARIO_SOURCE.items():
        if src is None:
            continue
        if src not in data["configs"]:
            raise SystemExit(f"exp11 missing config '{src}' for scenario '{scen}'")
        per_q = {}
        for r in data["configs"][src]["results"]:
            per_q[r["query_id"]] = r["retrieved_ids"]
            questions[r["query_id"]] = r["question"]
        ctx[scen] = per_q
    return ctx, questions


def build_prompt(scenario, question, retrieved_ids, index, query_type, P):
    """Return (prompt, system_prompt, chunk_dicts) replicating RAGPipeline."""
    if scenario == "sin_rag":
        return P["NO_RAG_PROMPT"].format(question=question), P["NO_RAG_SYSTEM_PROMPT"], []
    chunk_dicts = [cd for cid in retrieved_ids if (cd := index.get_chunk(cid))]
    context = P["build_context"](chunk_dicts, query_type)
    template = P["get_template"](query_type)
    if query_type == "cross_cloud":
        prompt = template.format(context_by_provider=context, question=question)
    else:
        prompt = template.format(context=context, question=question)
    return prompt, P["SYSTEM_PROMPT"], chunk_dicts


def ckpt_path(exp_dir, label, scenario):
    return exp_dir / f"checkpoint__{label}__{scenario}.json"


def main():
    ensure_hashseed_at_startup(SEED)
    ap = argparse.ArgumentParser(description="Generation matrix runner (exp12)")
    ap.add_argument("--exp-id", default="exp12_matrix")
    ap.add_argument("--models", required=True, help="comma-separated Ollama tags")
    ap.add_argument("--scenarios", default="sin_rag,lexico,denso,hibrido")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--determinism-check", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    set_all_seeds(SEED)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    exp_dir = PROJECT_ROOT / "experiments" / "results" / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Imports that load heavy deps
    from src.pipeline.rag_pipeline import load_hybrid_index
    from src.generation.llm_manager import LLMManager
    from src.generation.hallucination_detector import HallucinationDetector
    from src.generation.response_formatter import ResponseFormatter
    from src.retrieval.query_processor import QueryProcessor
    from src.generation import prompt_templates as PT
    P = {k: getattr(PT, k) for k in
         ("NO_RAG_PROMPT", "NO_RAG_SYSTEM_PROMPT", "SYSTEM_PROMPT",
          "build_context", "get_template")}

    logger.info("Loading index + exp11 contexts ...")
    index = load_hybrid_index(embedding_model="bge-large", chunking_strategy="adaptive", chunk_size=500)
    contexts, questions = load_exp11_contexts()

    all_qids = list(questions.keys())
    if args.max_queries:
        all_qids = all_qids[:args.max_queries]
    logger.info("Models=%s scenarios=%s queries=%d", models, scenarios, len(all_qids))

    # Query types (no model; cheap) — computed once, shared across models.
    qp = QueryProcessor()
    qtype = {qid: qp.process(questions[qid]).query_type for qid in all_qids}

    detector = HallucinationDetector(use_nli=True)
    _ = detector.nli_model  # load once
    rf = ResponseFormatter()

    smoke_report = {}  # (model) -> dict of smoke diagnostics

    for tag in models:
        label = model_label(tag)
        logger.info("=== MODEL %s (label=%s) ===", tag, label)
        try:
            llm = LLMManager(provider="ollama", model=tag, cache_enabled=True, seed=SEED)
        except Exception as e:
            logger.error("Cannot init model %s: %s", tag, e)
            smoke_report[tag] = {"error": str(e)}
            continue

        # Determinism probe (cache OFF): same hibrido query x3 must match at temp=0.
        if args.determinism_check and all_qids:
            qid0 = all_qids[0]
            rids0 = contexts.get("hibrido", {}).get(qid0, [])
            pr, sp, _ = build_prompt("hibrido", questions[qid0], rids0, index, qtype[qid0], P)
            llm_nc = LLMManager(provider="ollama", model=tag, cache_enabled=False, seed=SEED)
            outs = []
            for _ in range(3):
                r = llm_nc.generate(prompt=pr, system_prompt=sp, temperature=0.0,
                                    config_name=f"detcheck|{label}")
                outs.append(r.text)
            det_ok = all(o == outs[0] for o in outs)
            tok_s0 = (r.tokens_output / (r.latency_ms / 1000)) if r.latency_ms else 0.0
            vram = gpu_mem_used_mb()
            smoke_report[tag] = {"determinism_3x_identical": det_ok,
                                 "tok_per_s": round(tok_s0, 2),
                                 "vram_used_mb": vram,
                                 "tokens_out_sample": r.tokens_output}
            logger.info("  determinism(3x)=%s tok/s=%.1f vram=%sMB",
                        det_ok, tok_s0, vram)

        for scenario in scenarios:
            config_name = f"{scenario} | {label}"
            cpath = ckpt_path(exp_dir, label, scenario)
            results, done = [], set()
            if not args.no_resume and cpath.exists():
                ck = json.loads(cpath.read_text(encoding="utf-8"))
                results = ck["results"]; done = set(ck["completed_ids"])
                logger.info("  [%s] resume: %d done", config_name, len(done))

            todo = [q for q in all_qids if q not in done]
            for i, qid in enumerate(todo):
                question = questions[qid]
                rids = contexts.get(scenario, {}).get(qid, []) if scenario != "sin_rag" else []
                prompt, sysp, chunk_dicts = build_prompt(scenario, question, rids, index, qtype[qid], P)
                t = time.perf_counter()
                resp = llm.generate(prompt=prompt, system_prompt=sysp,
                                    temperature=0.0, config_name=config_name)
                gen_ms = resp.latency_ms or (time.perf_counter() - t) * 1000
                report = detector.check(resp.text, chunk_dicts)
                is_decline = rf._is_honest_decline(resp.text)
                tok_total = (resp.tokens_input or 0) + (resp.tokens_output or 0)
                results.append({
                    "query_id": qid, "config_name": config_name,
                    "scenario": scenario, "model": tag,
                    "question": question, "answer": resp.text,
                    "retrieved_ids": rids,
                    "hallucination_metrics": {
                        "faithfulness": report.faithfulness_score,
                        "hallucination_rate": report.hallucination_rate,
                        "total_claims": report.total_claims,
                        "supported_claims": report.supported_claims,
                        "contradicted_claims": report.contradicted_claims,
                        "unsupported_claims": report.unsupported_claims,
                        "method": report.method,
                        "processing_time_ms": report.processing_time_ms,
                        "is_honest_decline": bool(is_decline),
                    },
                    "tokens": {"input": resp.tokens_input, "output": resp.tokens_output},
                    "latency": {"generation_ms": round(gen_ms, 1),
                                "hallucination_check_ms": report.processing_time_ms,
                                "total_ms": round(gen_ms + report.processing_time_ms, 1)},
                    "cost_proxy": {"gen_latency_ms": round(gen_ms, 1),
                                   "tokens_total": tok_total,
                                   "tok_per_s": round((resp.tokens_output / (gen_ms / 1000)), 2) if gen_ms else 0.0},
                    "from_cache": resp.from_cache,
                    "error": resp.error,
                    "timestamp": datetime.now().isoformat(),
                })
                done.add(qid)
                if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(todo):
                    cpath.write_text(json.dumps(
                        {"config_name": config_name, "completed_ids": sorted(done),
                         "results": results}, ensure_ascii=False), encoding="utf-8")
                    logger.info("  [%s] %d/%d", config_name, len(done), len(all_qids))

    # Fold checkpoints into a single results.json (exp8 schema).
    configs = {}
    for tag in models:
        label = model_label(tag)
        for scenario in scenarios:
            cpath = ckpt_path(exp_dir, label, scenario)
            if not cpath.exists():
                continue
            ck = json.loads(cpath.read_text(encoding="utf-8"))
            cname = ck["config_name"]
            rs = ck["results"]
            configs[cname] = {"total_queries": len(rs),
                              "errors": sum(1 for r in rs if r.get("error")),
                              "scenario": scenario, "model": tag, "results": rs}

    payload = {
        "experiment_id": args.exp_id,
        "name": "Generation matrix (scenarios x models) — Nota 3",
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "temperature": 0.0,
        "context_source": "exp11_retrieval194_fullrerank (reused, no re-retrieval)",
        "metrics_schema_version": "nota3-v1",
        "num_queries": len(all_qids),
        "models": models, "scenarios": scenarios,
        "smoke_report": smoke_report,
        "configs": configs,
    }
    (exp_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %s", exp_dir / "results.json")

    # Console summary
    print("\n" + "=" * 78)
    print(f"GENERATION MATRIX — {args.exp_id}")
    print("=" * 78)
    if smoke_report:
        print("Smoke diagnostics:")
        for tag, d in smoke_report.items():
            print(f"  {tag:<22} {d}")
        print("-" * 78)
    print(f"{'config':<34} {'n':>4} {'faith_mean':>10} {'decl%':>6} {'tok/s':>7}")
    print("-" * 78)
    for cname, c in configs.items():
        rs = c["results"]
        eff = [r["hallucination_metrics"]["faithfulness"] for r in rs
               if r["hallucination_metrics"]["method"] not in ("none", "error")]
        fm = sum(eff) / len(eff) if eff else 0.0
        decl = sum(1 for r in rs if r["hallucination_metrics"]["is_honest_decline"]) / len(rs) if rs else 0
        toks = [r["cost_proxy"]["tok_per_s"] for r in rs if r["cost_proxy"]["tok_per_s"]]
        ts = sum(toks) / len(toks) if toks else 0
        print(f"{cname[:34]:<34} {len(rs):>4} {fm:>10.3f} {100*decl:>5.1f} {ts:>7.1f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
