"""
exp13 — cross-cloud terminology expansion ON vs OFF (the real test of exp7).

For the cross-cloud query subset (>1 cloud_provider), runs the hybrid pipeline
with the D11 expansion flag OFF and ON. OFF = the paper's main configuration
(RRF, raw query to both legs). ON = the expanded query routed to the BM25 leg
(D11 fix). Both arms: rerank -> generate (greedy, deterministic model) -> NLI.

Outputs experiments/results/<exp_id>/results.json with two configs
("exp_off | <model>", "exp_on | <model>") so:
  - compute_retrieval_metrics.py compares ON vs OFF RETRIEVAL (multi-oracle), and
  - compute_faithfulness_metrics.py compares ON vs OFF FAITHFULNESS (between-scenario).
Also logs the expanded BM25 query per consulta and counts how many queries had a
DIFFERENT retrieved set ON vs OFF (the direct measure of whether expansion does
anything at all — exp7's two arms were identical because expansion never reached
BM25 in RRF).

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 \
  python scripts/run_exp13_expansion.py --model granite4.1:8b
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.reproducibility import ensure_hashseed_at_startup, set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp13")
SEED = 42
QUERIES = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
SUBSET = PROJECT_ROOT / "data" / "evaluation" / "cross_cloud_subset.json"


def main():
    ensure_hashseed_at_startup(SEED)
    ap = argparse.ArgumentParser(description="exp13 expansion ON vs OFF")
    ap.add_argument("--model", default="granite4.1:8b")
    ap.add_argument("--exp-id", default="exp13_expansion")
    args = ap.parse_args()
    set_all_seeds(SEED)
    label = args.model.replace("/", "-").replace(":", "-")

    qs = json.loads(QUERIES.read_text(encoding="utf-8"))
    cc = [q for q in qs if len(q.get("cloud_providers", [])) > 1]
    SUBSET.write_text(json.dumps(cc, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Cross-cloud subset: %d queries -> %s", len(cc), SUBSET.name)

    from src.pipeline.rag_pipeline import load_hybrid_index
    from src.pipeline.pipeline_config import PROPOSED_HYBRID as CFG
    from src.retrieval.query_processor import QueryProcessor
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.reranking.cross_encoder_reranker import CrossEncoderReranker
    from src.generation.llm_manager import LLMManager
    from src.generation.hallucination_detector import HallucinationDetector
    from src.generation.prompt_templates import build_context, get_template, SYSTEM_PROMPT

    index = load_hybrid_index(embedding_model="bge-large", chunking_strategy="adaptive", chunk_size=500)
    qp = QueryProcessor()
    hybrid = HybridRetriever(index, query_processor=qp, reranker=None,
                             fusion_method=CFG.fusion_method or "rrf", alpha=CFG.alpha, rrf_k=CFG.rrf_k)
    reranker = CrossEncoderReranker(model_name="ms-marco-mini-12")
    _ = reranker.model
    llm = LLMManager(provider="ollama", model=args.model, cache_enabled=True, seed=SEED)
    detector = HallucinationDetector(use_nli=True)
    _ = detector.nli_model

    arms = [("exp_off", False), ("exp_on", True)]
    results = {f"{a} | {label}": [] for a, _ in arms}
    diff_count = 0

    for q in cc:
        qid, question = q["query_id"], q["question"]
        processed = qp.process(question)
        qtype = processed.query_type
        expanded = processed.bm25_query
        ids_by_arm = {}
        for arm, flag in arms:
            cname = f"{arm} | {label}"
            cands = hybrid.search(question, top_k=CFG.retrieval_top_k,
                                  top_k_candidates=CFG.retrieval_top_k,
                                  use_reranker=False, use_expansion=flag)
            reranked = reranker.rerank(question, cands, top_k=CFG.final_top_k)
            ids = [r.chunk_id for r in reranked]
            ids_by_arm[arm] = ids
            chunk_dicts = [cd for cid in ids if (cd := index.get_chunk(cid))]
            context = build_context(chunk_dicts, qtype)
            template = get_template(qtype)
            prompt = (template.format(context_by_provider=context, question=question)
                      if qtype == "cross_cloud" else
                      template.format(context=context, question=question))
            t = time.perf_counter()
            resp = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT,
                                temperature=0.0, config_name=cname)
            gen_ms = resp.latency_ms or (time.perf_counter() - t) * 1000
            report = detector.check(resp.text, chunk_dicts)
            results[cname].append({
                "query_id": qid, "config_name": cname, "scenario": arm, "model": args.model,
                "question": question, "answer": resp.text, "retrieved_ids": ids,
                "expansion_on": flag,
                "expanded_bm25_query": expanded if flag else question,
                "hallucination_metrics": {
                    "faithfulness": report.faithfulness_score,
                    "hallucination_rate": report.hallucination_rate,
                    "total_claims": report.total_claims,
                    "supported_claims": report.supported_claims,
                    "contradicted_claims": report.contradicted_claims,
                    "unsupported_claims": report.unsupported_claims,
                    "method": report.method,
                    "processing_time_ms": report.processing_time_ms,
                },
                "tokens": {"input": resp.tokens_input, "output": resp.tokens_output},
                "latency": {"generation_ms": round(gen_ms, 1)},
                "from_cache": resp.from_cache, "error": resp.error,
                "timestamp": datetime.now().isoformat(),
            })
        same = ids_by_arm["exp_off"] == ids_by_arm["exp_on"]
        if not same:
            diff_count += 1
        results[f"exp_on | {label}"][-1]["retrieval_changed_vs_off"] = (not same)
        if expanded != question:
            logger.info("  %s expanded: '%s' (retrieval %s)", qid, expanded[:90],
                        "CHANGED" if not same else "unchanged")

    exp_dir = PROJECT_ROOT / "experiments" / "results" / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": args.exp_id, "name": "Cross-cloud expansion ON vs OFF (D11)",
        "model": args.model, "seed": SEED, "temperature": 0.0,
        "num_queries": len(cc),
        "retrieval_changed_count": diff_count,
        "timestamp": datetime.now().isoformat(),
        "configs": {c: {"total_queries": len(rs),
                        "errors": sum(1 for r in rs if r.get("error")),
                        "results": rs} for c, rs in results.items()},
    }
    (exp_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    def faith(c):
        rs = results[c]
        eff = [r["hallucination_metrics"]["faithfulness"] for r in rs
               if r["hallucination_metrics"]["method"] not in ("none", "error")]
        return sum(eff) / len(eff) if eff else 0.0
    print("\n" + "=" * 64)
    print(f"exp13 EXPANSION ON vs OFF — {args.model} ({len(cc)} cross-cloud q)")
    print("=" * 64)
    print(f"Queries whose RETRIEVAL changed ON vs OFF: {diff_count}/{len(cc)}")
    print(f"Faithfulness OFF: {faith(f'exp_off | {label}'):.3f}")
    print(f"Faithfulness ON : {faith(f'exp_on | {label}'):.3f}")
    print(f"Wrote {exp_dir / 'results.json'}")
    print("Next: compute_retrieval_metrics.py --experiment", args.exp_id,
          "--oracle-model BAAI/bge-reranker-large --oracle-label bge-reranker-indep")
    print("      compute_faithfulness_metrics.py --experiment", args.exp_id)


if __name__ == "__main__":
    main()
