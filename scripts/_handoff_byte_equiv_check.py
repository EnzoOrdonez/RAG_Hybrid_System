"""
Block 3 — byte-equivalence verification.

Re-runs 3 queries from the saved exp8/results.json against the
post-PR code with PROPOSED_HYBRID config. Compares each output
field-by-field against the saved value. No code mutation.

Cache hit expected for the LLM call (config_name + prompt deterministic
under fixed seed, present in data/llm_cache).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.test_queries import load_queries  # noqa: E402
from src.pipeline.pipeline_config import PROPOSED_HYBRID  # noqa: E402
from src.pipeline.rag_pipeline import RAGPipeline, load_hybrid_index  # noqa: E402
from src.generation.llm_manager import LLMManager  # noqa: E402
from src.utils.reproducibility import set_all_seeds  # noqa: E402

EXP8_RESULTS = PROJECT_ROOT / "experiments" / "results" / "exp8" / "results.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
TARGET_IDS = ["q010", "q050", "q100"]
HYBRID_CONFIG_NAME = "RAG Hibrido Propuesto"


def main() -> int:
    set_all_seeds(42)
    exp8 = json.loads(EXP8_RESULTS.read_text(encoding="utf-8"))
    hybrid_rows = exp8["configs"][HYBRID_CONFIG_NAME]["results"]
    saved_by_id = {r["query_id"]: r for r in hybrid_rows}
    targets = [saved_by_id[qid] for qid in TARGET_IDS if qid in saved_by_id]
    if len(targets) != len(TARGET_IDS):
        print("Missing target query ids in exp8 results.")
        return 1

    queries = load_queries(str(QUERIES_PATH))
    queries_by_id = {q.query_id: q for q in queries}

    print("Loading hybrid index ...")
    idx = load_hybrid_index(
        embedding_model="bge-large",
        chunking_strategy="adaptive",
        chunk_size=500,
    )

    llm = LLMManager(
        provider="ollama",
        model="llama3.1:8b-instruct-q4_K_M",
        seed=42,
    )
    pipeline = RAGPipeline(
        config=PROPOSED_HYBRID,
        hybrid_index=idx,
        llm_manager=llm,
    )

    results = []
    for saved in targets:
        qid = saved["query_id"]
        tq = queries_by_id[qid]
        print(f"\nRunning {qid} ...")
        set_all_seeds(42)
        resp = pipeline.query(tq.question)

        new_ids = [c.get("chunk_id", "") for c in resp.retrieved_chunks]
        new_hall = resp.hallucination_report.model_dump() if resp.hallucination_report else {}

        saved_ids = saved.get("retrieved_ids", [])
        saved_hall = saved.get("hallucination_metrics", {})

        diff = {}
        diff["answer_equal"] = (resp.answer == saved.get("answer", ""))
        diff["retrieved_ids_equal"] = (new_ids == saved_ids)
        diff["retrieved_ids_count_new"] = len(new_ids)
        diff["retrieved_ids_count_saved"] = len(saved_ids)
        diff["faithfulness_new"] = new_hall.get("faithfulness_score")
        diff["faithfulness_saved"] = saved_hall.get("faithfulness")
        diff["faithfulness_equal"] = (
            diff["faithfulness_new"] == diff["faithfulness_saved"]
        )
        diff["total_claims_new"] = new_hall.get("total_claims")
        diff["total_claims_saved"] = saved_hall.get("total_claims")
        diff["total_claims_equal"] = (
            diff["total_claims_new"] == diff["total_claims_saved"]
        )
        diff["supported_new"] = new_hall.get("supported_claims")
        diff["supported_saved"] = saved_hall.get("supported_claims")
        diff["supported_equal"] = (
            diff["supported_new"] == diff["supported_saved"]
        )
        diff["contradicted_new"] = new_hall.get("contradicted_claims")
        diff["contradicted_saved"] = saved_hall.get("contradicted_claims")
        diff["contradicted_equal"] = (
            diff["contradicted_new"] == diff["contradicted_saved"]
        )
        diff["unsupported_new"] = new_hall.get("unsupported_claims")
        diff["unsupported_saved"] = saved_hall.get("unsupported_claims")
        diff["unsupported_equal"] = (
            diff["unsupported_new"] == diff["unsupported_saved"]
        )
        diff["method_new"] = new_hall.get("method")
        diff["method_saved"] = saved_hall.get("method")
        diff["method_equal"] = diff["method_new"] == diff["method_saved"]

        diff["query_id"] = qid
        diff["answer_len_new"] = len(resp.answer)
        diff["answer_len_saved"] = len(saved.get("answer", ""))
        if not diff["answer_equal"]:
            # Find first divergence
            a, b = resp.answer, saved.get("answer", "")
            n = min(len(a), len(b))
            for i in range(n):
                if a[i] != b[i]:
                    diff["first_diff_offset"] = i
                    diff["new_excerpt"] = a[max(0, i - 30):i + 30]
                    diff["saved_excerpt"] = b[max(0, i - 30):i + 30]
                    break
            else:
                diff["first_diff_offset"] = n
                diff["truncation"] = "one is a prefix of the other"

        results.append(diff)

    out_path = (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "exp9_llm_only_no_rag"
        / "_handoff"
        / "byte_equiv_check.json"
    )
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for r in results:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
