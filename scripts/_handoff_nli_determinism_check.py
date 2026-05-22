"""
Audit follow-up: run q100 N times through current code to determine
whether NLI classification drift is intra-process (non-determinism) or
inter-process (seed-state-drift from the original exp8 run).
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

QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
TARGET_ID = "q100"
N_RUNS = 3


def main() -> int:
    set_all_seeds(42)
    queries = load_queries(str(QUERIES_PATH))
    q = next(x for x in queries if x.query_id == TARGET_ID)

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

    runs = []
    for i in range(N_RUNS):
        set_all_seeds(42)
        resp = pipeline.query(q.question)
        hr = resp.hallucination_report.model_dump()
        runs.append({
            "run": i + 1,
            "total_claims": hr["total_claims"],
            "supported": hr["supported_claims"],
            "contradicted": hr["contradicted_claims"],
            "unsupported": hr["unsupported_claims"],
            "faithfulness": hr["faithfulness_score"],
            "method": hr["method"],
            "retrieved_ids_count": len(resp.retrieved_chunks),
            "answer_len": len(resp.answer),
        })

    out_path = (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "exp9_llm_only_no_rag"
        / "_handoff"
        / "nli_determinism_check.json"
    )
    out_path.write_text(
        json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(runs, indent=2, ensure_ascii=False))
    # Determinism verdict
    keys = ["total_claims", "supported", "contradicted", "unsupported", "faithfulness"]
    all_same = all(runs[0][k] == r[k] for r in runs for k in keys)
    print()
    if all_same:
        print("INTRA-PROCESS DETERMINISTIC: all 3 runs equal. Divergence with exp8 is inter-process state-drift.")
    else:
        print("INTRA-PROCESS NON-DETERMINISTIC: runs differ — NLI itself has run-to-run drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
