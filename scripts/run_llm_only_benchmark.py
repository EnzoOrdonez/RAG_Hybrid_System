"""
Run the LLM-only (no RAG) baseline benchmark — Control 0.

Quantifies the absolute value RAG adds over a vanilla LLM by running the
LLM over the 200-query benchmark without any retrieval. Outputs are
written to experiments/results/exp9_llm_only_no_rag/.

Usage:
    python scripts/run_llm_only_benchmark.py
    python scripts/run_llm_only_benchmark.py --max-queries 3
    python scripts/run_llm_only_benchmark.py --model mistral --resume
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_runner import BenchmarkRunner  # noqa: E402
from src.evaluation.test_queries import load_queries  # noqa: E402

QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
EXPERIMENT_ID = "exp9_llm_only_no_rag"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
LOG_PATH = RESULTS_DIR / EXPERIMENT_ID / "run.log"


def _resolve_model(short_name: str) -> str:
    """Map common short names to fully-qualified Ollama model tags."""
    aliases = {
        "llama3.1": "llama3.1:8b-instruct-q4_K_M",
        "llama": "llama3.1:8b-instruct-q4_K_M",
        "mistral": "mistral:7b-instruct",
        "qwen": "qwen2.5:7b-instruct",
    }
    return aliases.get(short_name, short_name)


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-only (no RAG) benchmark — Control 0."
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        help=(
            "Ollama model alias or full tag. Default: llama3.1 "
            "(maps to llama3.1:8b-instruct-q4_K_M)."
        ),
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=200,
        help="Number of queries to run. Default: 200.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if one exists.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N queries. Default: 10.",
    )
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=120,
        help="Per-query timeout in seconds. Default: 120.",
    )
    args = parser.parse_args()

    _setup_logging(LOG_PATH)
    log = logging.getLogger("run_llm_only_benchmark")

    model = _resolve_model(args.model)
    log.info("=" * 70)
    log.info("LLM-only (no RAG) benchmark — Control 0 (exp9_llm_only_no_rag)")
    log.info("Model:        %s", model)
    log.info("Max queries:  %d", args.max_queries)
    log.info("Resume:       %s", args.resume)
    log.info("Checkpoint:   every %d queries", args.checkpoint_interval)
    log.info("Timeout:      %d s/query", args.query_timeout)
    log.info("=" * 70)

    if not QUERIES_PATH.exists():
        log.error("Queries file not found: %s", QUERIES_PATH)
        return 1
    queries = load_queries(str(QUERIES_PATH))
    log.info("Loaded %d queries from %s", len(queries), QUERIES_PATH)

    from experiments.experiment_configs import (
        EXP9_LLM_ONLY_NO_RAG,
        ExperimentConfig,
    )

    base_config = EXP9_LLM_ONLY_NO_RAG.pipeline_configs[0]
    if model != base_config.llm_model:
        log.info(
            "Overriding LLM model: %s -> %s", base_config.llm_model, model
        )
        overridden = base_config.model_copy(update={"llm_model": model})
        exp_config = ExperimentConfig(
            experiment_id=EXP9_LLM_ONLY_NO_RAG.experiment_id,
            name=EXP9_LLM_ONLY_NO_RAG.name,
            description=EXP9_LLM_ONLY_NO_RAG.description,
            hypothesis=EXP9_LLM_ONLY_NO_RAG.hypothesis,
            variable=EXP9_LLM_ONLY_NO_RAG.variable,
            pipeline_configs=[overridden],
            max_queries=EXP9_LLM_ONLY_NO_RAG.max_queries,
            metrics=EXP9_LLM_ONLY_NO_RAG.metrics,
            seed=EXP9_LLM_ONLY_NO_RAG.seed,
        )
    else:
        exp_config = EXP9_LLM_ONLY_NO_RAG

    runner = BenchmarkRunner(
        checkpoint_interval=args.checkpoint_interval,
        query_timeout=args.query_timeout,
        seed=42,
    )

    t0 = time.perf_counter()
    results = runner.run_experiment(
        experiment_config=exp_config,
        queries=queries,
        resume=args.resume,
        max_queries=args.max_queries,
    )
    elapsed = time.perf_counter() - t0
    log.info(
        "Total wall time: %.1f s (%.1f min)", elapsed, elapsed / 60.0
    )

    runner.print_experiment_summary(EXPERIMENT_ID, results)

    config_name = exp_config.pipeline_configs[0].name
    qrs = results.get(config_name, [])
    if not qrs:
        log.error("No query results produced. Check %s.", LOG_PATH)
        return 1

    valid = [r for r in qrs if not r.error]
    errors = len(qrs) - len(valid)
    log.info(
        "Done: %d valid / %d errors / %d total",
        len(valid),
        errors,
        len(qrs),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
