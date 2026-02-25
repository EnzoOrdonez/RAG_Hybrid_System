"""
Run evaluation benchmarks for the thesis.

Usage:
    python scripts/run_benchmark.py --experiment exp8
    python scripts/run_benchmark.py --all
    python scripts/run_benchmark.py --experiment exp8 --quick --max-queries 20
    python scripts/run_benchmark.py --experiment exp8 --resume
    python scripts/run_benchmark.py --export-only
    python scripts/run_benchmark.py --list

Experiments:
    exp1: Chunking strategy comparison (15 configs)
    exp2: Embedding model comparison (3 models)
    exp3: Retrieval method comparison (4 methods)
    exp4: Reranker comparison (4 configs)
    exp5: LLM comparison (3 models)
    exp6: Ablation study (5 stages)
    exp7: Cross-cloud normalization (2 configs)
    exp8: End-to-end system comparison (3 pipelines)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Fix tqdm OSError on Python 3.14 + Windows (stderr flush issue)
os.environ["TQDM_DISABLE"] = "1"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "benchmark.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run thesis evaluation benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str, default=None,
        help="Experiment ID to run (e.g., exp1, exp8)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use subset of queries for fast testing",
    )
    parser.add_argument(
        "--max-queries", "-n",
        type=int, default=None,
        help="Maximum number of queries per experiment",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume, start fresh",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export results (no benchmark run)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments",
    )
    parser.add_argument(
        "--queries",
        type=str, default="data/evaluation/test_queries.json",
        help="Path to test queries JSON file",
    )
    parser.add_argument(
        "--results-dir",
        type=str, default=None,
        help="Results directory (default: experiments/results)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # List experiments
    if args.list:
        from experiments.experiment_configs import list_experiments
        exps = list_experiments()
        print("\nAvailable experiments:")
        print("-" * 70)
        for exp in exps:
            print(f"  {exp['id']:6s} | {exp['name']:40s} | {exp['num_configs']} configs | {exp['max_queries']} queries")
        print()
        return

    # Export only
    if args.export_only:
        logger.info("Export-only mode: generating tables and figures...")
        from src.evaluation.results_exporter import ResultsExporter
        exporter = ResultsExporter(results_dir=args.results_dir)
        exporter.export_all()
        logger.info("Export complete!")
        return

    # Validate args
    if not args.experiment and not args.all:
        parser.error("Must specify --experiment <id> or --all")

    # Quick mode defaults
    max_queries = args.max_queries
    if args.quick and max_queries is None:
        max_queries = 20
        logger.info("Quick mode: using %d queries", max_queries)

    # Load queries
    queries_path = str(project_root / args.queries)
    queries_file = Path(queries_path)

    from src.evaluation.test_queries import generate_all_queries, save_queries, load_queries

    queries_list = None
    if queries_file.exists():
        try:
            queries_list = load_queries(queries_path)
            logger.info("Loaded %d queries from %s", len(queries_list), queries_path)
        except (ValueError, Exception) as e:
            logger.warning("Failed to load queries (%s), regenerating...", e)

    if queries_list is None:
        logger.info("Generating queries (seed=%d)...", args.seed)
        queries_list = generate_all_queries(count=200, seed=args.seed)
        save_queries(queries_list, queries_path)
        logger.info("Generated and saved %d queries to %s", len(queries_list), queries_path)

    # Create benchmark runner
    from src.evaluation.benchmark_runner import BenchmarkRunner
    runner = BenchmarkRunner(
        results_dir=args.results_dir,
        seed=args.seed,
    )

    resume = not args.no_resume
    start_time = time.time()

    if args.all:
        # Run all experiments
        from experiments.experiment_configs import EXPERIMENT_CONFIGS
        experiment_ids = list(EXPERIMENT_CONFIGS.keys())
        logger.info("Running ALL %d experiments...", len(experiment_ids))

        for exp_id in experiment_ids:
            from experiments.experiment_configs import get_experiment
            exp_config = get_experiment(exp_id)

            # For exp7, filter to cross-cloud queries
            if exp_id == "exp7":
                exp_queries = [q for q in queries_list if len(q.cloud_providers) > 1]
            else:
                exp_queries = queries_list

            results = runner.run_experiment(
                exp_config, exp_queries,
                resume=resume,
                max_queries=max_queries,
            )
            runner.print_experiment_summary(exp_id, results)

    else:
        # Run single experiment
        from experiments.experiment_configs import get_experiment
        exp_config = get_experiment(args.experiment)

        if args.experiment == "exp7":
            exp_queries = [q for q in queries_list if len(q.cloud_providers) > 1]
        else:
            exp_queries = queries_list

        results = runner.run_experiment(
            exp_config, exp_queries,
            resume=resume,
            max_queries=max_queries,
        )
        runner.print_experiment_summary(args.experiment, results)

    elapsed = time.time() - start_time
    logger.info("\nTotal benchmark time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    # Auto-export
    logger.info("Exporting results...")
    from src.evaluation.results_exporter import ResultsExporter
    exporter = ResultsExporter(results_dir=args.results_dir)
    exporter.export_all()

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
