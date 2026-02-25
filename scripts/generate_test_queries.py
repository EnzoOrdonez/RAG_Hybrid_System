"""
Generate test queries for thesis evaluation.

Usage:
    python scripts/generate_test_queries.py --output data/evaluation/test_queries.json --count 200
    python scripts/generate_test_queries.py --cross-cloud-only
    python scripts/generate_test_queries.py --verify data/evaluation/test_queries.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.test_queries import (
    generate_all_queries,
    generate_cross_cloud_queries,
    load_queries,
    save_queries,
    verify_queries,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate test queries for evaluation")
    parser.add_argument(
        "--output", "-o",
        default="data/evaluation/test_queries.json",
        help="Output path for queries JSON",
    )
    parser.add_argument(
        "--count", "-n",
        type=int, default=200,
        help="Number of queries to generate (default: 200)",
    )
    parser.add_argument(
        "--cross-cloud-only",
        action="store_true",
        help="Generate only cross-cloud queries",
    )
    parser.add_argument(
        "--verify",
        type=str, default=None,
        help="Verify an existing query file instead of generating",
    )
    parser.add_argument(
        "--with-chunks",
        action="store_true",
        help="Find relevant chunk IDs (requires loaded index, slow)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Verify mode
    if args.verify:
        logger.info("Verifying query file: %s", args.verify)
        queries = load_queries(args.verify)
        stats = verify_queries(queries)
        print(json.dumps(stats, indent=2))
        return

    # Load index if needed
    hybrid_index = None
    if args.with_chunks:
        logger.info("Loading hybrid index for chunk retrieval...")
        try:
            from src.pipeline.rag_pipeline import load_hybrid_index
            hybrid_index = load_hybrid_index()
            logger.info("Index loaded successfully")
        except Exception as e:
            logger.warning("Could not load index: %s (continuing without chunk IDs)", e)

    # Generate queries
    if args.cross_cloud_only:
        logger.info("Generating cross-cloud queries only...")
        queries = generate_cross_cloud_queries(hybrid_index)
    else:
        logger.info("Generating %d queries (seed=%d)...", args.count, args.seed)
        queries = generate_all_queries(
            count=args.count,
            hybrid_index=hybrid_index,
            seed=args.seed,
        )

    # Save
    output_path = str(project_root / args.output)
    save_queries(queries, output_path)

    # Print stats
    stats = verify_queries(queries)
    print(f"\nGenerated {stats['total']} queries:")
    print(f"  By provider: {stats['by_provider']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  By difficulty: {stats['by_difficulty']}")
    print(f"  Cross-cloud: {stats['cross_cloud']}")
    print(f"  With ground truth: {stats['with_ground_truth']}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
