"""
Export experiment results to publication-ready formats.

Usage:
    python scripts/export_results.py
    python scripts/export_results.py --experiment exp8
    python scripts/export_results.py --format latex
    python scripts/export_results.py --format png
    python scripts/export_results.py --format csv
"""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export experiment results")
    parser.add_argument(
        "--experiment", "-e",
        type=str, default=None,
        help="Specific experiment ID to export (default: all)",
    )
    parser.add_argument(
        "--input",
        default="experiments/results",
        help="Results directory",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        default="all",
        choices=["png", "csv", "latex", "all"],
        help="Export format (default: all)",
    )
    args = parser.parse_args()

    from src.evaluation.results_exporter import ResultsExporter

    results_dir = str(project_root / args.input)
    output_dir = str(project_root / args.output)

    exporter = ResultsExporter(
        results_dir=results_dir,
        output_dir=output_dir,
    )

    if args.experiment:
        experiment_ids = [args.experiment]
    else:
        experiment_ids = None  # Auto-discover

    exporter.export_all(experiment_ids=experiment_ids)

    print(f"\nExport complete!")
    print(f"  Figures: {exporter.figures_dir}")
    print(f"  Tables:  {exporter.tables_dir}")
    print(f"  CSV:     {exporter.csv_dir}")


if __name__ == "__main__":
    main()
