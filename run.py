"""
Main entry point for the Hybrid RAG System.
Provides CLI access to all pipeline stages.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
from rich.console import Console

console = Console()
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def cmd_download(args):
    """Download documentation."""
    from src.ingestion.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    stats = pipeline.run(
        providers=args.provider,
        max_services=args.services,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        console.print("\n[bold]Dry Run - What would be downloaded:[/bold]")
        for prov, info in stats["providers"].items():
            console.print(f"\n[cyan]{prov.upper()}[/cyan] ({info['source_type']}):")
            for svc in info["services"]:
                console.print(f"  - {svc['name']} [{svc['category']}]")
    else:
        console.print(f"\n[bold green]Download complete![/bold green]")
        console.print(f"Total documents: {stats['total_documents']}")


def cmd_preprocess(args):
    """Run preprocessing pipeline."""
    from src.preprocessing.text_cleaner import TextCleaner
    from src.preprocessing.terminology_normalizer import TerminologyNormalizer
    from src.preprocessing.metadata_extractor import MetadataExtractor
    from src.preprocessing.deduplicator import Deduplicator

    with open(project_root / "config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    input_dir = Path(args.input)

    console.print("[bold]Step 1/4: Cleaning text...[/bold]")
    cleaner = TextCleaner(config)
    cleaner.process_directory(input_dir)

    console.print("[bold]Step 2/4: Normalizing terminology...[/bold]")
    normalizer = TerminologyNormalizer()
    normalizer.process_directory(input_dir)

    console.print("[bold]Step 3/4: Extracting metadata...[/bold]")
    extractor = MetadataExtractor()
    extractor.process_directory(input_dir)

    if not args.skip_dedup:
        console.print("[bold]Step 4/4: Deduplicating...[/bold]")
        deduplicator = Deduplicator(config)
        deduplicator.process_directory(input_dir, report=True)

    console.print("[bold green]Preprocessing complete![/bold green]")


def cmd_chunk(args):
    """Run chunking with specified strategy."""
    input_dir = Path(args.input)
    output_base = Path(args.output)
    sizes = [int(s) for s in args.sizes.split(",")]
    overlap = args.overlap

    strategies = args.strategy if args.strategy != ["all"] else [
        "fixed", "recursive", "semantic", "hierarchical", "adaptive"
    ]

    for strategy in strategies:
        console.print(f"\n[bold]Chunking with strategy: {strategy}[/bold]")
        output_dir = output_base / strategy

        if strategy == "fixed":
            from src.chunking.fixed_chunker import process_directory
        elif strategy == "recursive":
            from src.chunking.recursive_chunker import process_directory
        elif strategy == "semantic":
            from src.chunking.semantic_chunker import process_directory
        elif strategy == "hierarchical":
            from src.chunking.hierarchical_chunker import process_directory
        elif strategy == "adaptive":
            from src.chunking.adaptive_chunker import process_directory
        else:
            console.print(f"[red]Unknown strategy: {strategy}[/red]")
            continue

        process_directory(input_dir, output_dir, sizes, overlap)
        console.print(f"  [green]Done: {output_dir}[/green]")

    console.print("\n[bold green]Chunking complete![/bold green]")


def cmd_stats(args):
    """Show corpus statistics."""
    from scripts.download_docs import show_stats
    show_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG System - Universidad de Lima"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    dl = subparsers.add_parser("download", help="Download documentation")
    dl.add_argument("--provider", type=str, nargs="*")
    dl.add_argument("--services", type=int, default=None)
    dl.add_argument("--dry-run", action="store_true")

    # Preprocess command
    pp = subparsers.add_parser("preprocess", help="Run preprocessing")
    pp.add_argument("--input", default="data/processed")
    pp.add_argument("--skip-dedup", action="store_true")

    # Chunk command
    ch = subparsers.add_parser("chunk", help="Run chunking")
    ch.add_argument("--input", default="data/processed")
    ch.add_argument("--output", default="data/chunks")
    ch.add_argument(
        "--strategy", nargs="+", default=["all"],
        choices=["all", "fixed", "recursive", "semantic", "hierarchical", "adaptive"],
    )
    ch.add_argument("--sizes", default="300,500,700")
    ch.add_argument("--overlap", type=int, default=50)

    # Stats command
    subparsers.add_parser("stats", help="Show corpus statistics")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                project_root / "logs" / "pipeline.log", encoding="utf-8"
            ),
        ],
    )

    if args.command == "download":
        cmd_download(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "chunk":
        cmd_chunk(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
