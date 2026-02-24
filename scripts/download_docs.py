"""
Download documentation from all 5 sources.
Usage:
  python scripts/download_docs.py                     # All sources
  python scripts/download_docs.py --provider aws      # Single provider
  python scripts/download_docs.py --services 5        # 5 services per provider
  python scripts/download_docs.py --dry-run           # Preview only
  python scripts/download_docs.py --stats             # Show corpus stats
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from rich.console import Console
from rich.table import Table as RichTable

console = Console()


def load_config():
    with open(project_root / "config/config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(project_root / "config/cloud_services.yaml", encoding="utf-8") as f:
        cloud_config = yaml.safe_load(f)
    return config, cloud_config


def show_stats():
    """Show statistics about the current corpus."""
    stats_path = project_root / "data" / "corpus_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    else:
        stats = {"providers": {}, "total_documents": 0}

    # Count files on disk
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    chunks_dir = project_root / "data" / "chunks"

    table = RichTable(title="Corpus Statistics")
    table.add_column("Provider", style="cyan")
    table.add_column("Raw Files", justify="right")
    table.add_column("Processed", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("With Code", justify="right")

    providers = ["aws", "azure", "gcp", "k8s", "cncf"]
    total_raw = 0
    total_processed = 0

    for prov in providers:
        raw_count = len(list((raw_dir / prov).rglob("*.json"))) if (raw_dir / prov).exists() else 0
        proc_count = len(list((processed_dir / prov).rglob("*.json"))) if (processed_dir / prov).exists() else 0
        # Also check 'kubernetes' as provider name
        if prov == "k8s":
            proc_count += len(list((processed_dir / "kubernetes").rglob("*.json"))) if (processed_dir / "kubernetes").exists() else 0

        prov_stats = stats.get("providers", {}).get(prov, {})
        words = prov_stats.get("total_words", "-")
        code = prov_stats.get("with_code", "-")

        table.add_row(prov.upper(), str(raw_count), str(proc_count), str(words), str(code))
        total_raw += raw_count
        total_processed += proc_count

    table.add_row("TOTAL", str(total_raw), str(total_processed), "-", "-", style="bold")
    console.print(table)

    # Chunks stats
    if chunks_dir.exists():
        console.print("\n[bold]Chunks:[/bold]")
        for strategy_dir in sorted(chunks_dir.iterdir()):
            if strategy_dir.is_dir():
                for size_dir in sorted(strategy_dir.iterdir()):
                    if size_dir.is_dir():
                        count = len(list(size_dir.glob("*.json")))
                        if count > 0:
                            console.print(
                                f"  {strategy_dir.name}/{size_dir.name}: {count} chunks"
                            )


def dry_run(providers, max_services):
    """Show what would be downloaded."""
    from src.ingestion.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    stats = pipeline.run(providers=providers, max_services=max_services, dry_run=True)

    table = RichTable(title="Dry Run - What Would Be Downloaded")
    table.add_column("Provider", style="cyan")
    table.add_column("Source Type", style="green")
    table.add_column("Services", justify="right")
    table.add_column("Details")

    for prov, info in stats["providers"].items():
        details = "\n".join(
            f"  {s['name']} [{s['category']}]" for s in info["services"]
        )
        table.add_row(
            prov.upper(),
            info["source_type"],
            str(info["services_count"]),
            details,
        )

    console.print(table)


def download(providers, max_services):
    """Run the full download pipeline."""
    from src.ingestion.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline()

    console.print(
        f"\n[bold green]Starting download...[/bold green]"
        f"\n  Providers: {providers or 'ALL'}"
        f"\n  Max services per provider: {max_services or 'ALL'}\n"
    )

    stats = pipeline.run(providers=providers, max_services=max_services)

    console.print(f"\n[bold green]Download Complete![/bold green]")
    console.print(f"Total documents: {stats['total_documents']}")
    for prov, info in stats["providers"].items():
        console.print(
            f"  {prov}: {info['documents']} docs, "
            f"{info['total_words']} words"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download cloud documentation for the RAG system"
    )
    parser.add_argument(
        "--provider", type=str, nargs="*",
        help="Providers to download (aws, azure, gcp, kubernetes, cncf)"
    )
    parser.add_argument(
        "--services", type=int, default=None,
        help="Max services per provider (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show what would be downloaded"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show corpus statistics"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose logging"
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                project_root / "logs" / "download.log", encoding="utf-8"
            ),
        ],
    )

    if args.stats:
        show_stats()
    elif args.dry_run:
        dry_run(args.provider, args.services)
    else:
        download(args.provider, args.services)


if __name__ == "__main__":
    main()
