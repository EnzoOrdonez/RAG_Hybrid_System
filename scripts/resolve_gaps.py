"""
Resolve all Phase 2 data gaps:
  Gap 1: Re-crawl AWS (all 12 services)
  Gap 2: Re-crawl Azure (all 10 services)
  Gap 3: Re-crawl CNCF (all glossary terms)
  Gap 4: Generate chunks for sizes 300, 500, 700 × 5 strategies
  Gap 5: Apply terminology normalizer to all chunks
  Gap 6: Rebuild indices

Usage:
  python scripts/resolve_gaps.py --crawl          # Gaps 1-3
  python scripts/resolve_gaps.py --chunk          # Gap 4
  python scripts/resolve_gaps.py --normalize      # Gap 5
  python scripts/resolve_gaps.py --rebuild-index  # Gap 6
  python scripts/resolve_gaps.py --all            # Everything
  python scripts/resolve_gaps.py --stats          # Show current stats
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table

console = Console(force_terminal=True)
logger = logging.getLogger(__name__)


def show_stats():
    """Show current corpus statistics."""
    console.print("\n[bold cyan]═══ Current Corpus Stats ═══[/bold cyan]")

    # Processed docs
    processed_dir = PROJECT_ROOT / "data" / "processed"
    table = Table(title="Processed Documents")
    table.add_column("Provider", style="cyan")
    table.add_column("Docs", justify="right")
    table.add_column("Services")

    for prov_dir in sorted(processed_dir.iterdir()):
        if not prov_dir.is_dir():
            continue
        services = Counter()
        for f in prov_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                services[data.get("service_name", "?")] += 1
            except:
                pass
        total = sum(services.values())
        svc_str = ", ".join(f"{s}={n}" for s, n in services.most_common())
        table.add_row(prov_dir.name, str(total), svc_str)

    console.print(table)

    # Chunks
    chunks_dir = PROJECT_ROOT / "data" / "chunks"
    if chunks_dir.exists():
        table2 = Table(title="Chunks by Strategy × Size")
        table2.add_column("Strategy", style="green")
        table2.add_column("size_300", justify="right")
        table2.add_column("size_500", justify="right")
        table2.add_column("size_700", justify="right")

        for strat_dir in sorted(chunks_dir.iterdir()):
            if not strat_dir.is_dir():
                continue
            counts = {}
            for size_dir in strat_dir.iterdir():
                if size_dir.is_dir():
                    n = sum(1 for _ in size_dir.rglob("*.json"))
                    counts[size_dir.name] = n
            table2.add_row(
                strat_dir.name,
                str(counts.get("size_300", 0)),
                str(counts.get("size_500", 0)),
                str(counts.get("size_700", 0)),
            )
        console.print(table2)


def run_crawlers():
    """Gaps 1-3: Re-crawl AWS, Azure, CNCF."""
    from src.ingestion.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline()

    # Crawl each gap provider
    for provider in ["aws", "azure", "cncf"]:
        console.print(f"\n[bold yellow]Crawling {provider.upper()}...[/bold yellow]")
        start = time.time()
        try:
            stats = pipeline.run(providers=[provider])
            elapsed = time.time() - start
            prov_stats = stats.get("providers", {}).get(provider, {})
            console.print(
                f"[green]OK {provider.upper()}: "
                f"{prov_stats.get('documents', 0)} docs in {elapsed:.1f}s[/green]"
            )
        except Exception as e:
            console.print(f"[red]FAIL {provider.upper()} failed: {e}[/red]")
            logger.exception("Crawler error for %s", provider)


def run_chunking():
    """Gap 4: Generate chunks for all strategies × all sizes."""
    import yaml

    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sizes = config.get("chunking", {}).get("sizes_to_evaluate", [300, 500, 700])
    overlap = config.get("chunking", {}).get("overlap", 50)
    strategies = ["adaptive", "fixed", "recursive", "semantic", "hierarchical"]

    # Load all processed documents
    from src.ingestion.ingestion_pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    documents = pipeline.load_processed_documents()
    console.print(f"\n[bold]Loaded {len(documents)} processed documents[/bold]")

    for strategy in strategies:
        for size in sizes:
            output_dir = PROJECT_ROOT / "data" / "chunks" / strategy / f"size_{size}"

            # Skip if already has chunks (unless we want to regenerate)
            existing = sum(1 for _ in output_dir.rglob("*.json")) if output_dir.exists() else 0
            if existing > 0:
                console.print(
                    f"  [dim]{strategy}/size_{size}: {existing} chunks exist, skipping[/dim]"
                )
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            console.print(
                f"\n[cyan]Chunking: {strategy} @ size={size}, overlap={overlap}[/cyan]"
            )

            start = time.time()
            chunker = _create_chunker(strategy, size, overlap)
            total_chunks = 0

            for doc in documents:
                try:
                    chunks = chunker.chunk_document(doc)
                    # Save chunks grouped by provider/service
                    for chunk in chunks:
                        chunk_path = output_dir / f"{chunk.chunk_id}.json"
                        chunk_path.write_text(
                            chunk.model_dump_json(indent=2),
                            encoding="utf-8",
                        )
                        total_chunks += 1
                except Exception as e:
                    logger.debug("Chunking error for %s: %s", doc.doc_id, e)

            elapsed = time.time() - start
            console.print(
                f"  [green]OK {strategy}/size_{size}: "
                f"{total_chunks} chunks in {elapsed:.1f}s[/green]"
            )


def _create_chunker(strategy: str, size: int, overlap: int):
    """Factory for chunking strategies."""
    if strategy == "adaptive":
        from src.chunking.adaptive_chunker import AdaptiveChunker
        return AdaptiveChunker(chunk_size=size, overlap=overlap)
    elif strategy == "fixed":
        from src.chunking.fixed_chunker import FixedChunker
        return FixedChunker(chunk_size=size, overlap=overlap)
    elif strategy == "recursive":
        from src.chunking.recursive_chunker import RecursiveChunker
        return RecursiveChunker(chunk_size=size, overlap=overlap)
    elif strategy == "semantic":
        from src.chunking.semantic_chunker import SemanticChunker
        return SemanticChunker(chunk_size=size, overlap=overlap)
    elif strategy == "hierarchical":
        from src.chunking.hierarchical_chunker import HierarchicalChunker
        return HierarchicalChunker(chunk_size=size, overlap=overlap)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_normalization():
    """Gap 5: Apply terminology normalizer to all chunks."""
    from src.preprocessing.terminology_normalizer import TerminologyNormalizer
    from src.ingestion.doc_parser import Chunk

    normalizer = TerminologyNormalizer()
    chunks_dir = PROJECT_ROOT / "data" / "chunks"

    total_processed = 0
    total_enriched = 0

    for strategy_dir in sorted(chunks_dir.iterdir()):
        if not strategy_dir.is_dir():
            continue
        for size_dir in sorted(strategy_dir.iterdir()):
            if not size_dir.is_dir():
                continue

            json_files = list(size_dir.rglob("*.json"))
            if not json_files:
                continue

            console.print(
                f"[cyan]Normalizing {strategy_dir.name}/{size_dir.name}: "
                f"{len(json_files)} chunks[/cyan]"
            )

            for jf in json_files:
                try:
                    data = json.loads(jf.read_text(encoding="utf-8"))
                    # Check if already normalized
                    if data.get("normalized_terms"):
                        total_processed += 1
                        continue

                    # Create Chunk object and normalize
                    chunk = Chunk(**data)
                    normalizer.normalize_chunk(chunk)

                    # Update data
                    data["normalized_terms"] = chunk.normalized_terms
                    data["detected_siglas"] = chunk.detected_siglas
                    data["cross_cloud_equivalences"] = chunk.cross_cloud_equivalences

                    jf.write_text(
                        json.dumps(data, indent=2, default=str),
                        encoding="utf-8",
                    )
                    total_processed += 1
                    if chunk.normalized_terms:
                        total_enriched += 1
                except Exception as e:
                    logger.debug("Normalization error for %s: %s", jf, e)

    console.print(
        f"\n[green]OK Normalized {total_processed} chunks, "
        f"{total_enriched} with terms detected[/green]"
    )

    # Save inverted index
    index_path = PROJECT_ROOT / "data" / "indices" / "terminology_index.json"
    normalizer.save_inverted_index(index_path)
    console.print(f"[green]OK Inverted index saved to {index_path}[/green]")


def rebuild_indices():
    """Gap 6: Rebuild FAISS + BM25 indices from expanded data."""
    import subprocess

    console.print("\n[bold yellow]Rebuilding indices...[/bold yellow]")

    # Rebuild hybrid index for adaptive/500 (primary configuration)
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "build_index.py"),
        "--embedding", "bge-large",
        "--chunker", "adaptive",
        "--size", "500",
        "--force",
    ]
    console.print(f"[cyan]Running: {' '.join(cmd)}[/cyan]")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=3600)

    if result.returncode == 0:
        console.print("[green]OK Indices rebuilt successfully[/green]")
    else:
        console.print("[red]FAIL Index rebuild failed[/red]")


def main():
    parser = argparse.ArgumentParser(description="Resolve Phase 2 data gaps")
    parser.add_argument("--crawl", action="store_true", help="Gaps 1-3: Re-crawl AWS, Azure, CNCF")
    parser.add_argument("--chunk", action="store_true", help="Gap 4: Generate all chunk sizes")
    parser.add_argument("--normalize", action="store_true", help="Gap 5: Apply terminology normalizer")
    parser.add_argument("--rebuild-index", action="store_true", help="Gap 6: Rebuild indices")
    parser.add_argument("--all", action="store_true", help="Run all gaps")
    parser.add_argument("--stats", action="store_true", help="Show current stats")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.stats:
        show_stats()
        return

    if args.all:
        args.crawl = True
        args.chunk = True
        args.normalize = True
        args.rebuild_index = True

    if not any([args.crawl, args.chunk, args.normalize, args.rebuild_index]):
        parser.print_help()
        return

    start_total = time.time()

    if args.crawl:
        run_crawlers()

    if args.chunk:
        run_chunking()

    if args.normalize:
        run_normalization()

    if args.rebuild_index:
        rebuild_indices()

    elapsed = time.time() - start_total
    console.print(f"\n[bold green]All done in {elapsed:.1f}s[/bold green]")
    show_stats()


if __name__ == "__main__":
    main()
