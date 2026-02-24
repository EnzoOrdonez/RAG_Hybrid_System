"""
Main entry point for the Hybrid RAG System.
Provides CLI access to all pipeline stages.

Usage:
  # Phase 1 commands (ingestion/chunking)
  python run.py download --provider aws azure
  python run.py preprocess --input data/processed
  python run.py chunk --strategy adaptive --sizes 300,500,700

  # Phase 3 commands (RAG pipeline)
  python run.py --query "How to create a VPC in AWS?" --config hybrid
  python run.py --compare "What is the difference between Lambda and Azure Functions?"
  python run.py --health-check
  python run.py --interactive --config hybrid
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

console = Console()
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# ============================================================
# Phase 1 commands
# ============================================================

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


# ============================================================
# Phase 3 commands
# ============================================================

def _load_index():
    """Load the hybrid index."""
    from src.pipeline.rag_pipeline import load_hybrid_index
    console.print("[dim]Loading hybrid index...[/dim]")
    return load_hybrid_index()


def cmd_query(args):
    """Run a single query through a pipeline config."""
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.pipeline.pipeline_config import get_config

    config = get_config(args.config)
    hybrid_index = _load_index()

    pipeline = RAGPipeline(config=config, hybrid_index=hybrid_index)

    console.print(f"\n[bold cyan]Config: {config.name}[/bold cyan]")
    console.print(f"[bold]Question:[/bold] {args.query}\n")

    response = pipeline.query(args.query)
    _print_response(response)


def cmd_compare(args):
    """Compare the 3 systems with the same query."""
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.pipeline.pipeline_config import PIPELINE_CONFIGS

    hybrid_index = _load_index()

    console.print(f"\n[bold]Question:[/bold] {args.compare}")
    console.print("=" * 70)

    responses = {}
    for name, config in PIPELINE_CONFIGS.items():
        console.print(f"\n[bold cyan]━━━ {config.name} ━━━[/bold cyan]")
        pipeline = RAGPipeline(config=config, hybrid_index=hybrid_index)
        response = pipeline.query(args.compare)
        responses[name] = response
        _print_response(response, compact=True)

    # Comparison table
    _print_comparison_table(responses)


def cmd_health_check(args):
    """Verify all components are working."""
    console.print("\n[bold]═══ Health Check ═══[/bold]\n")
    all_ok = True

    # 1. Corpus
    console.print("[bold]1. Corpus[/bold]")
    chunks_dir = project_root / "data" / "chunks" / "adaptive" / "size_500"
    if chunks_dir.exists():
        n = sum(1 for _ in chunks_dir.rglob("*.json"))
        if n > 0:
            console.print(f"  [green]✓ {n} chunks in adaptive/500[/green]")
        else:
            console.print("  [red]✗ No chunks found[/red]")
            all_ok = False
    else:
        console.print("  [red]✗ Chunks directory not found[/red]")
        all_ok = False

    # 2. Indices
    console.print("[bold]2. Indices[/bold]")
    indices_dir = project_root / "data" / "indices"
    for name in ["faiss_bge-large_adaptive_500.index", "bm25_adaptive_500.pkl"]:
        path = indices_dir / name
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            console.print(f"  [green]✓ {name} ({size_mb:.1f} MB)[/green]")
        else:
            console.print(f"  [red]✗ {name} not found[/red]")
            all_ok = False

    # 3. Index load test
    console.print("[bold]3. Index load[/bold]")
    try:
        hybrid_index = _load_index()
        stats = hybrid_index.get_stats()
        console.print(f"  [green]✓ FAISS: {stats['faiss'].get('total_vectors', '?')} vectors[/green]")
        console.print(f"  [green]✓ BM25: {stats['bm25'].get('total_documents', '?')} docs[/green]")
        console.print(f"  [green]✓ Chunk map: {stats['chunk_map_size']} entries[/green]")
    except Exception as e:
        console.print(f"  [red]✗ Index load failed: {e}[/red]")
        all_ok = False

    # 4. LLM
    console.print("[bold]4. LLM (Ollama)[/bold]")
    try:
        from src.generation.llm_manager import LLMManager
        mgr = LLMManager(provider="ollama", model="llama3.1")
        if mgr.is_available():
            console.print(f"  [green]✓ Ollama available, llama3.1 model found[/green]")
        else:
            console.print(f"  [yellow]⚠ Ollama not running or llama3.1 not pulled[/yellow]")
            console.print(f"    Run: ollama serve && ollama pull llama3.1:8b-instruct-q4_K_M")
            # Not a hard failure - user might use API models
    except Exception as e:
        console.print(f"  [yellow]⚠ LLM check failed: {e}[/yellow]")

    # 5. NLI Model
    console.print("[bold]5. NLI Model[/bold]")
    try:
        from src.generation.hallucination_detector import HallucinationDetector
        hd = HallucinationDetector()
        _ = hd.nli_model
        if hd._nli_available:
            console.print(f"  [green]✓ NLI model loaded[/green]")
        else:
            console.print(f"  [yellow]⚠ NLI model unavailable (keyword fallback active)[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]⚠ NLI check failed: {e}[/yellow]")

    # Summary
    if all_ok:
        console.print(f"\n[bold green]✓ All core components healthy[/bold green]")
    else:
        console.print(f"\n[bold red]✗ Some components need attention[/bold red]")


def cmd_interactive(args):
    """Interactive chat mode."""
    from src.pipeline.rag_pipeline import RAGPipeline
    from src.pipeline.pipeline_config import get_config

    config = get_config(args.config)
    hybrid_index = _load_index()
    pipeline = RAGPipeline(config=config, hybrid_index=hybrid_index)

    console.print(f"\n[bold cyan]═══ Hybrid RAG Interactive Mode ═══[/bold cyan]")
    console.print(f"Config: {config.name}")
    console.print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("[bold]You> [/bold]" if sys.stdout.isatty() else "You> ")
        except (EOFError, KeyboardInterrupt):
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        response = pipeline.query(question)
        _print_response(response)
        console.print()

    console.print("[dim]Goodbye![/dim]")


# ============================================================
# Output helpers
# ============================================================

def _print_response(response, compact=False):
    """Print a RAG response."""
    # Answer
    if compact:
        answer_preview = response.answer[:300]
        if len(response.answer) > 300:
            answer_preview += "..."
        console.print(f"\n[white]{answer_preview}[/white]")
    else:
        console.print(f"\n[white]{response.answer}[/white]")

    # Sources
    if response.sources:
        console.print(f"\n[bold]Sources ({len(response.sources)}):[/bold]")
        for src in response.sources[:5]:
            console.print(f"  • {src.get('provider', '?')}/{src.get('service', '?')}/{src.get('section', '')}")

    # Metadata
    console.print(f"\n[dim]Confidence: {response.confidence}[/dim]", end="")

    if response.hallucination_report:
        hr = response.hallucination_report
        console.print(
            f"[dim] | Faithfulness: {hr.faithfulness_score:.2f} "
            f"({hr.supported_claims}/{hr.total_claims} claims) "
            f"| Rubric: {hr.suggested_rubric}/5[/dim]",
            end="",
        )

    if response.llm_response:
        lr = response.llm_response
        console.print(
            f"[dim] | Cache: {lr.from_cache} "
            f"| Tokens: {lr.tokens_input}→{lr.tokens_output}[/dim]",
            end="",
        )

    console.print()  # newline

    # Latency
    lat = response.latency
    console.print(
        f"[dim]Latency: total={lat.total_ms:.0f}ms "
        f"(query={lat.query_processing_ms:.0f}, "
        f"retrieval={lat.retrieval_ms:.0f}, "
        f"rerank={lat.reranking_ms:.0f}, "
        f"gen={lat.generation_ms:.0f}, "
        f"hall={lat.hallucination_check_ms:.0f})[/dim]"
    )

    if response.error:
        console.print(f"[red]Error: {response.error}[/red]")


def _print_comparison_table(responses: dict):
    """Print comparison table across the 3 systems."""
    console.print("\n[bold]═══ Latency Comparison ═══[/bold]")

    table = Table()
    table.add_column("Stage", style="cyan")
    for name in responses:
        table.add_column(responses[name].config_name, justify="right")

    stages = [
        ("Query processing", "query_processing_ms"),
        ("Retrieval", "retrieval_ms"),
        ("Reranking", "reranking_ms"),
        ("Generation", "generation_ms"),
        ("Hallucination check", "hallucination_check_ms"),
        ("TOTAL", "total_ms"),
    ]

    for label, field in stages:
        row = [label]
        for name in responses:
            lat = responses[name].latency
            val = getattr(lat, field, 0.0)
            if field == "reranking_ms" and val == 0.0 and "hybrid" not in name:
                row.append("N/A")
            else:
                row.append(f"{val:.0f} ms")
        table.add_row(*row)

    console.print(table)

    # Confidence comparison
    console.print("\n[bold]Confidence:[/bold]")
    for name, resp in responses.items():
        faith = ""
        if resp.hallucination_report:
            hr = resp.hallucination_report
            faith = f" (faithfulness={hr.faithfulness_score:.2f}, rubric={hr.suggested_rubric}/5)"
        console.print(f"  {resp.config_name}: {resp.confidence}{faith}")


# ============================================================
# Main parser
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG System - Tesis Universidad de Lima",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --query "What is S3?" --config hybrid
  python run.py --compare "Compare serverless in AWS vs Azure vs GCP"
  python run.py --health-check
  python run.py --interactive --config hybrid
  python run.py download --provider aws azure
  python run.py chunk --strategy adaptive --sizes 300,500,700
""",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging"
    )

    # Phase 3: RAG commands (top-level for convenience)
    parser.add_argument(
        "--query", type=str, help="Run a single query"
    )
    parser.add_argument(
        "--config", type=str, default="hybrid",
        choices=["lexical", "semantic", "hybrid"],
        help="Pipeline config (default: hybrid)",
    )
    parser.add_argument(
        "--compare", type=str,
        help="Compare all 3 systems with the same query",
    )
    parser.add_argument(
        "--health-check", action="store_true",
        help="Check all components",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive chat mode",
    )

    # Phase 1: Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Phase 1 commands")

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

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8"),
        ],
    )

    # Route to appropriate handler
    if args.health_check:
        cmd_health_check(args)
    elif args.query:
        cmd_query(args)
    elif args.compare:
        cmd_compare(args)
    elif args.interactive:
        cmd_interactive(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "chunk":
        cmd_chunk(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
