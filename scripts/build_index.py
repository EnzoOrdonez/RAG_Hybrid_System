"""
Build search indices from chunks.

Usage:
    python scripts/build_index.py                                         # Build all (default: adaptive/500/bge-large)
    python scripts/build_index.py --embedding bge-large --chunker adaptive --size 500
    python scripts/build_index.py --only-bm25                             # Build BM25 index only
    python scripts/build_index.py --only-dense                            # Build dense (FAISS) index only
    python scripts/build_index.py --force                                 # Force re-embedding (ignore cache)
    python scripts/build_index.py --stats                                 # Show index stats only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from src.embedding.embedding_manager import EmbeddingManager
from src.embedding.index.faiss_index import FaissIndex
from src.embedding.index.bm25_index import BM25Index
from src.embedding.index.hybrid_index import HybridIndex

console = Console()
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load main config file."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(chunks_dir: str, strategy: str, size: int) -> list:
    """Load all chunk JSON files for a given strategy/size."""
    chunk_path = Path(chunks_dir) / strategy / f"size_{size}"
    if not chunk_path.exists():
        console.print(f"[red]Chunks directory not found: {chunk_path}[/red]")
        console.print(f"[yellow]Available strategies: {[d.name for d in Path(chunks_dir).iterdir() if d.is_dir()]}[/yellow]")
        return []

    chunks = []
    json_files = sorted(chunk_path.glob("*.json"))
    console.print(f"Loading {len(json_files)} chunk files from {chunk_path}...")

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            chunks.append(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Failed to load %s: %s", jf, e)

    console.print(f"[green]Loaded {len(chunks)} chunks[/green]")
    return chunks


def build_full_index(args, config):
    """Build both FAISS and BM25 indices (hybrid)."""
    chunks_dir = str(project_root / config["paths"]["chunks_data"])
    indices_dir = str(project_root / config["paths"]["indices_data"])

    # Load chunks
    chunks = load_chunks(chunks_dir, args.chunker, args.size)
    if not chunks:
        return

    # Initialize embedding manager
    emb_config = config.get("embedding", {})
    batch_size = emb_config.get("batch_size", 64)

    console.print(f"\n[bold cyan]Embedding Model:[/bold cyan] {args.embedding}")
    console.print(f"[bold cyan]Chunking Strategy:[/bold cyan] {args.chunker}")
    console.print(f"[bold cyan]Chunk Size:[/bold cyan] {args.size}")
    console.print(f"[bold cyan]Total Chunks:[/bold cyan] {len(chunks)}")
    console.print(f"[bold cyan]Batch Size:[/bold cyan] {batch_size}")

    embedding_manager = EmbeddingManager(
        model_name=args.embedding,
        cache_dir=str(project_root / emb_config.get("cache_dir", "data/embeddings")),
        batch_size=batch_size,
    )

    # Build hybrid index
    ret_config = config.get("retrieval", {})
    bm25_config = ret_config.get("bm25", {})

    hybrid_index = HybridIndex(
        embedding_manager=embedding_manager,
        bm25_k1=bm25_config.get("k1", 1.2),
        bm25_b=bm25_config.get("b", 0.75),
        indices_dir=indices_dir,
    )

    console.print("\n[bold yellow]Building hybrid index (FAISS + BM25)...[/bold yellow]")
    start = time.time()

    hybrid_index.build(
        chunks=chunks,
        chunk_strategy=args.chunker,
        chunk_size=args.size,
        force_reembed=args.force,
    )

    elapsed = time.time() - start
    console.print(f"[green]Hybrid index built in {elapsed:.1f}s[/green]")

    # Save indices
    console.print("\n[bold yellow]Saving indices to disk...[/bold yellow]")
    hybrid_index.save(chunk_strategy=args.chunker, chunk_size=args.size)
    console.print("[green]Indices saved![/green]")

    # Show stats
    show_index_stats(hybrid_index, embedding_manager, args)


def build_bm25_only(args, config):
    """Build only the BM25 index (no embeddings needed)."""
    chunks_dir = str(project_root / config["paths"]["chunks_data"])
    indices_dir = Path(project_root / config["paths"]["indices_data"])
    indices_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_dir, args.chunker, args.size)
    if not chunks:
        return

    ret_config = config.get("retrieval", {})
    bm25_config = ret_config.get("bm25", {})

    bm25_index = BM25Index(
        k1=bm25_config.get("k1", 1.2),
        b=bm25_config.get("b", 0.75),
    )

    console.print("\n[bold yellow]Building BM25 index only...[/bold yellow]")
    start = time.time()

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]
    bm25_index.build_index(texts, chunk_ids)

    elapsed = time.time() - start
    console.print(f"[green]BM25 index built in {elapsed:.1f}s[/green]")

    # Save
    save_path = str(indices_dir / f"bm25_{args.chunker}_{args.size}.pkl")
    bm25_index.save(save_path)
    console.print(f"[green]BM25 index saved to {save_path}[/green]")

    # Stats
    stats = bm25_index.get_stats()
    console.print(f"\n[bold]BM25 Stats:[/bold]")
    console.print(f"  Documents: {stats['total_documents']}")
    console.print(f"  Avg doc length: {stats['avg_doc_length']:.1f} tokens")


def build_dense_only(args, config):
    """Build only the FAISS dense index."""
    chunks_dir = str(project_root / config["paths"]["chunks_data"])
    indices_dir = Path(project_root / config["paths"]["indices_data"])
    indices_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_dir, args.chunker, args.size)
    if not chunks:
        return

    emb_config = config.get("embedding", {})
    batch_size = emb_config.get("batch_size", 64)

    embedding_manager = EmbeddingManager(
        model_name=args.embedding,
        cache_dir=str(project_root / emb_config.get("cache_dir", "data/embeddings")),
        batch_size=batch_size,
    )

    console.print(f"\n[bold yellow]Building dense (FAISS) index only with {args.embedding}...[/bold yellow]")
    start = time.time()

    texts = [c["text"] for c in chunks]
    chunk_ids = [c["chunk_id"] for c in chunks]

    # Embed
    embeddings, cached_ids = embedding_manager.embed_and_cache(
        texts, chunk_ids, args.chunker, args.size, force=args.force
    )

    # Build FAISS
    faiss_index = FaissIndex(dimension=embedding_manager.get_dimension())
    faiss_index.build_index(embeddings, cached_ids)

    elapsed = time.time() - start
    console.print(f"[green]Dense index built in {elapsed:.1f}s[/green]")

    # Save
    prefix = f"{args.embedding}_{args.chunker}_{args.size}"
    save_path = str(indices_dir / f"faiss_{prefix}.index")
    faiss_index.save(save_path)
    console.print(f"[green]FAISS index saved to {save_path}[/green]")

    # Stats
    stats = faiss_index.get_stats()
    console.print(f"\n[bold]FAISS Stats:[/bold]")
    console.print(f"  Total vectors: {stats['total_vectors']}")
    console.print(f"  Dimension: {stats['dimension']}")
    console.print(f"  Index type: {stats['index_type']}")

    # Embedding stats
    emb_stats = embedding_manager.get_stats()
    console.print(f"\n[bold]Embedding Stats:[/bold]")
    console.print(f"  Model: {emb_stats['model']}")
    console.print(f"  Dimension: {emb_stats['dimension']}")
    for cached in emb_stats.get("cached", []):
        console.print(f"  Cache: {cached['file']} ({cached['size_mb']:.1f} MB, shape={cached['shape']})")


def show_index_stats(hybrid_index=None, embedding_manager=None, args=None):
    """Display comprehensive index statistics."""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  INDEX STATISTICS[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    if hybrid_index:
        stats = hybrid_index.get_stats()

        # FAISS table
        faiss_stats = stats.get("faiss", {})
        table = Table(title="FAISS (Dense) Index")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total vectors", str(faiss_stats.get("total_vectors", "N/A")))
        table.add_row("Dimension", str(faiss_stats.get("dimension", "N/A")))
        table.add_row("Index type", str(faiss_stats.get("index_type", "N/A")))
        console.print(table)

        # BM25 table
        bm25_stats = stats.get("bm25", {})
        table = Table(title="BM25 (Lexical) Index")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Documents", str(bm25_stats.get("total_documents", "N/A")))
        table.add_row("Avg doc length", f"{bm25_stats.get('avg_doc_length', 0):.1f} tokens")
        console.print(table)

        console.print(f"  Chunk map size: {stats.get('chunk_map_size', 0)}")

    if embedding_manager:
        emb_stats = embedding_manager.get_stats()
        table = Table(title="Embedding Cache")
        table.add_column("File", style="cyan")
        table.add_column("Shape", style="green")
        table.add_column("Size (MB)", style="yellow")
        for cached in emb_stats.get("cached", []):
            table.add_row(cached["file"], str(cached["shape"]), f"{cached['size_mb']:.1f}")
        console.print(table)


def show_stats_only(args, config):
    """Show stats about existing indices without building."""
    indices_dir = Path(project_root / config["paths"]["indices_data"])
    embeddings_dir = Path(project_root / config["paths"]["embeddings_data"])

    console.print("\n[bold magenta]Existing Index Files:[/bold magenta]")

    # FAISS indices
    faiss_files = sorted(indices_dir.glob("faiss_*.index"))
    if faiss_files:
        table = Table(title="FAISS Index Files")
        table.add_column("File", style="cyan")
        table.add_column("Size (MB)", style="green")
        for f in faiss_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            table.add_row(f.name, f"{size_mb:.1f}")
        console.print(table)
    else:
        console.print("[yellow]  No FAISS index files found.[/yellow]")

    # BM25 indices
    bm25_files = sorted(indices_dir.glob("bm25_*.pkl"))
    if bm25_files:
        table = Table(title="BM25 Index Files")
        table.add_column("File", style="cyan")
        table.add_column("Size (MB)", style="green")
        for f in bm25_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            table.add_row(f.name, f"{size_mb:.1f}")
        console.print(table)
    else:
        console.print("[yellow]  No BM25 index files found.[/yellow]")

    # Chunk maps
    map_files = sorted(indices_dir.glob("chunk_map_*.json"))
    if map_files:
        table = Table(title="Chunk Map Files")
        table.add_column("File", style="cyan")
        table.add_column("Size (MB)", style="green")
        for f in map_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            table.add_row(f.name, f"{size_mb:.1f}")
        console.print(table)

    # Embeddings cache
    emb_files = sorted(embeddings_dir.glob("*.npy")) if embeddings_dir.exists() else []
    if emb_files:
        table = Table(title="Embedding Cache Files")
        table.add_column("File", style="cyan")
        table.add_column("Shape", style="green")
        table.add_column("Size (MB)", style="yellow")
        for f in emb_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            try:
                arr = np.load(f)
                shape = str(arr.shape)
            except Exception:
                shape = "?"
            table.add_row(f.name, shape, f"{size_mb:.1f}")
        console.print(table)
    else:
        console.print("[yellow]  No cached embedding files found.[/yellow]")

    # Provider distribution from chunk maps
    for mf in map_files:
        try:
            chunk_map = json.loads(mf.read_text(encoding="utf-8"))
            providers = {}
            for cid, cdata in chunk_map.items():
                prov = cdata.get("cloud_provider", "unknown")
                providers[prov] = providers.get(prov, 0) + 1

            table = Table(title=f"Provider Distribution ({mf.name})")
            table.add_column("Provider", style="cyan")
            table.add_column("Chunks", style="green")
            table.add_column("Percentage", style="yellow")
            total = sum(providers.values())
            for prov, count in sorted(providers.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total) * 100 if total > 0 else 0
                table.add_row(prov, str(count), f"{pct:.1f}%")
            table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "[bold]100.0%[/bold]")
            console.print(table)
        except Exception as e:
            logger.warning("Failed to read chunk map %s: %s", mf, e)


def main():
    parser = argparse.ArgumentParser(
        description="Build search indices (FAISS + BM25) from chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_index.py                                         # Build all (default config)
  python scripts/build_index.py --embedding bge-large --chunker adaptive --size 500
  python scripts/build_index.py --embedding all-MiniLM-L6-v2 --size 300 # Different model
  python scripts/build_index.py --only-bm25                             # BM25 only (no GPU needed)
  python scripts/build_index.py --only-dense                            # FAISS only
  python scripts/build_index.py --force                                 # Re-embed from scratch
  python scripts/build_index.py --stats                                 # View existing index stats
        """,
    )

    parser.add_argument(
        "--embedding",
        default=None,
        choices=["all-MiniLM-L6-v2", "bge-large", "e5-large", "instructor-large"],
        help="Embedding model to use (default: from config.yaml)",
    )
    parser.add_argument(
        "--chunker",
        default=None,
        choices=["fixed", "recursive", "semantic", "hierarchical", "adaptive"],
        help="Chunking strategy (default: adaptive)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Chunk size (default: 500)",
    )
    parser.add_argument(
        "--only-bm25",
        action="store_true",
        help="Build BM25 index only (no embedding needed)",
    )
    parser.add_argument(
        "--only-dense",
        action="store_true",
        help="Build dense (FAISS) index only",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-embedding even if cache exists",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics only (don't build)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_config()

    # Apply defaults from config
    emb_config = config.get("embedding", {})
    if args.embedding is None:
        args.embedding = emb_config.get("default_model", "bge-large")
    if args.chunker is None:
        args.chunker = "adaptive"
    if args.size is None:
        args.size = 500

    console.print("\n[bold blue]" + "=" * 60 + "[/bold blue]")
    console.print("[bold blue]  HYBRID RAG - Index Builder[/bold blue]")
    console.print("[bold blue]" + "=" * 60 + "[/bold blue]")

    start_total = time.time()

    if args.stats:
        show_stats_only(args, config)
    elif args.only_bm25:
        build_bm25_only(args, config)
    elif args.only_dense:
        build_dense_only(args, config)
    else:
        build_full_index(args, config)

    elapsed_total = time.time() - start_total
    console.print(f"\n[bold green]Total time: {elapsed_total:.1f}s[/bold green]")


if __name__ == "__main__":
    main()
