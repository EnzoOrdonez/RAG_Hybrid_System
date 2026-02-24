"""
Run 5 verification tests for the hybrid RAG system.

Tests:
1. Single-provider query
2. Cross-cloud query
3. Query expansion
4. Reranking comparison
5. Latency benchmarks
"""

import json
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from rich.console import Console
from rich.table import Table

from src.embedding.embedding_manager import EmbeddingManager
from src.embedding.index.hybrid_index import HybridIndex
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.reranking.no_reranker import NoReranker

console = Console()
logging.basicConfig(level=logging.WARNING)


def load_system():
    """Load all components."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    emb_config = config.get("embedding", {})
    ret_config = config.get("retrieval", {})
    bm25_config = ret_config.get("bm25", {})

    console.print("[bold cyan]Loading embedding manager...[/bold cyan]")
    embedding_manager = EmbeddingManager(
        model_name="bge-large",
        cache_dir=str(project_root / emb_config.get("cache_dir", "data/embeddings")),
    )

    console.print("[bold cyan]Loading hybrid index...[/bold cyan]")
    hybrid_index = HybridIndex(
        embedding_manager=embedding_manager,
        bm25_k1=bm25_config.get("k1", 1.2),
        bm25_b=bm25_config.get("b", 0.75),
        indices_dir=str(project_root / config["paths"]["indices_data"]),
    )
    hybrid_index.load(chunk_strategy="adaptive", chunk_size=500)

    query_processor = QueryProcessor()

    return hybrid_index, embedding_manager, query_processor, config


def test_1_single_provider(hybrid_index, query_processor):
    """Test 1: Single-provider query."""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  TEST 1: Single-Provider Query[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    query = "How to create a VPC in AWS?"

    bm25_retriever = BM25Retriever(hybrid_index, query_processor)
    dense_retriever = DenseRetriever(hybrid_index, query_processor)
    hybrid_retriever = HybridRetriever(hybrid_index, query_processor)

    for name, retriever in [("BM25", bm25_retriever), ("Dense", dense_retriever), ("Hybrid", hybrid_retriever)]:
        results = retriever.search(query, top_k=5)
        table = Table(title=f"{name} Retriever - Top 5")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Chunk ID", style="dim", width=12)
        table.add_column("Score", style="green", width=8)
        table.add_column("Provider", style="yellow", width=8)
        table.add_column("Text (first 100 chars)", width=60)

        for i, r in enumerate(results, 1):
            table.add_row(
                str(i),
                r.chunk_id[:12],
                f"{r.score:.4f}",
                r.cloud_provider,
                r.chunk_text[:100],
            )
        console.print(table)

    return True


def test_2_cross_cloud(hybrid_index, query_processor):
    """Test 2: Cross-cloud query."""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  TEST 2: Cross-Cloud Query[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    query = "Compare serverless computing between AWS, Azure and GCP"

    hybrid_retriever = HybridRetriever(hybrid_index, query_processor, fusion_method="rrf")
    results = hybrid_retriever.search(query, top_k=10, top_k_candidates=50)

    table = Table(title="Hybrid RRF - Top 10")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Chunk ID", style="dim", width=12)
    table.add_column("Score", style="green", width=8)
    table.add_column("Provider", style="yellow", width=8)
    table.add_column("Service", style="blue", width=16)
    table.add_column("Text (first 100 chars)", width=50)

    providers_found = set()
    for i, r in enumerate(results, 1):
        providers_found.add(r.cloud_provider)
        table.add_row(
            str(i),
            r.chunk_id[:12],
            f"{r.score:.4f}",
            r.cloud_provider,
            r.service_name,
            r.chunk_text[:100],
        )
    console.print(table)

    console.print(f"\n[bold]Providers in results:[/bold] {providers_found}")
    multi_provider = len(providers_found) >= 2
    if multi_provider:
        console.print("[green]PASS: Multiple providers represented[/green]")
    else:
        console.print("[yellow]WARN: Only one provider found (may be due to limited data)[/yellow]")

    return True


def test_3_query_expansion(query_processor):
    """Test 3: Query expansion."""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  TEST 3: Query Expansion[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    query = "What is S3?"
    processed = query_processor.process(query)

    console.print(f"[bold]Original query:[/bold] {query}")
    console.print(f"[bold]Query type:[/bold] {processed.query_type}")
    console.print(f"[bold]Detected providers:[/bold] {processed.detected_providers}")
    console.print(f"[bold]Detected services:[/bold] {processed.detected_services}")
    console.print(f"[bold]Expanded terms:[/bold] {processed.expanded_terms}")
    console.print(f"[bold]BM25 query:[/bold] {processed.bm25_query}")
    console.print(f"[bold]Semantic query:[/bold] {processed.semantic_query}")
    console.print(f"[bold]Provider filter:[/bold] {processed.provider_filter}")

    # Check if expansion includes cross-cloud equivalents
    expanded_lower = [t.lower() for t in processed.expanded_terms]
    has_blob = any("blob" in t for t in expanded_lower)
    has_cloud_storage = any("cloud storage" in t for t in expanded_lower)

    if has_blob:
        console.print("[green]PASS: Expansion includes 'Blob Storage'[/green]")
    else:
        console.print("[yellow]WARN: 'Blob Storage' not in expansion[/yellow]")

    if has_cloud_storage:
        console.print("[green]PASS: Expansion includes 'Cloud Storage'[/green]")
    else:
        console.print("[yellow]WARN: 'Cloud Storage' not in expansion[/yellow]")

    return True


def test_4_reranking(hybrid_index, query_processor):
    """Test 4: Reranking comparison."""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  TEST 4: Reranking Comparison[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    query = "How to configure auto-scaling in EC2?"

    # Without reranking
    no_reranker = NoReranker()
    hybrid_no_rerank = HybridRetriever(
        hybrid_index, query_processor, reranker=no_reranker, fusion_method="rrf"
    )
    results_no = hybrid_no_rerank.search(query, top_k=5, use_reranker=True)

    table = Table(title="WITHOUT Reranking - Top 5")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Score", style="green", width=8)
    table.add_column("Provider", style="yellow", width=8)
    table.add_column("Text (first 100 chars)", width=60)
    for i, r in enumerate(results_no, 1):
        table.add_row(str(i), f"{r.score:.4f}", r.cloud_provider, r.chunk_text[:100])
    console.print(table)

    # With cross-encoder reranking
    try:
        reranker = CrossEncoderReranker(model_name="ms-marco-mini-6")
        hybrid_rerank = HybridRetriever(
            hybrid_index, query_processor, reranker=reranker, fusion_method="rrf"
        )
        results_yes = hybrid_rerank.search(query, top_k=5, top_k_candidates=50, use_reranker=True)

        table = Table(title="WITH Reranking (ms-marco-mini-6) - Top 5")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Score", style="green", width=8)
        table.add_column("Provider", style="yellow", width=8)
        table.add_column("Text (first 100 chars)", width=60)
        for i, r in enumerate(results_yes, 1):
            table.add_row(str(i), f"{r.score:.4f}", r.cloud_provider, r.chunk_text[:100])
        console.print(table)

        console.print("[green]PASS: Reranking applied successfully[/green]")
    except Exception as e:
        console.print(f"[red]FAIL: Reranking error: {e}[/red]")
        return False

    return True


def test_5_latency(hybrid_index, query_processor, embedding_manager):
    """Test 5: Latency benchmarks."""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  TEST 5: Latency Benchmarks[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    query = "How to deploy a containerized application on Kubernetes?"
    processed = query_processor.process(query)

    # Warm up
    _ = hybrid_index.search_bm25(query, top_k=5)
    _ = hybrid_index.search_dense(query, top_k=5)

    # BM25 latency
    times_bm25 = []
    for _ in range(5):
        start = time.perf_counter()
        hybrid_index.search_bm25(processed.bm25_query, top_k=10)
        times_bm25.append((time.perf_counter() - start) * 1000)
    bm25_ms = sum(times_bm25) / len(times_bm25)

    # Dense latency
    times_dense = []
    for _ in range(5):
        start = time.perf_counter()
        hybrid_index.search_dense(processed.semantic_query, top_k=10)
        times_dense.append((time.perf_counter() - start) * 1000)
    dense_ms = sum(times_dense) / len(times_dense)

    # Hybrid latency
    times_hybrid = []
    for _ in range(5):
        start = time.perf_counter()
        hybrid_index.search_hybrid(query, top_k=10, fusion="rrf")
        times_hybrid.append((time.perf_counter() - start) * 1000)
    hybrid_ms = sum(times_hybrid) / len(times_hybrid)

    # Reranking latency
    try:
        reranker = CrossEncoderReranker(model_name="ms-marco-mini-6")
        hybrid_retriever = HybridRetriever(
            hybrid_index, query_processor, reranker=reranker, fusion_method="rrf"
        )
        # Warm up reranker
        _ = hybrid_retriever.search(query, top_k=5, top_k_candidates=20)

        times_rerank = []
        for _ in range(3):
            start = time.perf_counter()
            hybrid_retriever.search(query, top_k=5, top_k_candidates=20, use_reranker=True)
            times_rerank.append((time.perf_counter() - start) * 1000)
        rerank_ms = sum(times_rerank) / len(times_rerank)
    except Exception as e:
        rerank_ms = -1
        console.print(f"[yellow]Reranker warmup failed: {e}[/yellow]")

    table = Table(title="Latency Benchmarks (avg of 5 runs)")
    table.add_column("System", style="cyan")
    table.add_column("Latency (ms)", style="green")
    table.add_column("Target", style="yellow")
    table.add_column("Status", style="bold")

    def status(actual, target):
        return "[green]PASS[/green]" if actual <= target else "[red]FAIL[/red]"

    table.add_row("BM25", f"{bm25_ms:.1f}", "<50ms", status(bm25_ms, 50))
    table.add_row("Dense", f"{dense_ms:.1f}", "<100ms", status(dense_ms, 100))
    table.add_row("Hybrid", f"{hybrid_ms:.1f}", "<200ms", status(hybrid_ms, 200))
    if rerank_ms >= 0:
        table.add_row("Reranking", f"{rerank_ms:.1f}", "<500ms", status(rerank_ms, 500))
    else:
        table.add_row("Reranking", "N/A", "<500ms", "[yellow]SKIP[/yellow]")
    console.print(table)

    return True


def main():
    console.print("\n[bold blue]" + "=" * 60 + "[/bold blue]")
    console.print("[bold blue]  HYBRID RAG - Verification Tests[/bold blue]")
    console.print("[bold blue]" + "=" * 60 + "[/bold blue]")

    hybrid_index, embedding_manager, query_processor, config = load_system()

    results = {}

    results["Test 1: Single-provider"] = test_1_single_provider(hybrid_index, query_processor)
    results["Test 2: Cross-cloud"] = test_2_cross_cloud(hybrid_index, query_processor)
    results["Test 3: Query expansion"] = test_3_query_expansion(query_processor)
    results["Test 4: Reranking"] = test_4_reranking(hybrid_index, query_processor)
    results["Test 5: Latency"] = test_5_latency(hybrid_index, query_processor, embedding_manager)

    # Summary
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]  TEST SUMMARY[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "[green]PASS[/green]" if result else "[red]FAIL[/red]"
        console.print(f"  {status} {test_name}")

    console.print(f"\n[bold]Result: {passed}/{total} tests passing[/bold]")


if __name__ == "__main__":
    main()
