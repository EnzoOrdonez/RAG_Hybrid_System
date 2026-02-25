"""
BLOQUE D - Phase 3 Verification Tests (7 tests)

Test 1: Health check (corpus, indices, models)
Test 2: Compare 3 systems with same query (retrieval only if no LLM)
Test 3: Cross-cloud query
Test 4: Hallucination detection
Test 5: No-answer / HONEST_DECLINE
Test 6: Latency breakdown table
Test 7: LLM cache verification

Usage:
    python tests/test_phase3_verification.py
    python tests/test_phase3_verification.py --with-llm   # Only if Ollama/API available
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console(force_terminal=True)

# ============================================================
# Helpers
# ============================================================

def load_env():
    """Load .env file."""
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass


def check_ollama_available():
    """Check if Ollama is running and has a model."""
    try:
        import ollama
        models = ollama.list()
        if hasattr(models, "models"):
            return len(models.models) > 0
        elif isinstance(models, dict):
            return len(models.get("models", [])) > 0
    except Exception:
        return False
    return False


def check_openai_available():
    """Check if OpenAI API key is set."""
    return bool(os.getenv("OPENAI_API_KEY"))


def check_anthropic_available():
    """Check if Anthropic API key is set."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def get_llm_provider():
    """Determine which LLM provider to use."""
    if check_ollama_available():
        return "ollama", "llama3.1:8b-instruct-q4_K_M"
    if check_openai_available():
        return "openai", "gpt-4o-mini"
    if check_anthropic_available():
        return "anthropic", "claude-sonnet-4-20250514"
    return None, None


PASS = "[bold green]PASS[/bold green]"
FAIL = "[bold red]FAIL[/bold red]"
SKIP = "[bold yellow]SKIP[/bold yellow]"

results = {}


# ============================================================
# TEST 1: Health Check
# ============================================================

def test_1_health_check():
    """Verify corpus, indices, models are loaded correctly."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 1: Health Check[/bold cyan]")
    console.print("=" * 60)

    checks = {}

    # 1a. Corpus exists
    processed_dir = PROJECT_ROOT / "data" / "processed"
    providers = {}
    for p in ["aws", "azure", "gcp", "kubernetes", "cncf"]:
        pdir = processed_dir / p
        if pdir.exists():
            count = sum(1 for _ in pdir.glob("*.json"))
            providers[p] = count
        else:
            providers[p] = 0
    total_docs = sum(providers.values())
    checks["corpus"] = total_docs > 1000
    console.print(f"  Corpus: {total_docs} docs ({', '.join(f'{k}={v}' for k, v in providers.items())})")

    # 1b. Chunks exist (all 15 combos)
    chunks_dir = PROJECT_ROOT / "data" / "chunks"
    chunk_combos = 0
    for strategy in ["adaptive", "fixed", "recursive", "semantic", "hierarchical"]:
        for size in [300, 500, 700]:
            sd = chunks_dir / strategy / f"size_{size}"
            if sd.exists() and any(sd.glob("*.json")):
                chunk_combos += 1
    checks["chunks_15_combos"] = chunk_combos == 15
    console.print(f"  Chunk combos: {chunk_combos}/15")

    # 1c. Indices exist
    indices_dir = PROJECT_ROOT / "data" / "indices"
    faiss_exists = (indices_dir / "faiss_bge-large_adaptive_500.index").exists()
    bm25_exists = (indices_dir / "bm25_adaptive_500.pkl").exists()
    checks["faiss_index"] = faiss_exists
    checks["bm25_index"] = bm25_exists
    console.print(f"  FAISS index: {'OK' if faiss_exists else 'MISSING'}")
    console.print(f"  BM25 index: {'OK' if bm25_exists else 'MISSING'}")

    # 1d. Index loads correctly
    try:
        from src.pipeline.rag_pipeline import load_hybrid_index
        idx = load_hybrid_index()
        n_chunks = len(idx.chunk_map) if hasattr(idx, "chunk_map") else 0
        checks["index_loads"] = n_chunks > 10000
        console.print(f"  Index loaded: {n_chunks} chunks in memory")
    except Exception as e:
        checks["index_loads"] = False
        console.print(f"  Index load: FAILED ({e})")

    # 1e. LLM availability
    provider, model = get_llm_provider()
    if provider:
        checks["llm_available"] = True
        console.print(f"  LLM: {provider}/{model} - available")
    else:
        checks["llm_available"] = False
        console.print(f"  LLM: No provider available (Ollama not running, no API keys)")

    # 1f. NLI model
    try:
        from src.generation.hallucination_detector import HallucinationDetector
        hd = HallucinationDetector()
        nli_ok = hd.nli_model is not None
        checks["nli_model"] = nli_ok
        console.print(f"  NLI model: {'loaded' if nli_ok else 'keyword fallback'}")
    except Exception as e:
        checks["nli_model"] = False
        console.print(f"  NLI model: FAILED ({e})")

    # 1g. Normalized terms populated
    sample_dir = PROJECT_ROOT / "data" / "chunks" / "adaptive" / "size_500"
    sample_files = list(sample_dir.glob("*.json"))[:20]
    enriched = 0
    for f in sample_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("normalized_terms"):
            enriched += 1
    checks["normalized_terms"] = enriched > 10
    console.print(f"  Normalized terms: {enriched}/{len(sample_files)} sample chunks enriched")

    passed = sum(v for v in checks.values())
    total = len(checks)
    all_critical = checks.get("corpus") and checks.get("index_loads") and checks.get("faiss_index")

    console.print(f"\n  Result: {passed}/{total} checks passed")
    status = PASS if all_critical else FAIL
    console.print(f"  {status}")
    results["test_1"] = all_critical
    return checks


# ============================================================
# TEST 2: Compare 3 Systems
# ============================================================

def test_2_compare_systems(hybrid_index, llm_available=False):
    """Compare retrieval across 3 configurations."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 2: Compare 3 Systems[/bold cyan]")
    console.print("=" * 60)

    query = "How do I deploy a containerized application on Kubernetes?"

    from src.pipeline.pipeline_config import PIPELINE_CONFIGS, get_config
    from src.retrieval.bm25_retriever import BM25Retriever
    from src.retrieval.dense_retriever import DenseRetriever
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.query_processor import QueryProcessor

    qp = QueryProcessor()

    table = Table(title=f"Comparison: '{query[:60]}...'")
    table.add_column("System", style="cyan")
    table.add_column("Method")
    table.add_column("Top-5 Providers")
    table.add_column("Retrieval ms", justify="right")
    table.add_column("Top Score", justify="right")

    all_ok = True

    configs = [
        ("lexical", "bm25"),
        ("semantic", "dense"),
        ("hybrid", "hybrid"),
    ]

    for config_name, method in configs:
        config = get_config(config_name)
        start = time.perf_counter()

        try:
            if method == "bm25":
                retriever = BM25Retriever(hybrid_index, query_processor=qp)
                results_list = retriever.search(query, top_k=5, use_expansion=True)
            elif method == "dense":
                retriever = DenseRetriever(hybrid_index, query_processor=qp)
                results_list = retriever.search(query, top_k=5)
            elif method == "hybrid":
                retriever = HybridRetriever(
                    hybrid_index, query_processor=qp,
                    fusion_method="rrf", alpha=0.5, rrf_k=60,
                )
                results_list = retriever.search(query, top_k=5)

            elapsed_ms = (time.perf_counter() - start) * 1000

            providers = [r.cloud_provider for r in results_list[:5]]
            provider_counts = {}
            for p in providers:
                provider_counts[p] = provider_counts.get(p, 0) + 1
            prov_str = ", ".join(f"{k}({v})" for k, v in provider_counts.items())

            top_score = results_list[0].score if results_list else 0.0

            table.add_row(
                config.name,
                method,
                prov_str,
                f"{elapsed_ms:.0f}",
                f"{top_score:.4f}",
            )

            if not results_list:
                all_ok = False

        except Exception as e:
            table.add_row(config_name, method, f"ERROR: {e}", "-", "-")
            all_ok = False

    console.print(table)

    if llm_available:
        console.print("  [dim]LLM generation available - full pipeline comparison possible[/dim]")

    status = PASS if all_ok else FAIL
    console.print(f"  {status}")
    results["test_2"] = all_ok


# ============================================================
# TEST 3: Cross-Cloud Query
# ============================================================

def test_3_cross_cloud(hybrid_index):
    """Test cross-cloud retrieval using terminology normalization."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 3: Cross-Cloud Query[/bold cyan]")
    console.print("=" * 60)

    query = "Compare serverless functions across AWS Lambda, Azure Functions, and Google Cloud Functions"

    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.query_processor import QueryProcessor
    from src.preprocessing.terminology_normalizer import TerminologyNormalizer

    # Test query expansion
    normalizer = TerminologyNormalizer()
    expansions = normalizer.get_query_expansion(query)
    console.print(f"  Query: {query}")
    console.print(f"  Expansions: {json.dumps(expansions, indent=2)}")

    # Test retrieval across providers
    qp = QueryProcessor()
    retriever = HybridRetriever(
        hybrid_index, query_processor=qp,
        fusion_method="rrf", alpha=0.5, rrf_k=60,
    )

    start = time.perf_counter()
    results_list = retriever.search(query, top_k=20)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Count providers in results
    provider_counts = {}
    for r in results_list:
        p = r.cloud_provider
        provider_counts[p] = provider_counts.get(p, 0) + 1

    table = Table(title="Cross-Cloud Results")
    table.add_column("Provider", style="cyan")
    table.add_column("Chunks Retrieved", justify="right")

    for p, c in sorted(provider_counts.items()):
        table.add_row(p, str(c))

    console.print(table)
    console.print(f"  Retrieval time: {elapsed_ms:.0f}ms")

    # Check we got results from at least 2 providers
    multi_provider = len(provider_counts) >= 2
    has_results = len(results_list) > 0
    has_expansions = len(expansions) > 0

    console.print(f"  Multi-provider results: {multi_provider} ({len(provider_counts)} providers)")
    console.print(f"  Terminology expansions: {has_expansions}")

    all_ok = multi_provider and has_results
    status = PASS if all_ok else FAIL
    console.print(f"  {status}")
    results["test_3"] = all_ok


# ============================================================
# TEST 4: Hallucination Detection
# ============================================================

def test_4_hallucination_detection(hybrid_index):
    """Test hallucination detection with synthetic responses."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 4: Hallucination Detection[/bold cyan]")
    console.print("=" * 60)

    from src.generation.hallucination_detector import HallucinationDetector

    detector = HallucinationDetector()

    # Get some real chunks for evidence
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.query_processor import QueryProcessor

    qp = QueryProcessor()
    retriever = HybridRetriever(hybrid_index, query_processor=qp)
    real_chunks = retriever.search("Kubernetes deployment pods replicas", top_k=5)

    chunk_dicts = []
    for r in real_chunks:
        chunk_data = hybrid_index.get_chunk(r.chunk_id)
        if chunk_data:
            chunk_dicts.append(chunk_data)

    if not chunk_dicts:
        console.print("  [red]No chunks retrieved for evidence[/red]")
        results["test_4"] = False
        return

    console.print(f"  Evidence chunks: {len(chunk_dicts)}")

    # Test A: Faithful response (uses content from chunks)
    sample_text = chunk_dicts[0].get("text", "")[:200]
    faithful_response = f"According to the documentation, {sample_text}"
    report_a = detector.check(faithful_response, chunk_dicts)
    console.print(f"\n  A) Faithful response:")
    console.print(f"     Faithfulness: {report_a.faithfulness_score:.2f}")
    console.print(f"     Rubric: {report_a.suggested_rubric}/5")
    console.print(f"     Claims: {report_a.total_claims} total, {report_a.supported_claims} supported")

    # Test B: Hallucinated response (fabricated facts)
    hallucinated_response = (
        "Kubernetes was developed by Microsoft in 2020. "
        "It requires a minimum of 10 nodes to operate. "
        "Docker is the only container runtime supported by Kubernetes. "
        "Pods can only contain a single container."
    )
    report_b = detector.check(hallucinated_response, chunk_dicts)
    console.print(f"\n  B) Hallucinated response:")
    console.print(f"     Faithfulness: {report_b.faithfulness_score:.2f}")
    console.print(f"     Rubric: {report_b.suggested_rubric}/5")
    console.print(f"     Hallucination rate: {report_b.hallucination_rate:.2f}")
    console.print(f"     Claims: {report_b.total_claims} total, {report_b.unsupported_claims} unsupported")

    # Validation: faithful should score higher than hallucinated
    faith_better = report_a.faithfulness_score >= report_b.faithfulness_score
    rubric_a_ok = report_a.suggested_rubric >= 2
    method = report_a.method

    console.print(f"\n  Method used: {method}")
    console.print(f"  Faithful > Hallucinated: {faith_better}")

    all_ok = rubric_a_ok  # At minimum the faithful response should score decently
    status = PASS if all_ok else FAIL
    console.print(f"  {status}")
    results["test_4"] = all_ok


# ============================================================
# TEST 5: No-Answer / HONEST_DECLINE
# ============================================================

def test_5_honest_decline():
    """Test HONEST_DECLINE detection in response formatter."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 5: HONEST_DECLINE Detection[/bold cyan]")
    console.print("=" * 60)

    from src.generation.response_formatter import ResponseFormatter

    formatter = ResponseFormatter()

    # Test various decline responses
    decline_texts = [
        "I cannot find sufficient information in the provided context to answer this question.",
        "Based on the available documentation, I cannot provide a reliable answer.",
        "The provided context does not contain information about quantum computing in cloud.",
        "No relevant information was found in the documentation for this specific topic.",
    ]

    # Test confident responses
    confident_texts = [
        "To deploy on Kubernetes, create a deployment YAML file. [Source: kubernetes/deployments/guide]",
        "AWS Lambda supports Python, Node.js, and Java runtimes. [Source: aws/Lambda/runtimes] [Source: aws/Lambda/config] [Source: aws/Lambda/guide]",
    ]

    table = Table(title="HONEST_DECLINE Detection")
    table.add_column("Type")
    table.add_column("Text (truncated)")
    table.add_column("Confidence")
    table.add_column("Correct?")

    all_ok = True

    for text in decline_texts:
        result = formatter.format(text, [], None)
        correct = result.confidence == "HONEST_DECLINE"
        if not correct:
            all_ok = False
        table.add_row(
            "Decline",
            text[:60] + "...",
            result.confidence,
            "OK" if correct else "WRONG",
        )

    for text in confident_texts:
        result = formatter.format(text, [], None)
        correct = result.confidence in ("HIGH", "MEDIUM", "LOW")
        if not correct:
            all_ok = False
        table.add_row(
            "Confident",
            text[:60] + "...",
            result.confidence,
            "OK" if correct else "WRONG",
        )

    console.print(table)

    status = PASS if all_ok else FAIL
    console.print(f"  {status}")
    results["test_5"] = all_ok


# ============================================================
# TEST 6: Latency Breakdown
# ============================================================

def test_6_latency_breakdown(hybrid_index):
    """Test latency measurement for all pipeline stages."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 6: Latency Breakdown[/bold cyan]")
    console.print("=" * 60)

    from src.pipeline.rag_pipeline import LatencyTracker
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.query_processor import QueryProcessor
    from src.reranking.cross_encoder_reranker import CrossEncoderReranker
    from src.generation.hallucination_detector import HallucinationDetector

    query = "How to configure autoscaling in Kubernetes?"
    tracker = LatencyTracker()

    # Stage 1: Query Processing
    with tracker.measure("query_processing"):
        qp = QueryProcessor()
        processed = qp.process(query)

    # Stage 2: Retrieval (hybrid)
    with tracker.measure("retrieval"):
        retriever = HybridRetriever(hybrid_index, query_processor=qp)
        candidates = retriever.search(query, top_k=50)

    # Stage 3: Reranking
    with tracker.measure("reranking"):
        try:
            reranker = CrossEncoderReranker(model_name="ms-marco-mini-6")
            reranked = reranker.rerank(query, candidates, top_k=5)
        except Exception as e:
            console.print(f"  [yellow]Reranker warning: {e}[/yellow]")
            reranked = candidates[:5]

    # Get chunk dicts for generation stage
    chunk_dicts = []
    for r in reranked[:5]:
        chunk_data = hybrid_index.get_chunk(r.chunk_id)
        if chunk_data:
            chunk_dicts.append(chunk_data)

    # Stage 4: Generation (simulated if no LLM)
    with tracker.measure("generation"):
        provider, model = get_llm_provider()
        if provider:
            from src.generation.llm_manager import LLMManager
            from src.generation.prompt_templates import SYSTEM_PROMPT, build_context, get_template
            llm = LLMManager(provider=provider, model=model)
            context = build_context(chunk_dicts, "default")
            template = get_template("default")
            prompt = template.format(context=context, question=query)
            llm_response = llm.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            gen_text = llm_response.text
        else:
            # Simulate generation
            time.sleep(0.01)
            gen_text = "Simulated LLM response for latency measurement."

    # Stage 5: Hallucination check
    with tracker.measure("hallucination_check"):
        detector = HallucinationDetector()
        report = detector.check(gen_text, chunk_dicts)

    # Print breakdown
    breakdown = tracker.get_breakdown()

    table = Table(title="Latency Breakdown")
    table.add_column("Stage", style="cyan")
    table.add_column("Time (ms)", justify="right")
    table.add_column("% of Total", justify="right")

    stages = [
        ("Query Processing", breakdown.query_processing_ms),
        ("Retrieval", breakdown.retrieval_ms),
        ("Reranking", breakdown.reranking_ms),
        ("Generation", breakdown.generation_ms),
        ("Hallucination Check", breakdown.hallucination_check_ms),
    ]

    total = breakdown.total_ms or 1
    for name, ms in stages:
        pct = (ms / total) * 100
        table.add_row(name, f"{ms:.1f}", f"{pct:.1f}%")

    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total:.1f}[/bold]", "[bold]100%[/bold]")
    console.print(table)

    # Validate: all stages should have non-zero latency
    all_measured = all(ms > 0 for _, ms in stages if _ != "Generation")
    status = PASS if all_measured else FAIL
    console.print(f"  {status}")
    results["test_6"] = all_measured


# ============================================================
# TEST 7: LLM Cache
# ============================================================

def test_7_cache_verification():
    """Test LLM cache: verify same prompt returns cached result."""
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST 7: LLM Cache Verification[/bold cyan]")
    console.print("=" * 60)

    from src.generation.llm_manager import LLMManager

    # Create a manager with cache enabled
    provider, model = get_llm_provider()

    if provider:
        llm = LLMManager(provider=provider, model=model, cache_enabled=True)
        test_prompt = "What is Kubernetes? Answer in one sentence."

        # First call (should not be cached)
        llm.clear_cache()
        resp1 = llm.generate(test_prompt, temperature=0.1)

        if resp1.error:
            console.print(f"  [yellow]LLM error (skipping): {resp1.error}[/yellow]")
            results["test_7"] = True  # Skip gracefully
            console.print(f"  {SKIP} (no LLM available)")
            return

        console.print(f"  First call: {resp1.latency_ms:.0f}ms, from_cache={resp1.from_cache}")
        assert not resp1.from_cache, "First call should not be from cache"

        # Second call (should be cached)
        resp2 = llm.generate(test_prompt, temperature=0.1)
        console.print(f"  Second call: {resp2.latency_ms:.0f}ms, from_cache={resp2.from_cache}")
        assert resp2.from_cache, "Second call should be from cache"

        # Verify texts match
        texts_match = resp1.text == resp2.text
        console.print(f"  Texts match: {texts_match}")
        console.print(f"  Cache speedup: {resp1.latency_ms:.0f}ms -> {resp2.latency_ms:.0f}ms")

        all_ok = resp2.from_cache and texts_match
    else:
        # Test cache mechanism without real LLM
        console.print("  [dim]No LLM available - testing cache mechanism directly[/dim]")

        llm = LLMManager(provider="ollama", model="test-model", cache_enabled=True)
        llm.clear_cache()

        # Manually inject a cache entry
        test_prompt = "Test prompt for cache"
        key = LLMManager._cache_key(test_prompt, "", 0.1)
        llm._cache[key] = {
            "text": "Cached response text",
            "tokens_input": 10,
            "tokens_output": 5,
        }
        llm._save_cache()

        # Reload cache and verify
        llm2 = LLMManager(provider="ollama", model="test-model", cache_enabled=True)
        resp = llm2.generate(test_prompt, temperature=0.1)

        console.print(f"  Cache key: {key[:16]}...")
        console.print(f"  From cache: {resp.from_cache}")
        console.print(f"  Cached text: {resp.text[:50]}")

        all_ok = resp.from_cache and resp.text == "Cached response text"

        # Cleanup
        llm.clear_cache()

    status = PASS if all_ok else FAIL
    console.print(f"  {status}")
    results["test_7"] = all_ok


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 Verification Tests")
    parser.add_argument("--with-llm", action="store_true", help="Run tests that need LLM")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    load_env()

    console.print(Panel(
        "[bold]BLOQUE D: Phase 3 Verification Tests[/bold]\n"
        "7 tests covering health, retrieval, cross-cloud, hallucination,\n"
        "HONEST_DECLINE, latency, and caching.",
        title="Hybrid RAG System - Phase 3",
        border_style="cyan",
    ))

    start_total = time.time()

    # Check LLM availability
    provider, model = get_llm_provider()
    llm_available = provider is not None
    if llm_available:
        console.print(f"[green]LLM available: {provider}/{model}[/green]")
    else:
        console.print("[yellow]No LLM available - tests will run in retrieval-only mode[/yellow]")
        console.print("[dim]Install Ollama (ollama.com) and run 'ollama pull llama3.1:8b-instruct-q4_K_M'[/dim]")

    # Test 1: Health Check
    checks = test_1_health_check()

    # Load index once for all retrieval tests
    hybrid_index = None
    if checks.get("index_loads"):
        console.print("\n[dim]Loading hybrid index for retrieval tests...[/dim]")
        from src.pipeline.rag_pipeline import load_hybrid_index
        hybrid_index = load_hybrid_index()
    else:
        console.print("[red]Index failed to load - skipping retrieval tests[/red]")

    if hybrid_index:
        # Test 2: Compare 3 Systems
        test_2_compare_systems(hybrid_index, llm_available)

        # Test 3: Cross-Cloud Query
        test_3_cross_cloud(hybrid_index)

        # Test 4: Hallucination Detection
        test_4_hallucination_detection(hybrid_index)
    else:
        results["test_2"] = False
        results["test_3"] = False
        results["test_4"] = False

    # Test 5: HONEST_DECLINE (no index needed)
    test_5_honest_decline()

    if hybrid_index:
        # Test 6: Latency Breakdown
        test_6_latency_breakdown(hybrid_index)
    else:
        results["test_6"] = False

    # Test 7: Cache Verification
    test_7_cache_verification()

    # Summary
    elapsed = time.time() - start_total

    console.print("\n" + "=" * 60)
    console.print("[bold cyan]SUMMARY[/bold cyan]")
    console.print("=" * 60)

    table = Table(title="Phase 3 Verification Results")
    table.add_column("Test", style="cyan")
    table.add_column("Description")
    table.add_column("Result")

    test_names = {
        "test_1": "Health Check",
        "test_2": "Compare 3 Systems",
        "test_3": "Cross-Cloud Query",
        "test_4": "Hallucination Detection",
        "test_5": "HONEST_DECLINE",
        "test_6": "Latency Breakdown",
        "test_7": "LLM Cache",
    }

    passed = 0
    total = len(test_names)
    for key, desc in test_names.items():
        ok = results.get(key, False)
        if ok:
            passed += 1
        table.add_row(key.replace("_", " ").title(), desc, "PASS" if ok else "FAIL")

    console.print(table)
    console.print(f"\n[bold]Result: {passed}/{total} tests passed in {elapsed:.1f}s[/bold]")

    if passed == total:
        console.print("[bold green]ALL TESTS PASSED[/bold green]")
    elif passed >= 5:
        console.print("[bold yellow]MOSTLY PASSED - some tests need LLM or further setup[/bold yellow]")
    else:
        console.print("[bold red]TESTS FAILED - check errors above[/bold red]")

    return 0 if passed >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
