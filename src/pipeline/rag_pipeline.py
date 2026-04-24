"""
RAG Pipeline - Configurable end-to-end pipeline connecting all stages.

Supports 3 configurations:
  1. BASELINE_LEXICAL (Control 1): BM25 -> No reranking -> LLM
  2. BASELINE_SEMANTIC (Control 2): Dense -> No reranking -> LLM
  3. PROPOSED_HYBRID (Experimental): Hybrid RRF -> Cross-encoder reranking -> LLM

Pipeline stages:
  query -> processing -> retrieval -> reranking -> generation -> hallucination check
"""

import logging
import time
from contextlib import contextmanager
from typing import List, Optional

from pydantic import BaseModel

from src.generation.hallucination_detector import HallucinationDetector, HallucinationReport
from src.generation.llm_manager import LLMManager, LLMResponse
from src.generation.prompt_templates import SYSTEM_PROMPT, build_context, get_template
from src.generation.response_formatter import ResponseFormatter
from src.pipeline.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


class LatencyBreakdown(BaseModel):
    """Latency per pipeline stage in milliseconds."""
    query_processing_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    hallucination_check_ms: float = 0.0
    total_ms: float = 0.0


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline."""
    answer: str
    sources: List[dict] = []
    confidence: str = "UNKNOWN"
    retrieved_chunks: List[dict] = []
    hallucination_report: Optional[HallucinationReport] = None
    latency: LatencyBreakdown = LatencyBreakdown()
    llm_response: Optional[LLMResponse] = None
    config_name: str = ""
    error: Optional[str] = None


class LatencyTracker:
    """Measures latency per pipeline stage."""

    def __init__(self):
        self._stages = {}

    @contextmanager
    def measure(self, stage_name: str):
        """Context manager that measures time for a stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._stages[stage_name] = elapsed_ms

    def get_breakdown(self) -> LatencyBreakdown:
        """Return latencies of all stages."""
        breakdown = LatencyBreakdown(
            query_processing_ms=self._stages.get("query_processing", 0.0),
            retrieval_ms=self._stages.get("retrieval", 0.0),
            reranking_ms=self._stages.get("reranking", 0.0),
            generation_ms=self._stages.get("generation", 0.0),
            hallucination_check_ms=self._stages.get("hallucination_check", 0.0),
        )
        breakdown.total_ms = sum([
            breakdown.query_processing_ms,
            breakdown.retrieval_ms,
            breakdown.reranking_ms,
            breakdown.generation_ms,
            breakdown.hallucination_check_ms,
        ])
        return breakdown


class RAGPipeline:
    """Configurable end-to-end RAG pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        hybrid_index=None,
        llm_manager: Optional[LLMManager] = None,
    ):
        self.config = config
        self.hybrid_index = hybrid_index
        self.query_processor = self._build_query_processor()
        self.retriever = self._build_retriever()
        self.reranker = self._build_reranker()
        self.llm = llm_manager or LLMManager(
            provider=config.llm_provider,
            model=config.llm_model,
        )
        self.hallucination_detector = HallucinationDetector()
        self.response_formatter = ResponseFormatter()

    def _build_query_processor(self):
        if self.config.query_expansion or self.config.terminology_normalization:
            from src.retrieval.query_processor import QueryProcessor
            return QueryProcessor()
        return None

    def _build_retriever(self):
        """Create the appropriate retriever."""
        from src.retrieval.query_processor import QueryProcessor

        query_processor = self.query_processor or QueryProcessor()

        if self.config.retrieval_method == "bm25":
            from src.retrieval.bm25_retriever import BM25Retriever
            return BM25Retriever(self.hybrid_index, query_processor=query_processor)

        if self.config.retrieval_method == "dense":
            from src.retrieval.dense_retriever import DenseRetriever
            return DenseRetriever(self.hybrid_index, query_processor=query_processor)

        if self.config.retrieval_method == "hybrid":
            from src.retrieval.hybrid_retriever import HybridRetriever
            return HybridRetriever(
                self.hybrid_index,
                query_processor=query_processor,
                reranker=None,
                fusion_method=self.config.fusion_method or "rrf",
                alpha=self.config.alpha,
                rrf_k=self.config.rrf_k,
            )

        raise ValueError(f"Unknown retrieval method: {self.config.retrieval_method}")

    def _build_reranker(self):
        """Create the reranker if configured."""
        if not self.config.reranker:
            return None

        from src.reranking.cross_encoder_reranker import CrossEncoderReranker

        model_map = {
            "cross-encoder/ms-marco-MiniLM-L-6-v2": "ms-marco-mini-6",
            "cross-encoder/ms-marco-MiniLM-L-12-v2": "ms-marco-mini-12",
            "BAAI/bge-reranker-large": "bge-reranker-large",
        }
        short_name = model_map.get(self.config.reranker, self.config.reranker)
        cross_encoder = CrossEncoderReranker(model_name=short_name)

        if not self.config.multidimensional_scoring:
            return cross_encoder

        from src.reranking.multidimensional_scorer import MultidimensionalScorer

        embedding_manager = getattr(self.hybrid_index, "embedding_manager", None)
        return MultidimensionalScorer(
            cross_encoder_reranker=cross_encoder,
            embedding_manager=embedding_manager,
        )

    def query(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline for a question."""
        latency = LatencyTracker()

        try:
            if self.query_processor is None and (
                self.config.query_expansion or self.config.terminology_normalization
            ):
                self.query_processor = self._build_query_processor()

            with latency.measure("query_processing"):
                if self.query_processor:
                    processed = self.query_processor.process(
                        question,
                        enable_query_expansion=self.config.query_expansion,
                        enable_terminology_normalization=self.config.terminology_normalization,
                    )
                    query_type = processed.query_type
                else:
                    processed = None
                    query_type = "default"

            with latency.measure("retrieval"):
                if self.config.retrieval_method == "bm25":
                    candidates = self.retriever.search(
                        question,
                        top_k=self.config.retrieval_top_k,
                        use_expansion=self.config.query_expansion,
                        enable_query_expansion=self.config.query_expansion,
                        enable_terminology_normalization=self.config.terminology_normalization,
                    )
                elif self.config.retrieval_method == "dense":
                    candidates = self.retriever.search(
                        question,
                        top_k=self.config.retrieval_top_k,
                        enable_query_expansion=self.config.query_expansion,
                        enable_terminology_normalization=self.config.terminology_normalization,
                    )
                elif self.config.retrieval_method == "hybrid":
                    candidates = self.retriever.search(
                        question,
                        top_k=self.config.retrieval_top_k,
                        top_k_candidates=self.config.retrieval_top_k,
                        use_reranker=False,
                        enable_query_expansion=self.config.query_expansion,
                        enable_terminology_normalization=self.config.terminology_normalization,
                    )
                else:
                    candidates = []

            if not candidates:
                return RAGResponse(
                    answer="No relevant documentation found for this query.",
                    config_name=self.config.name,
                    latency=latency.get_breakdown(),
                    error="No chunks retrieved",
                )

            with latency.measure("reranking"):
                reranker_cutoff = max(self.config.reranker_top_k, self.config.final_top_k)
                if self.reranker is not None:
                    reranked = self.reranker.rerank(question, candidates, top_k=reranker_cutoff)
                else:
                    reranked = candidates
                selected_results = reranked[: self.config.final_top_k]

            chunk_dicts = []
            for result in selected_results:
                chunk_data = self.hybrid_index.get_chunk(result.chunk_id) if self.hybrid_index else {}
                if chunk_data:
                    chunk_dicts.append(chunk_data)
                else:
                    chunk_dicts.append({
                        "chunk_id": result.chunk_id,
                        "text": result.chunk_text,
                        "cloud_provider": result.cloud_provider,
                        "service_name": result.service_name,
                        "heading_path": result.heading_path,
                    })

            context = build_context(chunk_dicts, query_type)
            template = get_template(query_type)

            if query_type == "cross_cloud":
                full_prompt = template.format(context_by_provider=context, question=question)
            else:
                full_prompt = template.format(context=context, question=question)

            with latency.measure("generation"):
                llm_response = self.llm.generate(
                    prompt=full_prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=self.config.temperature,
                    config_name=self.config.name,
                )

            if llm_response.error:
                return RAGResponse(
                    answer=f"Error generating response: {llm_response.error}",
                    retrieved_chunks=chunk_dicts,
                    latency=latency.get_breakdown(),
                    llm_response=llm_response,
                    config_name=self.config.name,
                    error=llm_response.error,
                )

            with latency.measure("hallucination_check"):
                hall_report = self.hallucination_detector.check(
                    response=llm_response.text,
                    retrieved_chunks=chunk_dicts,
                )

            formatted = self.response_formatter.format(llm_response.text, chunk_dicts, hall_report)

            return RAGResponse(
                answer=formatted.text,
                sources=formatted.sources,
                confidence=formatted.confidence,
                retrieved_chunks=chunk_dicts,
                hallucination_report=hall_report,
                latency=latency.get_breakdown(),
                llm_response=llm_response,
                config_name=self.config.name,
            )

        except Exception as exc:
            logger.exception("Pipeline error: %s", exc)
            return RAGResponse(
                answer=f"Pipeline error: {str(exc)}",
                latency=latency.get_breakdown(),
                config_name=self.config.name,
                error=str(exc),
            )


def load_hybrid_index(
    embedding_model: str = "bge-large",
    chunking_strategy: str = "adaptive",
    chunk_size: int = 500,
):
    """Load the hybrid index from disk."""
    from src.embedding.embedding_manager import EmbeddingManager
    from src.embedding.index.hybrid_index import HybridIndex

    embedding_manager = EmbeddingManager(model_name=embedding_model)
    index = HybridIndex(embedding_manager=embedding_manager)
    index.load(chunk_strategy=chunking_strategy, chunk_size=chunk_size)
    return index


def create_pipeline(config_name: str = "hybrid", hybrid_index=None) -> RAGPipeline:
    """Create a RAG pipeline from a named config."""
    from src.pipeline.pipeline_config import get_config

    config = get_config(config_name)
    if hybrid_index is None:
        hybrid_index = load_hybrid_index()
    return RAGPipeline(config=config, hybrid_index=hybrid_index)
