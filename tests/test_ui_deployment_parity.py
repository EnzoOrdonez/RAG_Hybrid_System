"""Behavioral coverage for the survey deployment path consumed by Streamlit."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest

from src.pipeline.pipeline_config import PROPOSED_HYBRID, SURVEY_DEPLOY, get_config
from src.retrieval.bm25_retriever import RetrievalResult


if importlib.util.find_spec("streamlit") is None:
    def _cache_resource(*args, **kwargs):
        def decorate(function):
            function.clear = lambda: None
            return function
        return decorate

    sys.modules["streamlit"] = SimpleNamespace(cache_resource=_cache_resource)


class _LLMProbe:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _patch_pipeline_construction(monkeypatch):
    import src.generation.llm_manager as llm_manager
    import src.pipeline.rag_pipeline as rag_pipeline

    class PipelineProbe:
        def __init__(self, config, hybrid_index, llm_manager=None):
            self.config = config
            self.hybrid_index = hybrid_index
            self.llm_manager = llm_manager

    monkeypatch.setattr(llm_manager, "LLMManager", _LLMProbe)
    monkeypatch.setattr(rag_pipeline, "RAGPipeline", PipelineProbe)


def test_ui_hybrid_uses_survey_deploy_without_changing_experiment_registry(monkeypatch):
    """Streamlit gets the survey config; experiment callers keep the signed behavior."""
    from src.ui.components import index_loader

    _patch_pipeline_construction(monkeypatch)
    index_loader.load_pipeline.clear()

    pipeline = index_loader.load_pipeline("hybrid", _hybrid_index=object())

    assert pipeline.config == SURVEY_DEPLOY
    assert pipeline.config.prompt_routing is True
    assert pipeline.config.balance_cross_cloud_providers is True
    assert pipeline.config.temperature == 0.0
    assert pipeline.config.retrieval_top_k == 50
    assert pipeline.config.final_top_k == 5
    assert pipeline.llm_manager.model == SURVEY_DEPLOY.llm_model
    assert pipeline.llm_manager.seed == 42
    assert pipeline.llm_manager.cache_enabled is False
    assert get_config("hybrid") == PROPOSED_HYBRID
    assert get_config("hybrid").prompt_routing is False
    assert get_config("hybrid").balance_cross_cloud_providers is False

    index_loader.load_pipeline.clear()


@pytest.mark.parametrize("config_name", ["lexical", "semantic"])
def test_ui_keeps_control_configs_unchanged(monkeypatch, config_name):
    """Control retrieval stays intact while the UI generator matches exp19b."""
    from src.ui.components import index_loader

    _patch_pipeline_construction(monkeypatch)
    index_loader.load_pipeline.clear()

    pipeline = index_loader.load_pipeline(config_name, _hybrid_index=object())

    expected = get_config(config_name).model_copy(
        update={"llm_model": SURVEY_DEPLOY.llm_model}
    )
    assert pipeline.config == expected
    assert pipeline.llm_manager.model == SURVEY_DEPLOY.llm_model
    assert pipeline.llm_manager.seed == 42
    assert pipeline.llm_manager.cache_enabled is False

    index_loader.load_pipeline.clear()


def test_ui_explicit_model_override_is_visible_in_config(monkeypatch):
    from src.ui.components import index_loader

    _patch_pipeline_construction(monkeypatch)
    index_loader.load_pipeline.clear()

    pipeline = index_loader.load_pipeline(
        "hybrid", _hybrid_index=object(), llm_model="alternate:model"
    )

    assert pipeline.config.llm_model == "alternate:model"
    assert pipeline.llm_manager.model == "alternate:model"
    index_loader.load_pipeline.clear()


def test_chat_defaults_follow_experimental_generation_recipe():
    from pathlib import Path

    root = Path(__file__).parents[1]
    source = (root / "src/ui/pages/chat_page.py").read_text(
        encoding="utf-8"
    )
    loader_source = (root / "src/ui/components/index_loader.py").read_text(
        encoding="utf-8"
    )
    assert "default_model = SURVEY_DEPLOY.llm_model" in source
    assert "UI_MAX_TOKENS = 1024" in loader_source
    assert "value=UI_MAX_TOKENS" in source


def test_survey_stream_routes_cross_cloud_prompt(monkeypatch):
    """The participant-facing stream must use the measured cross-cloud template."""
    import src.pipeline.rag_pipeline as rag_pipeline

    results = [
        RetrievalResult(chunk_id="aws-1", score=1.0, chunk_text="AWS evidence"),
        RetrievalResult(chunk_id="azure-1", score=0.9, chunk_text="Azure evidence"),
    ]

    class RetrieverStub:
        def search(self, *args, **kwargs):
            return results

    class RerankerStub:
        def rerank(self, question, candidates, top_k):
            return list(candidates[:top_k])

    class LLMStreamProbe:
        def __init__(self):
            self.prompt = None

        def generate_stream(self, prompt, **kwargs):
            self.prompt = prompt
            yield "answer"

    class IndexStub:
        chunk_map = {
            "aws-1": {"chunk_id": "aws-1", "text": "AWS evidence",
                      "cloud_provider": "aws", "service_name": "s3"},
            "azure-1": {"chunk_id": "azure-1", "text": "Azure evidence",
                        "cloud_provider": "azure", "service_name": "blob"},
        }

        def get_chunk(self, chunk_id):
            return self.chunk_map[chunk_id]

    retriever = RetrieverStub()
    reranker = RerankerStub()
    llm = LLMStreamProbe()
    monkeypatch.setattr(rag_pipeline.RAGPipeline, "_build_retriever", lambda self: retriever)
    monkeypatch.setattr(rag_pipeline.RAGPipeline, "_build_reranker", lambda self: reranker)
    monkeypatch.setattr(rag_pipeline, "HallucinationDetector", lambda: object())

    pipeline = rag_pipeline.RAGPipeline(
        config=SURVEY_DEPLOY,
        hybrid_index=IndexStub(),
        llm_manager=llm,
    )
    pipeline._routing_qp = SimpleNamespace(process=lambda question: SimpleNamespace(
        query_type="cross_cloud", detected_providers=["aws", "azure"]))

    list(pipeline.query_stream("Compare AWS S3 and Azure Blob Storage"))

    assert "Context from multiple cloud providers:" in llm.prompt
    assert "Comparative Answer:" in llm.prompt


def test_survey_stream_balances_cross_cloud_evidence_from_the_full_pool(monkeypatch):
    """Streaming must expose evidence for each requested provider when the pool has it."""
    import src.pipeline.rag_pipeline as rag_pipeline

    results = [
        RetrievalResult(chunk_id="aws-1", score=1.0),
        RetrievalResult(chunk_id="aws-2", score=0.9),
        RetrievalResult(chunk_id="azure-1", score=0.8),
        RetrievalResult(chunk_id="azure-2", score=0.7),
    ]

    class RetrieverStub:
        def search(self, *args, **kwargs):
            return results

    class RerankerStub:
        def rerank(self, question, candidates, top_k):
            return list(candidates[:top_k])

    class LLMStub:
        def generate_stream(self, prompt, **kwargs):
            yield "answer"

    class IndexStub:
        chunk_map = {
            "aws-1": {"chunk_id": "aws-1", "text": "AWS 1", "cloud_provider": "aws"},
            "aws-2": {"chunk_id": "aws-2", "text": "AWS 2", "cloud_provider": "aws"},
            "azure-1": {"chunk_id": "azure-1", "text": "Azure 1", "cloud_provider": "azure"},
            "azure-2": {"chunk_id": "azure-2", "text": "Azure 2", "cloud_provider": "azure"},
        }

        def get_chunk(self, chunk_id):
            return self.chunk_map[chunk_id]

    retriever = RetrieverStub()
    reranker = RerankerStub()
    monkeypatch.setattr(rag_pipeline.RAGPipeline, "_build_retriever", lambda self: retriever)
    monkeypatch.setattr(rag_pipeline.RAGPipeline, "_build_reranker", lambda self: reranker)
    monkeypatch.setattr(rag_pipeline, "HallucinationDetector", lambda: object())

    config = SURVEY_DEPLOY.model_copy(update={"retrieval_top_k": 4, "final_top_k": 2})
    pipeline = rag_pipeline.RAGPipeline(
        config=config,
        hybrid_index=IndexStub(),
        llm_manager=LLMStub(),
    )
    pipeline._routing_qp = SimpleNamespace(process=lambda question: SimpleNamespace(
        query_type="cross_cloud", detected_providers=["aws", "azure"]))

    events = list(pipeline.query_stream("Compare AWS and Azure"))
    payload = next(value for kind, value in events if kind == "done")

    assert [chunk["chunk_id"] for chunk in payload["retrieved_chunks"]] == [
        "aws-1", "azure-1"]


def test_survey_stream_does_not_balance_non_cross_cloud_queries(monkeypatch):
    """Prompt routing applies broadly, but provider balancing stays cross-cloud only."""
    import src.pipeline.rag_pipeline as rag_pipeline

    results = [
        RetrievalResult(chunk_id="aws-1", score=1.0),
        RetrievalResult(chunk_id="aws-2", score=0.9),
        RetrievalResult(chunk_id="azure-1", score=0.8),
    ]

    class RetrieverStub:
        def search(self, *args, **kwargs):
            return results

    class RerankerStub:
        def rerank(self, question, candidates, top_k):
            return list(candidates[:top_k])

    class LLMProbe:
        def __init__(self):
            self.prompt = None

        def generate_stream(self, prompt, **kwargs):
            self.prompt = prompt
            yield "answer"

    class IndexStub:
        chunk_map = {
            "aws-1": {"chunk_id": "aws-1", "text": "AWS 1", "cloud_provider": "aws"},
            "aws-2": {"chunk_id": "aws-2", "text": "AWS 2", "cloud_provider": "aws"},
            "azure-1": {"chunk_id": "azure-1", "text": "Azure 1", "cloud_provider": "azure"},
        }

        def get_chunk(self, chunk_id):
            return self.chunk_map[chunk_id]

    retriever = RetrieverStub()
    reranker = RerankerStub()
    llm = LLMProbe()
    monkeypatch.setattr(rag_pipeline.RAGPipeline, "_build_retriever", lambda self: retriever)
    monkeypatch.setattr(rag_pipeline.RAGPipeline, "_build_reranker", lambda self: reranker)
    monkeypatch.setattr(rag_pipeline, "HallucinationDetector", lambda: object())

    config = SURVEY_DEPLOY.model_copy(update={"retrieval_top_k": 3, "final_top_k": 2})
    pipeline = rag_pipeline.RAGPipeline(
        config=config,
        hybrid_index=IndexStub(),
        llm_manager=llm,
    )
    pipeline._routing_qp = SimpleNamespace(process=lambda question: SimpleNamespace(
        query_type="procedural", detected_providers=["aws", "azure"]))

    events = list(pipeline.query_stream("How do I configure AWS S3?"))
    payload = next(value for kind, value in events if kind == "done")

    assert [chunk["chunk_id"] for chunk in payload["retrieved_chunks"]] == [
        "aws-1", "aws-2"]
    assert "Step-by-step Answer:" in llm.prompt
