"""Auditoría Fable: caracterización del camino de respuesta (P2), de punta a punta y sin modelos.

Fija lo que hoy producen el prompt, la fusión, el recuperador híbrido, el formateador y
`RAGPipeline.query`, para que un cambio de clase A pueda demostrar que no altera nada y un
cambio de clase B se note. El índice, el reranker y el LLM son dobles; el verificador corre
con `use_nli=False` (solapamiento de palabras), así que no se carga ningún modelo (D1).
"""

import hashlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embedding.index.hybrid_index import HybridIndex  # noqa: E402
from src.generation import prompt_templates as pt  # noqa: E402
from src.generation.hallucination_detector import HallucinationDetector  # noqa: E402
from src.generation.llm_manager import LLMResponse  # noqa: E402
from src.generation.response_formatter import ResponseFormatter  # noqa: E402
from src.pipeline.pipeline_config import (  # noqa: E402
    LLM_ONLY_NO_RAG, PROPOSED_HYBRID, SURVEY_DEPLOY)
from src.pipeline.rag_pipeline import RAGPipeline  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.query_processor import QueryProcessor  # noqa: E402


def _chunk(cid, provider, service, text, heading=""):
    return {"chunk_id": cid, "text": text, "cloud_provider": provider,
            "service_name": service, "heading_path": heading, "doc_type": "guide"}


CHUNKS = [
    _chunk("a1", "aws", "S3", "Amazon S3 stores objects inside buckets in a region.", "Buckets"),
    _chunk("a2", "aws", "S3", "S3 bucket names are globally unique."),
    _chunk("a3", "aws", "EC2", "EC2 provides resizable compute capacity."),
    _chunk("a4", "aws", "Lambda", "Lambda runs code without provisioning servers."),
    _chunk("a5", "aws", "IAM", "IAM controls access to AWS resources."),
    _chunk("a6", "aws", "VPC", "A VPC is an isolated virtual network."),
    _chunk("z1", "azure", "Blob Storage", "Blob Storage keeps blobs inside containers."),
    _chunk("z2", "azure", "AKS", "AKS is a managed Kubernetes service."),
    _chunk("g1", "gcp", "GKE", "GKE runs containerized applications."),
]
RANKING = [(c["chunk_id"], 1.0 - i / 100) for i, c in enumerate(CHUNKS)]


class FakeIndex:
    """Doble de HybridIndex: devuelve un ranking fijo y registra cómo lo llamaron."""

    def __init__(self, ranking=RANKING, error=None):
        self.chunk_map = {c["chunk_id"]: c for c in CHUNKS}
        self.ranking, self.error, self.calls = list(ranking), error, []

    def get_chunk(self, chunk_id):
        return self.chunk_map.get(chunk_id)

    def search_hybrid(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return list(self.ranking)


class FakeLLM:
    def __init__(self, text="S3 stores objects in buckets [Source: aws/S3/Buckets].", error=None):
        self.text, self.error, self.calls = text, error, []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(text="" if self.error else self.text, model="fake", provider="fake",
                           tokens_input=1, tokens_output=1, latency_ms=0.0, from_cache=False,
                           error=self.error)


class FakeReranker:
    """Invierte el orden de los candidatos y devuelve top_k, como haría un cross-encoder."""

    def __init__(self):
        self.top_ks = []

    def rerank(self, query, candidates, top_k=5):
        self.top_ks.append(top_k)
        return list(reversed(candidates))[:top_k]


def _pipeline(config, index=None, llm=None, reranker=None):
    index = FakeIndex() if index is None and config.retrieval_method != "none" else index
    llm = llm or FakeLLM()
    pipeline = RAGPipeline(config=config, hybrid_index=index, llm_manager=llm)
    pipeline.hallucination_detector = HallucinationDetector(use_nli=False)   # sin NLI (D1)
    pipeline.reranker = reranker
    return pipeline, index, llm


LEGACY = PROPOSED_HYBRID.model_copy(update={"reranker": None})
DEPLOYED = SURVEY_DEPLOY.model_copy(update={"reranker": None})


def _by_id(ids):
    return [c for cid in ids for c in CHUNKS if c["chunk_id"] == cid]


# ------------------------------------------------------------------- el prompt
@pytest.mark.parametrize("name,digest", [
    ("SYSTEM_PROMPT", "cc27e5a8837bd1126e5854a9ddcaa0a661f31df5c5d9123ece84f503cc601c9d"),
    ("RAG_PROMPT", "65237ffd3f79338cbba18c6759653115788886c824492ab76cdc8a1debc05f21"),
    ("CROSS_CLOUD_PROMPT", "f51eeda24674f67a75d3d07730e9ba3a779dd4e63e747581cf47777533c17aed"),
    ("PROCEDURAL_PROMPT", "ae9a6b18ba44dc40b5b3fb139e92b85229671083a4ee82f0cd3cf5f2f590b67a"),
    ("NO_RAG_SYSTEM_PROMPT", "ad626597694c5a604668dcc238e6b545234bb8f19ddf60c272b76a75fbcf26bf"),
    ("NO_RAG_PROMPT", "104b81856ea344bb0ca6b949cebae05a86d88792c93f79b0d8fe815423b8ac71"),
])
def test_canonical_prompt_texts_are_frozen(name, digest):
    """Cambiar una coma del prompt es un cambio de clase B: invalida la caché y la evidencia."""
    assert hashlib.sha256(getattr(pt, name).encode("utf-8")).hexdigest() == digest


def test_standard_context_numbers_chunks_and_labels_each_source():
    chunks = [_chunk("a1", "aws", "S3", "S3 stores objects in buckets.", "Buckets > Overview"),
              _chunk("z1", "azure", "Blob Storage", "Blob Storage keeps blobs in containers.")]
    assert pt.build_context(chunks, "default") == (
        "[1] [Source: aws/S3/Buckets > Overview]\nS3 stores objects in buckets.\n\n"
        "[2] [Source: azure/Blob Storage]\nBlob Storage keeps blobs in containers.")


def test_cross_cloud_context_groups_by_provider_in_alphabetical_order():
    chunks = [_chunk("a1", "aws", "S3", "S3 stores objects in buckets.", "Buckets > Overview"),
              _chunk("z1", "azure", "Blob Storage", "Blob Storage keeps blobs in containers."),
              _chunk("a2", "aws", "EC2", "Line one.\nLine two.")]
    assert pt.build_context(chunks, "cross_cloud") == (
        "### AWS\n  [1] [Source: aws/S3/Buckets > Overview]\n  S3 stores objects in buckets.\n"
        "  [2] [Source: aws/EC2]\n  Line one.\nLine two.\n\n"
        "### AZURE\n  [1] [Source: azure/Blob Storage]\n  Blob Storage keeps blobs in containers.\n")


def test_template_routing_and_prompt_variants():
    assert pt.get_template("cross_cloud") is pt.CROSS_CLOUD_PROMPT
    assert pt.get_template("procedural") is pt.PROCEDURAL_PROMPT
    for query_type in ("conceptual", "single_provider", "default", "desconocido"):
        assert pt.get_template(query_type) is pt.RAG_PROMPT
    assert pt.variant_prompt("baseline") == (pt.SYSTEM_PROMPT, "")
    with pytest.raises(ValueError):
        pt.variant_prompt("inventada")


# ------------------------------------------------------------------- la fusión
def test_rrf_adds_reciprocal_ranks_and_breaks_ties_in_favour_of_the_bm25_leg():
    index = HybridIndex.__new__(HybridIndex)          # sin modelo de embeddings ni FAISS
    fused = index._rrf_fusion([("a", 9.0), ("b", 8.0)], [("b", 0.9), ("c", 0.8)], 60, 10)
    assert [cid for cid, _ in fused] == ["b", "a", "c"]
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert dict(fused)["a"] == pytest.approx(1 / 61) and dict(fused)["c"] == pytest.approx(1 / 62)
    # empate exacto (ambos primeros en su pierna): queda delante el de BM25, que se inserta antes
    assert [cid for cid, _ in index._rrf_fusion([("x", 1.0)], [("y", 1.0)], 60, 10)] == ["x", "y"]
    assert len(index._rrf_fusion([("a", 1.0), ("b", 1.0)], [("c", 1.0)], 60, 2)) == 2


def test_rrf_ignores_the_raw_scores_of_both_legs():
    index = HybridIndex.__new__(HybridIndex)
    low = index._rrf_fusion([("a", 0.1), ("b", 0.01)], [("b", 0.2)], 60, 10)
    high = index._rrf_fusion([("a", 900.0), ("b", 800.0)], [("b", 0.99)], 60, 10)
    assert low == high


def test_linear_fusion_min_max_normalises_each_leg():
    index = HybridIndex.__new__(HybridIndex)
    assert HybridIndex._normalize_scores([]) == []
    assert HybridIndex._normalize_scores([("a", 3.0), ("b", 3.0)]) == [("a", 1.0), ("b", 1.0)]
    fused = dict(index._linear_fusion([("a", 2.0), ("b", 1.0)], [("b", 0.9), ("c", 0.1)], 0.5, 10))
    assert fused == {"a": pytest.approx(0.5), "b": pytest.approx(0.5), "c": pytest.approx(0.0)}


# --------------------------------------------------------- el recuperador híbrido
def test_rrf_search_sends_the_raw_query_to_both_legs_unless_expansion_is_on():
    index, question = FakeIndex(), "What is EKS?"
    retriever = HybridRetriever(index)

    results = retriever.search(question, top_k=3, top_k_candidates=50, use_reranker=False)

    assert [r.chunk_id for r in results] == ["a1", "a2", "a3"]
    assert index.calls[-1] == {"query": question, "top_k": 50, "fusion": "rrf", "alpha": 0.5,
                               "top_k_candidates": 50, "rrf_k": 60, "bm25_query": None}
    assert results[0].chunk_text == CHUNKS[0]["text"] and results[0].retrieval_method == "hybrid_rrf"

    retriever.search(question, top_k=3, top_k_candidates=50, use_reranker=False, use_expansion=True)
    expanded = QueryProcessor().process(question).bm25_query
    assert index.calls[-1]["query"] == question                 # la pierna densa no se expande
    assert index.calls[-1]["bm25_query"] == expanded


def test_linear_search_sends_the_expanded_query_to_both_legs():
    index, question = FakeIndex(), "What is EKS?"
    HybridRetriever(index).search(question, top_k=3, fusion="linear", use_reranker=False)
    assert index.calls[-1]["query"] == QueryProcessor().process(question).bm25_query
    assert index.calls[-1]["bm25_query"] is None and index.calls[-1]["fusion"] == "linear"


def test_single_provider_question_filters_the_fused_candidates():
    index = FakeIndex(ranking=[("z1", 0.9), ("a1", 0.8), ("g1", 0.7), ("a2", 0.6)])
    results = HybridRetriever(index).search("What is Amazon S3?", top_k=5, use_reranker=False)
    assert [r.chunk_id for r in results] == ["a1", "a2"]


# ------------------------------------------------------------------ el formateador
def test_confidence_levels_follow_distinct_citations_and_declines():
    fmt = ResponseFormatter()
    three = "A [Source: aws/S3/x]. B [Source: aws/EC2/y]. C [Source: gcp/GKE/z]. A again [Source: aws/S3/x]."
    assert fmt.format(three, CHUNKS).confidence == "HIGH"
    assert fmt.format(three, CHUNKS).citation_count == 3          # la cita repetida cuenta una vez
    assert fmt.format("Only one [Source: aws/S3/x].", CHUNKS).confidence == "MEDIUM"
    assert fmt.format("No citations at all.", CHUNKS).confidence == "LOW"
    declined = "Based on the available documentation, I cannot find sufficient information. [Source: aws/S3/x]"
    assert fmt.format(declined, CHUNKS).confidence == "HONEST_DECLINE"   # la declinación manda
    empty = fmt.format("   ", CHUNKS)
    assert (empty.confidence, empty.text) == ("ERROR", "[Error: LLM returned an empty response]")


def test_citations_are_matched_to_the_first_chunk_of_that_provider_and_service():
    out = ResponseFormatter().format("Fact [Source: AWS/s3/Buckets/Overview]. Other [Source: oracle/DB].", CHUNKS)
    assert out.sources == [
        {"provider": "AWS", "service": "s3", "section": "Buckets/Overview", "matched_chunk_id": "a1"},
        {"provider": "oracle", "service": "DB", "section": "", "matched_chunk_id": None}]


def test_long_answers_are_cut_at_8000_characters_and_flagged():
    out = ResponseFormatter().format("x" * 9000, CHUNKS)
    assert out.was_truncated and out.original_length == 9000
    assert out.text == "x" * 8000 + "\n\n[Response truncated due to length]"


# -------------------------------------------------------- RAGPipeline.query entero
def test_legacy_config_retrieves_50_keeps_5_and_uses_the_default_template():
    question = "How do I configure AWS Lambda?"                    # procedural para el clasificador
    pipeline, index, llm = _pipeline(LEGACY)

    response = pipeline.query(question)

    assert index.calls[-1]["top_k"] == index.calls[-1]["top_k_candidates"] == 50
    assert index.calls[-1]["query"] == question and index.calls[-1]["bm25_query"] is None
    assert [c["chunk_id"] for c in response.retrieved_chunks] == ["a1", "a2", "a3", "a4", "a5"]
    # prompt_routing=False: todo sale por la plantilla por defecto (F1 del ledger de verano)
    assert llm.calls[-1]["prompt"] == pt.RAG_PROMPT.format(
        context=pt.build_context(_by_id(["a1", "a2", "a3", "a4", "a5"]), "default"), question=question)
    assert llm.calls[-1]["system_prompt"] == pt.SYSTEM_PROMPT
    assert llm.calls[-1]["temperature"] == 0.0 and llm.calls[-1]["config_name"] == LEGACY.name
    assert response.error is None and response.confidence == "MEDIUM"
    assert response.sources[0]["matched_chunk_id"] == "a1"
    assert response.hallucination_report.method == "keyword_fallback"
    assert response.hallucination_report.supported_claims == 1


def test_deployed_config_routes_the_prompt_by_query_type():
    question = "How do I configure AWS Lambda?"
    pipeline, _, llm = _pipeline(DEPLOYED)
    pipeline.query(question)
    assert llm.calls[-1]["prompt"] == pt.PROCEDURAL_PROMPT.format(
        context=pt.build_context(_by_id(["a1", "a2", "a3", "a4", "a5"]), "procedural"), question=question)


def test_deployed_config_balances_providers_only_on_cross_cloud_questions():
    question = "Compare AWS and Azure object storage"
    balanced, _, llm = _pipeline(DEPLOYED)
    legacy, _, _ = _pipeline(LEGACY)

    balanced_ids = [c["chunk_id"] for c in balanced.query(question).retrieved_chunks]
    legacy_ids = [c["chunk_id"] for c in legacy.query(question).retrieved_chunks]

    assert legacy_ids == ["a1", "a2", "a3", "a4", "a5"]                 # top-5 solo de AWS
    assert balanced_ids == ["a1", "a2", "a3", "z1", "z2"]               # cuota ceil(5/2) = 3
    assert llm.calls[-1]["prompt"] == pt.CROSS_CLOUD_PROMPT.format(
        context_by_provider=pt.build_context(_by_id(balanced_ids), "cross_cloud"), question=question)

    single, _, _ = _pipeline(DEPLOYED)
    assert [c["chunk_id"] for c in single.query("What is object storage?").retrieved_chunks] == legacy_ids


def test_reranker_sees_the_whole_pool_only_when_balancing():
    plain, deployed = FakeReranker(), FakeReranker()
    legacy_pipeline, _, _ = _pipeline(LEGACY, reranker=plain)
    deployed_pipeline, _, _ = _pipeline(DEPLOYED, reranker=deployed)

    legacy_ids = [c["chunk_id"] for c in legacy_pipeline.query("Compare AWS and Azure object storage").retrieved_chunks]
    deployed_ids = [c["chunk_id"] for c in deployed_pipeline.query("Compare AWS and Azure object storage").retrieved_chunks]

    assert plain.top_ks == [5] and deployed.top_ks == [50]
    assert legacy_ids == ["g1", "z2", "z1", "a6", "a5"]                 # orden del reranker, top-5
    assert deployed_ids == ["z2", "z1", "a6", "a5", "a4"]               # g1 no es un proveedor pedido


def test_no_candidates_returns_the_canned_message_without_calling_the_llm():
    pipeline, _, llm = _pipeline(LEGACY, index=FakeIndex(ranking=[]))
    response = pipeline.query("What is object storage?")
    assert response.answer == "No relevant documentation found for this query."
    assert response.error == "No chunks retrieved" and response.retrieved_chunks == []
    assert llm.calls == []


def test_llm_failure_is_reported_through_error_and_a_canned_answer():
    pipeline, _, _ = _pipeline(LEGACY, llm=FakeLLM(error="All 3 retries failed: boom"))
    response = pipeline.query("What is object storage?")
    assert response.answer == "Error generating response: All 3 retries failed: boom"
    assert response.error == "All 3 retries failed: boom"
    assert len(response.retrieved_chunks) == 5 and response.hallucination_report is None


def test_an_exception_during_retrieval_becomes_a_pipeline_error_response():
    pipeline, _, llm = _pipeline(LEGACY, index=FakeIndex(error=RuntimeError("faiss roto")))
    response = pipeline.query("What is object storage?")
    assert (response.answer, response.error) == ("Pipeline error: faiss roto", "faiss roto")
    assert llm.calls == []


def test_no_rag_config_uses_its_own_prompts_and_the_no_evidence_path():
    question = "What is Amazon S3?"
    pipeline, index, llm = _pipeline(LLM_ONLY_NO_RAG, llm=FakeLLM(text="I'm not familiar with that service."))

    response = pipeline.query(question)

    assert index is None and response.retrieved_chunks == []
    assert llm.calls[-1]["prompt"] == pt.NO_RAG_PROMPT.format(question=question)
    assert llm.calls[-1]["system_prompt"] == pt.NO_RAG_SYSTEM_PROMPT
    assert response.confidence == "HONEST_DECLINE"
    report = response.hallucination_report
    # N3/N5: sin evidencia toda afirmación es `unsupported_no_evidence`, incluida la declinación
    assert (report.method, report.total_claims, report.faithfulness_score) == ("no_evidence", 1, 0.0)
    assert report.claim_details[0].status == "unsupported_no_evidence"
