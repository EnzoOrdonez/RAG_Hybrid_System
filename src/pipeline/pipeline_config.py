"""
Pipeline configurations for the 4 systems evaluated in the thesis.

1. BASELINE_LEXICAL (Control 1): BM25 only, no reranking, no expansion
2. BASELINE_SEMANTIC (Control 2): Dense only, no reranking, no expansion
3. PROPOSED_HYBRID (Experimental): Hybrid + reranking + expansion + normalization
4. LLM_ONLY_NO_RAG (Control 0): No retrieval, no reranking. Pure LLM baseline
   used to quantify the value RAG adds over a vanilla LLM.
"""

from typing import Optional

from pydantic import BaseModel


class PipelineConfig(BaseModel):
    """Configuration for a complete RAG pipeline."""
    name: str
    retrieval_method: str  # "bm25", "dense", "hybrid", "none"
    embedding_model: Optional[str] = None
    fusion_method: Optional[str] = None  # "linear", "rrf"
    alpha: float = 0.5
    rrf_k: int = 60
    reranker: Optional[str] = None
    multidimensional_scoring: bool = False
    query_expansion: bool = False
    terminology_normalization: bool = False
    retrieval_top_k: int = 50
    reranker_top_k: int = 20
    final_top_k: int = 5
    # exp17: re-select the final top-k with a per-provider quota, ONLY on queries the
    # QueryProcessor typed as cross_cloud. Default False so every signed configuration
    # keeps its measured behaviour; the survey/deployment config turns it on. The gain
    # is a pilot (n=25, direction-consistent in 3 verifiers, not significant), so this
    # is a deployment choice, not a thesis claim. See src/retrieval/coverage_balancer.py.
    balance_cross_cloud_providers: bool = False

    # Route the prompt by query_type (cross_cloud / procedural templates) inside
    # RAGPipeline.query(). Default False = the LEGACY behaviour, and deliberately so:
    # RAGPipeline only builds a QueryProcessor when query_expansion is on, and
    # query_expansion has been False since N4/exp13, so query() has been typing every
    # question as "default". exp8 ("End-to-End System Comparison") was generated through
    # that path, and exp8_stats_corrected.csv is immutable -- flipping the default would
    # make a re-run of exp8 disagree with its own signed artifact.
    # The measured generation path (scripts/run_generation_matrix.py, exp12/16/17) builds
    # a QueryProcessor unconditionally and DOES route: 115/194 queries get a non-default
    # template (51 cross_cloud + 64 procedural). So this knob is what makes a deployed
    # pipeline match the experiments the survey recommendation rests on.
    prompt_routing: bool = False
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.0  # greedy decoding for determinism (Nota 3, see llm_manager)
    chunking_strategy: str = "adaptive"
    chunk_size: int = 500


# ============================================================
# The 3 thesis systems
# ============================================================

BASELINE_LEXICAL = PipelineConfig(
    name="RAG Lexico (BM25)",
    retrieval_method="bm25",
    embedding_model=None,
    reranker=None,
    query_expansion=False,
    terminology_normalization=False,
    retrieval_top_k=50,
    final_top_k=5,
    llm_provider="ollama",
    llm_model="llama3.1:8b-instruct-q4_K_M",
    temperature=0.0,
)

BASELINE_SEMANTIC = PipelineConfig(
    name="RAG Semantico (Dense)",
    retrieval_method="dense",
    embedding_model="BAAI/bge-large-en-v1.5",
    reranker=None,
    query_expansion=False,
    terminology_normalization=False,
    retrieval_top_k=50,
    final_top_k=5,
    llm_provider="ollama",
    llm_model="llama3.1:8b-instruct-q4_K_M",
    temperature=0.0,
)

PROPOSED_HYBRID = PipelineConfig(
    name="RAG Hibrido Propuesto",
    retrieval_method="hybrid",
    fusion_method="rrf",
    alpha=0.5,
    rrf_k=60,
    embedding_model="BAAI/bge-large-en-v1.5",
    reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
    multidimensional_scoring=True,
    # OFF per ledger N4 (exp13): with D11 correctly wired, cross-cloud
    # expansion does not improve retrieval (trends slightly worse) nor
    # faithfulness — the "+16.8%" claim is retired. exp11/exp12 also ran
    # without expansion. The D11 mechanism stays implemented and testable
    # via this flag / the exp13 runner.
    query_expansion=False,
    terminology_normalization=True,
    retrieval_top_k=50,
    reranker_top_k=20,
    final_top_k=5,
    llm_provider="ollama",
    llm_model="llama3.1:8b-instruct-q4_K_M",
    temperature=0.0,
)


# ============================================================
# Config registry
# ============================================================

LLM_ONLY_NO_RAG = PipelineConfig(
    name="LLM Only (No RAG)",
    retrieval_method="none",
    embedding_model=None,
    reranker=None,
    query_expansion=False,
    terminology_normalization=False,
    retrieval_top_k=0,
    final_top_k=0,
    llm_provider="ollama",
    llm_model="llama3.1:8b-instruct-q4_K_M",
    temperature=0.0,
)


# ============================================================
# Deployment configuration for the SUS/Likert surveys
# ============================================================
# Not a thesis system: this is what gets put in front of participants. It is
# PROPOSED_HYBRID with two knobs flipped, and nothing else -- the summer phase showed
# every other lever is inert or harmful (Tier A: rerank/top-k/order/lost-in-the-middle
# all null in 3 verifiers; exp16: anchored prompting null-to-negative, raises declination
# and cuts content), so "change nothing else" is an evidence-backed choice.
#
#   prompt_routing=True                 makes the deployed pipeline use the SAME prompt
#                                       routing the measured runs used (115/194 queries
#                                       get a non-default template). Without it the demo
#                                       silently uses the default template everywhere.
#   balance_cross_cloud_providers=True  the single positive lever of the whole phase
#                                       (exp17: coverage 7/25 -> 25/25; HHEM +0.081,
#                                       all 3 verifiers up; declination 56%->32%, more
#                                       claims, no extra copying). Pilot n=25, not
#                                       significant -- a deployment choice, not a claim.
#
# Kept OUT of PIPELINE_CONFIGS on purpose: `get_config("hybrid")` must keep returning the
# measured system so nothing in the experiment paths picks this up by accident.
SURVEY_DEPLOY = PROPOSED_HYBRID.model_copy(update={
    "name": "RAG Hibrido (despliegue encuestas)",
    "prompt_routing": True,
    "balance_cross_cloud_providers": True,
})


PIPELINE_CONFIGS = {
    "lexical": BASELINE_LEXICAL,
    "semantic": BASELINE_SEMANTIC,
    "hybrid": PROPOSED_HYBRID,
    "llm_only": LLM_ONLY_NO_RAG,
}


def get_config(name: str) -> PipelineConfig:
    """Get a pipeline configuration by name."""
    if name not in PIPELINE_CONFIGS:
        available = ", ".join(PIPELINE_CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return PIPELINE_CONFIGS[name]
