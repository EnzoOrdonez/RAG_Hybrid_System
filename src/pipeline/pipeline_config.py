"""
Pipeline configurations for the 3 systems evaluated in the thesis.

1. BASELINE_LEXICAL (Control 1): BM25 only, no reranking, no expansion
2. BASELINE_SEMANTIC (Control 2): Dense only, no reranking, no expansion
3. PROPOSED_HYBRID (Experimental): Hybrid + reranking + expansion + normalization
"""

from typing import Optional

from pydantic import BaseModel


class PipelineConfig(BaseModel):
    """Configuration for a complete RAG pipeline."""
    name: str
    retrieval_method: str  # "bm25", "dense", "hybrid"
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
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.1
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
    temperature=0.1,
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
    temperature=0.1,
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
    query_expansion=True,
    terminology_normalization=True,
    retrieval_top_k=50,
    reranker_top_k=20,
    final_top_k=5,
    llm_provider="ollama",
    llm_model="llama3.1:8b-instruct-q4_K_M",
    temperature=0.1,
)


# ============================================================
# Config registry
# ============================================================

PIPELINE_CONFIGS = {
    "lexical": BASELINE_LEXICAL,
    "semantic": BASELINE_SEMANTIC,
    "hybrid": PROPOSED_HYBRID,
}


def get_config(name: str) -> PipelineConfig:
    """Get a pipeline configuration by name."""
    if name not in PIPELINE_CONFIGS:
        available = ", ".join(PIPELINE_CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return PIPELINE_CONFIGS[name]
