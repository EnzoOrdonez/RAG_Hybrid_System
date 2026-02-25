"""
8 Experiment Configurations for thesis evaluation.

Each experiment tests a specific hypothesis about the RAG system.
All experiments use seed=42 for reproducibility.

Experiments:
  1. Chunking strategy comparison (5 strategies x 3 sizes = 15 combinations)
  2. Embedding model comparison
  3. Retrieval method comparison (BM25 vs Dense vs Hybrid)
  4. Reranker model comparison
  5. LLM comparison (llama3.1 vs qwen2.5 vs mistral)
  6. Ablation study (component-by-component addition)
  7. Cross-cloud terminology normalization
  8. End-to-end system comparison (3 pipelines)
"""

from typing import Any, Dict, List

from src.pipeline.pipeline_config import (
    BASELINE_LEXICAL,
    BASELINE_SEMANTIC,
    PROPOSED_HYBRID,
    PipelineConfig,
)


# ============================================================
# Experiment configuration structure
# ============================================================

class ExperimentConfig:
    """Configuration for a single experiment."""

    def __init__(
        self,
        experiment_id: str,
        name: str,
        description: str,
        hypothesis: str,
        variable: str,
        pipeline_configs: List[PipelineConfig],
        k_values: List[int] = None,
        max_queries: int = 200,
        metrics: List[str] = None,
        seed: int = 42,
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.description = description
        self.hypothesis = hypothesis
        self.variable = variable
        self.pipeline_configs = pipeline_configs
        self.k_values = k_values or [5, 10, 20]
        self.max_queries = max_queries
        self.metrics = metrics or ["retrieval", "generation", "hallucination", "latency"]
        self.seed = seed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "variable": self.variable,
            "num_configs": len(self.pipeline_configs),
            "config_names": [c.name for c in self.pipeline_configs],
            "k_values": self.k_values,
            "max_queries": self.max_queries,
            "metrics": self.metrics,
            "seed": self.seed,
        }


# ============================================================
# Experiment 1: Chunking Strategy Comparison
# ============================================================

def _exp1_configs() -> List[PipelineConfig]:
    """15 configs: 5 strategies x 3 sizes."""
    strategies = ["fixed", "recursive", "semantic", "hierarchical", "adaptive"]
    sizes = [300, 500, 700]
    configs = []
    for strategy in strategies:
        for size in sizes:
            configs.append(PipelineConfig(
                name=f"chunk_{strategy}_{size}",
                retrieval_method="hybrid",
                fusion_method="rrf",
                embedding_model="BAAI/bge-large-en-v1.5",
                reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
                query_expansion=False,
                terminology_normalization=False,
                chunking_strategy=strategy,
                chunk_size=size,
                retrieval_top_k=50,
                reranker_top_k=20,
                final_top_k=5,
                llm_provider="ollama",
                llm_model="llama3.1:8b-instruct-q4_K_M",
                temperature=0.1,
            ))
    return configs

EXP1_CHUNKING = ExperimentConfig(
    experiment_id="exp1",
    name="Chunking Strategy Comparison",
    description="Compare 5 chunking strategies across 3 chunk sizes (300, 500, 700 tokens)",
    hypothesis="Adaptive chunking with 500 tokens achieves best NDCG@5",
    variable="chunking_strategy + chunk_size",
    pipeline_configs=_exp1_configs(),
    max_queries=200,
    metrics=["retrieval", "latency"],
)


# ============================================================
# Experiment 2: Embedding Model Comparison
# ============================================================

def _exp2_configs() -> List[PipelineConfig]:
    """Compare embedding models."""
    models = [
        ("BAAI/bge-large-en-v1.5", "bge-large"),
        ("BAAI/bge-base-en-v1.5", "bge-base"),
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6"),
    ]
    configs = []
    for model_full, model_short in models:
        configs.append(PipelineConfig(
            name=f"embed_{model_short}",
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model=model_full,
            reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            reranker_top_k=20,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ))
    return configs

EXP2_EMBEDDING = ExperimentConfig(
    experiment_id="exp2",
    name="Embedding Model Comparison",
    description="Compare bge-large, bge-base, and MiniLM-L6 embedding models",
    hypothesis="bge-large-en-v1.5 (1024d) outperforms smaller models",
    variable="embedding_model",
    pipeline_configs=_exp2_configs(),
    max_queries=200,
    metrics=["retrieval", "latency"],
)


# ============================================================
# Experiment 3: Retrieval Method Comparison
# ============================================================

def _exp3_configs() -> List[PipelineConfig]:
    """BM25 vs Dense vs Hybrid."""
    return [
        PipelineConfig(
            name="retrieval_bm25",
            retrieval_method="bm25",
            reranker=None,
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
        PipelineConfig(
            name="retrieval_dense",
            retrieval_method="dense",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker=None,
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
        PipelineConfig(
            name="retrieval_hybrid_rrf",
            retrieval_method="hybrid",
            fusion_method="rrf",
            alpha=0.5,
            rrf_k=60,
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker=None,
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
        PipelineConfig(
            name="retrieval_hybrid_linear",
            retrieval_method="hybrid",
            fusion_method="linear",
            alpha=0.5,
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker=None,
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
    ]

EXP3_RETRIEVAL = ExperimentConfig(
    experiment_id="exp3",
    name="Retrieval Method Comparison",
    description="Compare BM25, Dense, Hybrid-RRF, and Hybrid-Linear retrieval",
    hypothesis="Hybrid RRF outperforms individual methods by >= 10% on NDCG@5",
    variable="retrieval_method",
    pipeline_configs=_exp3_configs(),
    max_queries=200,
    metrics=["retrieval", "latency"],
)


# ============================================================
# Experiment 4: Reranker Comparison
# ============================================================

def _exp4_configs() -> List[PipelineConfig]:
    """Compare reranker models."""
    rerankers = [
        (None, "no_reranker"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "rerank_mini6"),
        ("cross-encoder/ms-marco-MiniLM-L-12-v2", "rerank_mini12"),
        ("BAAI/bge-reranker-large", "rerank_bge_large"),
    ]
    configs = []
    for reranker, label in rerankers:
        configs.append(PipelineConfig(
            name=label,
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker=reranker,
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            reranker_top_k=20,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ))
    return configs

EXP4_RERANKER = ExperimentConfig(
    experiment_id="exp4",
    name="Reranker Model Comparison",
    description="Compare no-reranker, MiniLM-L-6, MiniLM-L-12, and bge-reranker-large",
    hypothesis="Cross-encoder reranking improves NDCG@5 by >= 15%",
    variable="reranker",
    pipeline_configs=_exp4_configs(),
    max_queries=200,
    metrics=["retrieval", "latency"],
)


# ============================================================
# Experiment 5: LLM Comparison
# ============================================================

def _exp5_configs() -> List[PipelineConfig]:
    """Compare Ollama LLM models."""
    models = [
        "llama3.1:8b-instruct-q4_K_M",
        "qwen2.5:7b-instruct",
        "mistral:7b-instruct",
    ]
    configs = []
    for model in models:
        short_name = model.split(":")[0]
        configs.append(PipelineConfig(
            name=f"llm_{short_name}",
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
            query_expansion=True,
            terminology_normalization=True,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            reranker_top_k=20,
            final_top_k=5,
            llm_provider="ollama",
            llm_model=model,
            temperature=0.1,
        ))
    return configs

EXP5_LLM = ExperimentConfig(
    experiment_id="exp5",
    name="LLM Model Comparison",
    description="Compare llama3.1:8b, qwen2.5:7b, and mistral:7b on generation quality",
    hypothesis="LLM choice impacts faithfulness score by <= 10%",
    variable="llm_model",
    pipeline_configs=_exp5_configs(),
    max_queries=200,
    metrics=["generation", "hallucination", "latency"],
)


# ============================================================
# Experiment 6: Ablation Study
# ============================================================

def _exp6_configs() -> List[PipelineConfig]:
    """Progressive component addition ablation study."""
    return [
        # Stage 1: BM25 only (baseline)
        PipelineConfig(
            name="ablation_bm25_only",
            retrieval_method="bm25",
            reranker=None,
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
        # Stage 2: + Dense (hybrid)
        PipelineConfig(
            name="ablation_+dense",
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker=None,
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
        # Stage 3: + Reranker
        PipelineConfig(
            name="ablation_+reranker",
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            reranker_top_k=20,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
        # Stage 4: + Query expansion
        PipelineConfig(
            name="ablation_+expansion",
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
            query_expansion=True,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            reranker_top_k=20,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
        # Stage 5: + Terminology normalization (FULL system)
        PipelineConfig(
            name="ablation_+normalization",
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
            query_expansion=True,
            terminology_normalization=True,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            reranker_top_k=20,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
    ]

EXP6_ABLATION = ExperimentConfig(
    experiment_id="exp6",
    name="Ablation Study",
    description="Progressive component addition: BM25 → +Dense → +Reranker → +Expansion → +Normalization",
    hypothesis="Each component adds measurable improvement; full system outperforms BM25 by >= 30%",
    variable="components",
    pipeline_configs=_exp6_configs(),
    max_queries=200,
    metrics=["retrieval", "generation", "hallucination", "latency"],
)


# ============================================================
# Experiment 7: Cross-Cloud Terminology Normalization
# ============================================================

def _exp7_configs() -> List[PipelineConfig]:
    """Test normalization on cross-cloud queries."""
    return [
        PipelineConfig(
            name="cross_cloud_no_norm",
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
            query_expansion=False,
            terminology_normalization=False,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            reranker_top_k=20,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
        PipelineConfig(
            name="cross_cloud_with_norm",
            retrieval_method="hybrid",
            fusion_method="rrf",
            embedding_model="BAAI/bge-large-en-v1.5",
            reranker="cross-encoder/ms-marco-MiniLM-L-12-v2",
            query_expansion=True,
            terminology_normalization=True,
            chunking_strategy="adaptive",
            chunk_size=500,
            retrieval_top_k=50,
            reranker_top_k=20,
            final_top_k=5,
            llm_provider="ollama",
            llm_model="llama3.1:8b-instruct-q4_K_M",
            temperature=0.1,
        ),
    ]

EXP7_CROSS_CLOUD = ExperimentConfig(
    experiment_id="exp7",
    name="Cross-Cloud Terminology Normalization",
    description="Compare retrieval quality with and without terminology normalization on cross-cloud queries",
    hypothesis="Normalization improves cross-cloud Recall@5 by >= 20%",
    variable="terminology_normalization",
    pipeline_configs=_exp7_configs(),
    max_queries=30,  # Only cross-cloud queries
    metrics=["retrieval", "generation", "hallucination", "latency"],
)


# ============================================================
# Experiment 8: End-to-End System Comparison (thesis main result)
# ============================================================

EXP8_END_TO_END = ExperimentConfig(
    experiment_id="exp8",
    name="End-to-End System Comparison",
    description="Compare the 3 thesis systems: Lexical baseline, Semantic baseline, Proposed hybrid",
    hypothesis="Proposed hybrid outperforms both baselines by >= 15% NDCG@5 and >= 10% faithfulness",
    variable="system_config",
    pipeline_configs=[
        BASELINE_LEXICAL,
        BASELINE_SEMANTIC,
        PROPOSED_HYBRID,
    ],
    max_queries=200,
    metrics=["retrieval", "generation", "hallucination", "latency"],
)


# ============================================================
# Registry
# ============================================================

EXPERIMENT_CONFIGS = {
    "exp1": EXP1_CHUNKING,
    "exp2": EXP2_EMBEDDING,
    "exp3": EXP3_RETRIEVAL,
    "exp4": EXP4_RERANKER,
    "exp5": EXP5_LLM,
    "exp6": EXP6_ABLATION,
    "exp7": EXP7_CROSS_CLOUD,
    "exp8": EXP8_END_TO_END,
}


def get_experiment(experiment_id: str) -> ExperimentConfig:
    """Get an experiment configuration by ID."""
    if experiment_id not in EXPERIMENT_CONFIGS:
        available = ", ".join(EXPERIMENT_CONFIGS.keys())
        raise ValueError(f"Unknown experiment '{experiment_id}'. Available: {available}")
    return EXPERIMENT_CONFIGS[experiment_id]


def list_experiments() -> list:
    """List all available experiments."""
    return [
        {
            "id": exp_id,
            "name": exp.name,
            "description": exp.description,
            "num_configs": len(exp.pipeline_configs),
            "max_queries": exp.max_queries,
        }
        for exp_id, exp in EXPERIMENT_CONFIGS.items()
    ]
