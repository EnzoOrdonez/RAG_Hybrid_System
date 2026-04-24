"""
Load indices and models once using st.cache_resource.
Avoids reloading heavy models on each Streamlit rerun.
"""

import logging

import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading indices...")
def load_hybrid_index():
    """Load FAISS + BM25 hybrid index."""
    from src.pipeline.rag_pipeline import load_hybrid_index as _load
    return _load(
        embedding_model="bge-large",
        chunking_strategy="adaptive",
        chunk_size=500,
    )


@st.cache_resource(show_spinner="Building pipeline...")
def load_pipeline(
    config_name: str,
    llm_model: str = None,
    enable_reranking: bool = None,
    enable_query_expansion: bool = None,
    alpha: float = None,
    final_top_k: int = None,
    _hybrid_index=None,
):
    """Build a RAGPipeline with UI overrides encoded in the cache key."""
    from src.pipeline.pipeline_config import get_config
    from src.pipeline.rag_pipeline import RAGPipeline

    config = get_config(config_name)

    if llm_model:
        config.llm_model = llm_model
    if enable_reranking is False:
        config.reranker = None
        config.multidimensional_scoring = False
    if enable_query_expansion is not None:
        config.query_expansion = enable_query_expansion
        if not enable_query_expansion:
            config.terminology_normalization = False
    if alpha is not None:
        config.alpha = alpha
    if final_top_k is not None:
        config.final_top_k = final_top_k

    if _hybrid_index is None:
        _hybrid_index = load_hybrid_index()

    return RAGPipeline(config=config, hybrid_index=_hybrid_index)


def check_ollama() -> bool:
    """Check if Ollama is reachable."""
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_ollama_models() -> list:
    """Return list of model names available in Ollama."""
    import json
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return [model["name"] for model in data.get("models", [])]
    except Exception:
        return []
