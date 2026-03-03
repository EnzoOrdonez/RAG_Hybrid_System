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
def load_pipeline(config_name: str, _hybrid_index=None):
    """Build a RAGPipeline with a given config."""
    from src.pipeline.pipeline_config import get_config
    from src.pipeline.rag_pipeline import RAGPipeline

    config = get_config(config_name)

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
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
