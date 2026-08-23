"""
Load indices and models once using st.cache_resource.
Avoids reloading heavy models on each Streamlit rerun.
"""

import logging

import streamlit as st

logger = logging.getLogger(__name__)

# UI/deployment defaults mirror the exp19b generation recipe. The immutable
# configurations used by experiments remain untouched.
UI_GENERATOR_SEED = 42
UI_MAX_TOKENS = 1024


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
def load_pipeline(config_name: str, _hybrid_index=None, llm_model: str = ""):
    """Build a RAGPipeline with a given config.

    Streamlit fixes its defaults to exp19b's Granite model, seed 42 and
    cache-off recipe. ``llm_model`` may explicitly override the model from the
    chat selector. Each (config, model) pair has its own cached pipeline; the
    experiment registry itself remains unchanged.
    """
    from src.pipeline.pipeline_config import SURVEY_DEPLOY, get_config
    from src.pipeline.rag_pipeline import RAGPipeline

    # UI-only mapping: hybrid consumes the full summer deployment; controls keep
    # their retrieval recipe but share its measured generator. Experiment and CLI
    # configs remain unchanged.
    base_config = SURVEY_DEPLOY if config_name == "hybrid" else get_config(config_name)
    effective_model = llm_model or SURVEY_DEPLOY.llm_model
    config = base_config.model_copy(update={"llm_model": effective_model})

    if _hybrid_index is None:
        _hybrid_index = load_hybrid_index()

    from src.generation.llm_manager import LLMManager
    # Cache-off both avoids latency bias between participants and matches exp19b.
    llm = LLMManager(
        provider="ollama",
        model=effective_model,
        cache_enabled=False,
        seed=UI_GENERATOR_SEED,
    )
    return RAGPipeline(config=config, hybrid_index=_hybrid_index, llm_manager=llm)


def warm_model(model: str, keep_alive: str = "30m") -> bool:
    """Load `model` into Ollama memory with a long keep_alive (demo warm-up).

    POST /api/generate with no prompt just loads the model; returns fast.
    Prevents the first query of a session (and queries after long pauses)
    from paying the 10-30 s cold load inside the user-visible request.
    """
    import json
    import urllib.request
    try:
        body = json.dumps({"model": model, "keep_alive": keep_alive}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status == 200
    except Exception:
        return False


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
