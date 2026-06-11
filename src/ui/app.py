"""
CloudRAG — Hybrid RAG for Cloud Documentation
Streamlit Application with 5 pages.

Run with: streamlit run src/ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# DEMO DEVICE POLICY (N5 demo fix, 2026-06-11): keep the in-process torch
# models (BGE embedder, reranker, NLI) on CPU so Ollama gets the whole GPU.
# Measured on the RTX 3060 6GB: with these models on CUDA, Ollama runs the
# LLM at 38% GPU (~6 tok/s, 90-120 s per answer); with the GPU freed it runs
# at 75%+ GPU (15-35 tok/s). Per-query CPU cost of the aux models is ~1-3 s.
# Applies ONLY to this Streamlit process — benchmark scripts never import
# this module and keep their own device selection. Set CLOUDRAG_DEMO_GPU=1
# to opt out (e.g. when running the UI on a >=12 GB GPU).
if os.environ.get("CLOUDRAG_DEMO_GPU") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st

st.set_page_config(
    page_title="CloudRAG — Hybrid RAG System",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page registry
PAGES = {
    "💬 Chat": "chat",
    "📊 Metrics Dashboard": "dashboard",
    "📁 Document Explorer": "explorer",
    "🧪 Evaluation Mode": "evaluation",
    "⚙️ Experiment Runner": "experiments",
}

# Sidebar navigation
with st.sidebar:
    st.title("☁️ CloudRAG")
    st.caption("Hybrid RAG for Cloud Documentation")
    st.divider()

    page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

    st.divider()

    # System status
    st.subheader("System Status")

    # Check Ollama
    from src.ui.components.index_loader import check_ollama
    ollama_ok = check_ollama()
    if ollama_ok:
        st.success("Ollama: Online", icon="✅")
    else:
        st.warning("Ollama: Offline", icon="⚠️")

    # Check indices
    import os
    indices_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "indices")
    indices_dir = os.path.normpath(indices_dir)
    if os.path.exists(indices_dir) and os.listdir(indices_dir):
        st.success("Indices: Available", icon="✅")
    else:
        st.warning("Indices: Not built", icon="⚠️")

    # Corpus stats
    chunks_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chunks", "adaptive", "size_500")
    chunks_dir = os.path.normpath(chunks_dir)
    if os.path.exists(chunks_dir):
        chunk_count = 0
        for provider_dir in os.listdir(chunks_dir):
            full_path = os.path.join(chunks_dir, provider_dir)
            if os.path.isdir(full_path):
                chunk_count += len([f for f in os.listdir(full_path) if f.endswith(".json")])
        if chunk_count > 0:
            st.info(f"Corpus: {chunk_count} chunk files", icon="📄")
    else:
        st.info("Corpus: Not loaded", icon="📄")

# Route to selected page
selected_page = PAGES[page]

if selected_page == "chat":
    from src.ui.pages.chat_page import render
    render()
elif selected_page == "dashboard":
    from src.ui.pages.dashboard_page import render
    render()
elif selected_page == "explorer":
    from src.ui.pages.explorer_page import render
    render()
elif selected_page == "evaluation":
    from src.ui.pages.evaluation_page import render
    render()
elif selected_page == "experiments":
    from src.ui.pages.experiments_page import render
    render()
