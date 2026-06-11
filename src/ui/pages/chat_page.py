"""Chat interface page with RAG pipeline integration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import time

import streamlit as st

from src.ui.components.provider_colors import (
    PROVIDER_COLORS,
    SYSTEM_COLORS,
    faithfulness_color,
    faithfulness_label,
    get_provider_badge,
)
from src.ui.components.index_loader import check_ollama, get_ollama_models


def _format_answer_with_citations(answer: str, sources: list) -> str:
    """Add colored provider badges inline in the answer."""
    # Build a simple markdown version with source list
    if not sources:
        return answer
    return answer


def _render_latency_breakdown(latency):
    """Show latency per pipeline stage."""
    stages = [
        ("Query Processing", latency.query_processing_ms),
        ("Retrieval", latency.retrieval_ms),
        ("Reranking", latency.reranking_ms),
        ("Generation", latency.generation_ms),
        ("Hallucination Check", latency.hallucination_check_ms),
    ]
    cols = st.columns(len(stages))
    for col, (name, ms) in zip(cols, stages):
        if ms > 0:
            if ms >= 1000:
                col.metric(name, f"{ms/1000:.1f}s")
            else:
                col.metric(name, f"{ms:.0f}ms")


def render():
    """Render the Chat Interface page."""
    st.header("Chat Interface")

    # Sidebar controls
    with st.sidebar:
        st.subheader("Configuration")

        system_options = {
            "RAG Hibrido Propuesto": "hybrid",
            "RAG Lexico (BM25)": "lexical",
            "RAG Semantico (Dense)": "semantic",
        }
        selected_system = st.selectbox(
            "System",
            list(system_options.keys()),
            index=0,
        )
        config_key = system_options[selected_system]

        # LLM model selection (wired: injected into the pipeline below).
        # Default llama3.1 q4: the only model that fits the 6 GB GPU whole
        # (gemma4:e4b is 9.6 GB -> permanent CPU offload on this machine).
        ollama_ok = check_ollama()
        if ollama_ok:
            models = get_ollama_models()
            if models:
                default_model = "llama3.1:8b-instruct-q4_K_M"
                model_idx = 0
                for i, m in enumerate(models):
                    if default_model in m or m in default_model:
                        model_idx = i
                        break
                selected_model = st.selectbox("LLM Model", models, index=model_idx)
            else:
                selected_model = None
                st.warning("No models found in Ollama")
        else:
            selected_model = None
            st.warning("Ollama not running - retrieval only mode")

        verify_nli = st.toggle(
            "Verificación de fidelidad (NLI)",
            value=True,
            help="Corre el verificador NLI DESPUÉS de mostrar la respuesta "
                 "(no bloquea la generación).",
        )

        st.divider()

        # Advanced settings
        with st.expander("Advanced Settings"):
            top_k = st.slider("Retrieved chunks (K)", 1, 20, 5)
            enable_reranking = st.checkbox(
                "Enable Re-ranking",
                value=(config_key == "hybrid"),
            )
            enable_expansion = st.checkbox(
                "Query Expansion",
                value=False,  # OFF por defecto: N4 — la expansión no aporta (exp13)
            )
            alpha = st.slider("Hybrid Alpha", 0.0, 1.0, 0.5, 0.1)
            max_tokens = st.slider("Max tokens (demo)", 128, 1024, 512, 64)

        if st.button("Clear Chat History"):
            st.session_state.pop("chat_messages", None)
            st.rerun()

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if msg.get("chunks"):
                with st.expander(f"Retrieved Chunks ({len(msg['chunks'])})"):
                    for i, chunk in enumerate(msg["chunks"]):
                        provider = chunk.get("cloud_provider", "unknown")
                        badge = get_provider_badge(provider)
                        heading = chunk.get("heading_path", "")
                        text_preview = chunk.get("text", "")[:300]
                        st.markdown(
                            f"{badge} **{heading}**\n\n{text_preview}...",
                            unsafe_allow_html=True,
                        )
                        if i < len(msg["chunks"]) - 1:
                            st.divider()

    # Chat input
    user_input = st.chat_input("Ask about cloud documentation...")

    if user_input:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process query: staged progress + token streaming + deferred NLI.
        with st.chat_message("assistant"):
            try:
                import copy

                from src.ui.components.index_loader import (
                    load_hybrid_index,
                    load_pipeline,
                    warm_model,
                )

                hybrid_index = load_hybrid_index()
                base_pipeline = load_pipeline(
                    config_key, _hybrid_index=hybrid_index,
                    llm_model=selected_model or "",
                )

                # Warm the model once per (session, model): first query and
                # post-pause queries skip the 10-30 s cold load.
                warm_key = f"warmed::{selected_model}"
                if selected_model and not st.session_state.get(warm_key):
                    with st.spinner(f"Cargando modelo {selected_model}..."):
                        warm_model(selected_model, keep_alive="30m")
                    st.session_state[warm_key] = True

                # Per-render overrides on COPIES — cached pipeline/config are
                # shared across sessions and must never be mutated.
                pipeline = copy.copy(base_pipeline)
                pipeline.config = base_pipeline.config.model_copy()
                pipeline.config.final_top_k = top_k
                pipeline.config.query_expansion = enable_expansion
                if not enable_reranking:
                    pipeline.reranker = None

                stage_labels = {
                    "retrieval": "Recuperando documentación...",
                    "reranking": "Re-rankeando evidencia...",
                    "generation": "Generando respuesta (streaming)...",
                }
                status = st.status("Procesando consulta...", expanded=False)
                answer_area = st.empty()
                acc = ""
                payload = {}
                for kind, value in pipeline.query_stream(
                        user_input, max_tokens=max_tokens, keep_alive="30m"):
                    if kind == "stage":
                        status.update(label=stage_labels.get(value, value))
                    elif kind == "token":
                        acc += value
                        answer_area.markdown(acc + "▌")
                    else:  # done
                        payload = value
                status.update(label="Respuesta completa", state="complete")

                answer_text = payload.get("answer") or acc or "No response generated."
                answer_area.markdown(answer_text)
                if payload.get("error"):
                    st.error(payload["error"])

                chunks = payload.get("retrieved_chunks") or []

                # Deferred verification: AFTER the user already has the answer.
                if verify_nli and chunks and not payload.get("error"):
                    with st.spinner("Verificando fidelidad (NLI)..."):
                        report, formatted, nli_ms = pipeline.verify_answer(
                            answer_text, chunks)
                    faith = report.faithfulness_score
                    color = faithfulness_color(faith)
                    label = faithfulness_label(faith)
                    st.markdown(
                        f"**Faithfulness:** "
                        f'<span style="color:{color};font-weight:bold;">'
                        f"{faith:.2f} ({label})</span>"
                        f"<span style='color:gray;'> · verificación {nli_ms/1000:.1f}s"
                        f" · instrumento NLI, lectura relativa (N5)</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(faith)
                    if formatted.confidence and formatted.confidence != "UNKNOWN":
                        st.caption(f"Confidence: {formatted.confidence}")

                lat = payload.get("latency")
                if lat is not None and lat.total_ms > 0:
                    with st.expander("Latency Breakdown"):
                        _render_latency_breakdown(lat)
                        st.caption(f"Total: {lat.total_ms/1000:.2f}s")

                if chunks:
                    with st.expander(f"Retrieved Chunks ({len(chunks)})"):
                        for i, chunk in enumerate(chunks):
                            provider = chunk.get("cloud_provider", "unknown")
                            badge = get_provider_badge(provider)
                            heading = chunk.get("heading_path", "")
                            text_preview = chunk.get("text", "")[:300]
                            st.markdown(
                                f"{badge} **{heading}**\n\n{text_preview}...",
                                unsafe_allow_html=True,
                            )
                            if i < len(chunks) - 1:
                                st.divider()

                # Save to history
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "chunks": chunks,
                })

            except Exception as e:
                error_msg = f"Pipeline error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
