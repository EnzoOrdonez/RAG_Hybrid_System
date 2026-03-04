"""Document Explorer page for browsing the cloud documentation corpus."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks" / "adaptive" / "size_500"


def _load_all_chunks() -> list:
    """Load all chunks from the default chunking strategy."""
    if "all_chunks" in st.session_state:
        return st.session_state.all_chunks

    chunks = []
    if not CHUNKS_DIR.exists():
        return chunks

    for provider_dir in sorted(CHUNKS_DIR.iterdir()):
        if not provider_dir.is_dir():
            continue
        for chunk_file in sorted(provider_dir.glob("*.json")):
            try:
                data = json.loads(chunk_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    chunks.extend(data)
                elif isinstance(data, dict):
                    chunks.append(data)
            except Exception:
                continue

    st.session_state.all_chunks = chunks
    return chunks


def _get_corpus_stats(chunks: list) -> dict:
    """Compute corpus statistics."""
    stats = {
        "total_chunks": len(chunks),
        "by_provider": {},
        "by_doc_type": {},
        "token_sizes": [],
    }

    for chunk in chunks:
        provider = chunk.get("cloud_provider", "unknown")
        stats["by_provider"][provider] = stats["by_provider"].get(provider, 0) + 1

        doc_type = chunk.get("doc_type", "unknown")
        stats["by_doc_type"][doc_type] = stats["by_doc_type"].get(doc_type, 0) + 1

        tokens = chunk.get("tokens", 0)
        if tokens:
            stats["token_sizes"].append(tokens)

    return stats


def render():
    """Render the Document Explorer page."""
    st.header("Document Explorer")

    chunks = _load_all_chunks()

    if not chunks:
        st.info(
            "No chunks data available. Run the ingestion pipeline first.\n\n"
            f"Expected path: `{CHUNKS_DIR}`"
        )
        return

    # Tabs for different views
    tab_search, tab_stats, tab_browse = st.tabs(["Search", "Statistics", "Browse"])

    # === SEARCH TAB ===
    with tab_search:
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("Search in corpus", placeholder="e.g., VPC, Lambda, auto-scaling")
        with col2:
            providers = sorted(set(c.get("cloud_provider", "unknown") for c in chunks))
            provider_filter = st.selectbox("Filter by Provider", ["All"] + providers)

        if search_term:
            filtered = []
            search_lower = search_term.lower()
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                heading = chunk.get("heading_path", "").lower()
                service = chunk.get("service_name", "").lower()
                if search_lower in text or search_lower in heading or search_lower in service:
                    if provider_filter == "All" or chunk.get("cloud_provider", "") == provider_filter:
                        filtered.append(chunk)

            st.write(f"Found **{len(filtered)}** chunks matching '{search_term}'")

            for i, chunk in enumerate(filtered[:50]):  # Limit display to 50
                from src.ui.components.provider_colors import get_provider_badge
                provider = chunk.get("cloud_provider", "unknown")
                badge = get_provider_badge(provider)
                heading = chunk.get("heading_path", "N/A")
                service = chunk.get("service_name", "")
                text_preview = chunk.get("text", "")[:400]
                tokens = chunk.get("tokens", "?")

                with st.expander(f"{provider.upper()} | {service} - {heading[:80]}", expanded=(i < 3)):
                    st.markdown(badge, unsafe_allow_html=True)
                    st.caption(f"Tokens: {tokens} | Service: {service} | Type: {chunk.get('doc_type', 'N/A')}")
                    st.text(text_preview)
                    chunk_id = chunk.get("chunk_id", "N/A")
                    st.caption(f"Chunk ID: {chunk_id}")

            if len(filtered) > 50:
                st.info(f"Showing 50 of {len(filtered)} results. Refine your search for more specific results.")

    # === STATISTICS TAB ===
    with tab_stats:
        stats = _get_corpus_stats(chunks)

        st.metric("Total Chunks", f"{stats['total_chunks']:,}")

        # Chunks by provider
        st.subheader("Chunks by Provider")
        try:
            import plotly.graph_objects as go
            from src.ui.components.provider_colors import PROVIDER_COLORS

            providers_data = stats["by_provider"]
            fig = go.Figure(data=[go.Bar(
                x=list(providers_data.keys()),
                y=list(providers_data.values()),
                marker_color=[PROVIDER_COLORS.get(p, "#666") for p in providers_data.keys()],
            )])
            fig.update_layout(
                yaxis_title="Number of Chunks",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            import pandas as pd
            st.dataframe(pd.DataFrame(
                list(stats["by_provider"].items()),
                columns=["Provider", "Chunks"],
            ))

        # Token size distribution
        if stats["token_sizes"]:
            st.subheader("Token Size Distribution")
            try:
                import plotly.express as px
                import pandas as pd
                df = pd.DataFrame({"tokens": stats["token_sizes"]})
                fig = px.histogram(df, x="tokens", nbins=50, color_discrete_sequence=["#1B3A5C"])
                fig.update_layout(
                    xaxis_title="Tokens per Chunk",
                    yaxis_title="Count",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                import numpy as np
                sizes = stats["token_sizes"]
                st.write(f"Mean: {sum(sizes)/len(sizes):.0f} tokens")
                st.write(f"Min: {min(sizes)}, Max: {max(sizes)}")

        # By doc type
        if stats["by_doc_type"]:
            st.subheader("Chunks by Document Type")
            import pandas as pd
            st.dataframe(
                pd.DataFrame(
                    list(stats["by_doc_type"].items()),
                    columns=["Doc Type", "Count"],
                ).sort_values("Count", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    # === BROWSE TAB ===
    with tab_browse:
        st.subheader("Browse by Provider and Service")

        browse_provider = st.selectbox(
            "Provider",
            sorted(set(c.get("cloud_provider", "unknown") for c in chunks)),
            key="browse_provider",
        )

        provider_chunks = [c for c in chunks if c.get("cloud_provider") == browse_provider]
        services = sorted(set(c.get("service_name", "N/A") for c in provider_chunks))

        browse_service = st.selectbox("Service", ["All"] + services, key="browse_service")

        if browse_service != "All":
            display_chunks = [c for c in provider_chunks if c.get("service_name") == browse_service]
        else:
            display_chunks = provider_chunks

        st.write(f"**{len(display_chunks)}** chunks")

        for i, chunk in enumerate(display_chunks[:30]):
            heading = chunk.get("heading_path", "N/A")
            text = chunk.get("text", "")[:300]
            tokens = chunk.get("tokens", "?")

            with st.expander(f"[{i+1}] {heading[:80]}", expanded=False):
                st.caption(f"Tokens: {tokens} | Doc Type: {chunk.get('doc_type', 'N/A')}")
                st.text(text)

        if len(display_chunks) > 30:
            st.info(f"Showing 30 of {len(display_chunks)} chunks. Use Search tab for specific content.")
