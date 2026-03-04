"""Metrics Dashboard page with Plotly visualizations."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"


def _load_aggregated_metrics(exp_id: str) -> dict:
    """Load aggregated metrics for an experiment."""
    path = RESULTS_DIR / exp_id / "aggregated_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_full_results(exp_id: str) -> dict:
    """Load full per-query results."""
    path = RESULTS_DIR / exp_id / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _get_available_experiments() -> list:
    """List experiments that have results."""
    available = []
    if not RESULTS_DIR.exists():
        return available
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir() and (d / "aggregated_metrics.json").exists():
            available.append(d.name)
    return available


def _render_exp8_dashboard(metrics: dict, full_results: dict):
    """Render End-to-End comparison dashboard (exp8)."""
    import plotly.graph_objects as go
    from src.ui.components.provider_colors import SYSTEM_COLORS

    systems = list(metrics.keys())
    if not systems:
        st.warning("No data available for exp8.")
        return

    # Overview metrics cards
    st.subheader("System Comparison")
    cols = st.columns(len(systems))
    for col, sys_name in zip(cols, systems):
        data = metrics[sys_name]
        color = SYSTEM_COLORS.get(sys_name, "#666")
        faith = data.get("hall_faithfulness_mean", 0)
        total_ms = data.get("lat_total_ms_mean", 0)
        queries = data.get("total_queries", 0)

        col.markdown(
            f'<div style="border-left:4px solid {color};padding:8px;">'
            f'<strong>{sys_name}</strong><br>'
            f'Faithfulness: {faith:.3f}<br>'
            f'Avg Latency: {total_ms/1000:.1f}s<br>'
            f'Queries: {queries}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Faithfulness comparison bar chart
    st.subheader("Faithfulness Score")
    fig_faith = go.Figure()
    for sys_name in systems:
        data = metrics[sys_name]
        color = SYSTEM_COLORS.get(sys_name, "#666")
        mean = data.get("hall_faithfulness_mean", 0)
        std = data.get("hall_faithfulness_std", 0)
        fig_faith.add_trace(go.Bar(
            name=sys_name,
            x=[sys_name],
            y=[mean],
            error_y=dict(type="data", array=[std], visible=True),
            marker_color=color,
        ))
    fig_faith.update_layout(
        yaxis_title="Faithfulness (0-1)",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400,
    )
    st.plotly_chart(fig_faith, use_container_width=True)

    # Latency comparison
    st.subheader("Latency Breakdown (Mean)")
    stages = ["query_processing_ms", "retrieval_ms", "reranking_ms", "generation_ms", "hallucination_check_ms"]
    stage_labels = ["Query Processing", "Retrieval", "Reranking", "Generation", "Hallucination Check"]

    fig_lat = go.Figure()
    for sys_name in systems:
        data = metrics[sys_name]
        color = SYSTEM_COLORS.get(sys_name, "#666")
        values = [data.get(f"lat_{s}_mean", 0) / 1000 for s in stages]  # Convert to seconds
        fig_lat.add_trace(go.Bar(
            name=sys_name,
            x=stage_labels,
            y=values,
            marker_color=color,
        ))
    fig_lat.update_layout(
        barmode="group",
        yaxis_title="Seconds",
        height=400,
    )
    st.plotly_chart(fig_lat, use_container_width=True)

    # Total latency percentiles
    st.subheader("Total Latency Percentiles")
    fig_perc = go.Figure()
    percentiles = ["p50", "p95"]
    for sys_name in systems:
        data = metrics[sys_name]
        color = SYSTEM_COLORS.get(sys_name, "#666")
        values = [data.get(f"lat_total_ms_{p}", 0) / 1000 for p in percentiles]
        fig_perc.add_trace(go.Bar(
            name=sys_name,
            x=["P50", "P95"],
            y=values,
            marker_color=color,
        ))
    fig_perc.update_layout(
        barmode="group",
        yaxis_title="Seconds",
        height=350,
    )
    st.plotly_chart(fig_perc, use_container_width=True)

    # Raw metrics table
    st.subheader("Detailed Metrics")
    import pandas as pd
    rows = []
    for sys_name in systems:
        data = metrics[sys_name]
        rows.append({
            "System": sys_name,
            "Queries": data.get("total_queries", 0),
            "Errors": data.get("errors", 0),
            "Faithfulness": f"{data.get('hall_faithfulness_mean', 0):.3f} +/- {data.get('hall_faithfulness_std', 0):.3f}",
            "Hallucination Rate": f"{data.get('hall_hallucination_rate_mean', 0):.3f}",
            "Latency P50 (s)": f"{data.get('lat_total_ms_p50', 0)/1000:.1f}",
            "Latency P95 (s)": f"{data.get('lat_total_ms_p95', 0)/1000:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_generic_dashboard(exp_id: str, metrics: dict):
    """Render a generic dashboard for any experiment."""
    import plotly.graph_objects as go
    import pandas as pd

    systems = list(metrics.keys())
    if not systems:
        st.warning(f"No data available for {exp_id}.")
        return

    st.subheader("Configurations")
    st.write(f"{len(systems)} configurations evaluated")

    # Collect all metric keys
    all_metric_keys = set()
    for data in metrics.values():
        all_metric_keys.update(data.keys())
    metric_keys = sorted([k for k in all_metric_keys if k.endswith("_mean") and not k.startswith("lat_")])

    if metric_keys:
        st.subheader("Metrics Comparison")
        selected_metric = st.selectbox("Select Metric", metric_keys)

        fig = go.Figure()
        configs = list(metrics.keys())
        values = [metrics[c].get(selected_metric, 0) for c in configs]
        std_key = selected_metric.replace("_mean", "_std")
        stds = [metrics[c].get(std_key, 0) for c in configs]

        fig.add_trace(go.Bar(
            x=configs,
            y=values,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color="#1B3A5C",
        ))
        fig.update_layout(
            yaxis_title=selected_metric,
            height=400,
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Latency metrics
    lat_keys = sorted([k for k in all_metric_keys if k.startswith("lat_") and k.endswith("_mean")])
    if lat_keys:
        st.subheader("Latency")
        rows = []
        for config in systems:
            row = {"Config": config}
            for lk in lat_keys:
                label = lk.replace("lat_", "").replace("_mean", "")
                row[label] = f"{metrics[config].get(lk, 0):.1f}ms"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Full table
    st.subheader("All Metrics")
    rows = []
    for config in systems:
        row = {"Config": config}
        for k, v in sorted(metrics[config].items()):
            if isinstance(v, float):
                row[k] = f"{v:.4f}"
            else:
                row[k] = v
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render():
    """Render the Metrics Dashboard page."""
    st.header("Metrics Dashboard")

    available = _get_available_experiments()

    if not available:
        st.info(
            "No experiments have been run yet. "
            "Use the Experiment Runner page to run experiments."
        )
        return

    # Experiment selector
    exp_descriptions = {
        "exp1": "Chunking Strategy Comparison",
        "exp2": "Embedding Model Comparison",
        "exp3": "Retrieval Method Comparison",
        "exp4": "Reranker Comparison",
        "exp5": "LLM Comparison",
        "exp6": "Ablation Study",
        "exp7": "Cross-Cloud Normalization",
        "exp8": "End-to-End System Comparison",
    }

    exp_options = [f"{e} - {exp_descriptions.get(e, e)}" for e in available]
    selected_idx = st.selectbox(
        "Select Experiment",
        range(len(exp_options)),
        format_func=lambda i: exp_options[i],
    )
    selected_exp = available[selected_idx]

    # Load data
    metrics = _load_aggregated_metrics(selected_exp)
    full_results = _load_full_results(selected_exp)

    if not metrics:
        st.warning(f"No aggregated metrics found for {selected_exp}.")
        return

    # Show existing figures if available
    existing_figs = []
    if FIGURES_DIR.exists():
        for f in FIGURES_DIR.iterdir():
            if f.suffix == ".png":
                existing_figs.append(f)

    # Render appropriate dashboard
    if selected_exp == "exp8":
        _render_exp8_dashboard(metrics, full_results)
    else:
        _render_generic_dashboard(selected_exp, metrics)

    # Show pre-generated figures
    if existing_figs:
        st.divider()
        st.subheader("Pre-generated Figures")
        for fig_path in sorted(existing_figs):
            st.image(str(fig_path), caption=fig_path.stem, use_container_width=True)
