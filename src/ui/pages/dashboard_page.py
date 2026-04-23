"""Metrics Dashboard page with Plotly visualizations."""

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "output" / "figures"


def _load_aggregated_metrics(exp_id: str) -> dict:
    path = RESULTS_DIR / exp_id / "aggregated_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_full_results(exp_id: str) -> dict:
    path = RESULTS_DIR / exp_id / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _get_available_experiments() -> list:
    available = []
    if not RESULTS_DIR.exists():
        return available
    for directory in sorted(RESULTS_DIR.iterdir()):
        if directory.is_dir() and (directory / "aggregated_metrics.json").exists():
            available.append(directory.name)
    return available


def _render_exp8_dashboard(metrics: dict, full_results: dict, exp_id: str):
    """Render End-to-End comparison dashboard for exp8/exp8b."""
    import pandas as pd
    import plotly.graph_objects as go

    from src.ui.components.provider_colors import SYSTEM_COLORS

    systems = list(metrics.keys())
    if not systems:
        st.warning(f"No data available for {exp_id}.")
        return

    if exp_id == "exp8b":
        st.caption("Official thesis baseline: exp8b uses Mistral 7B and is the primary comparison.")
    elif exp_id == "exp8":
        st.caption("Comparison baseline: exp8 uses Llama 3.1 and is kept for reference.")

    st.subheader("System Comparison")
    cols = st.columns(len(systems))
    for col, sys_name in zip(cols, systems):
        data = metrics[sys_name]
        color = SYSTEM_COLORS.get(sys_name, "#666")
        faith = data.get("hall_faithfulness_mean", 0)
        n_eff = data.get("hall_n_effective")
        n_tot = data.get("hall_n_total")
        total_ms = data.get("lat_total_ms_mean", 0)
        queries = data.get("total_queries", 0)

        # Show effective N next to faithfulness when the benchmark has
        # filtered method=none/error (Flag 137/140 fix).
        if n_eff is not None and n_tot is not None:
            faith_line = f"Faithfulness: {faith:.3f} (n={n_eff}/{n_tot})"
        else:
            faith_line = f"Faithfulness: {faith:.3f}"

        col.markdown(
            f'<div style="border-left:4px solid {color};padding:8px;">'
            f'<strong>{sys_name}</strong><br>'
            f'{faith_line}<br>'
            f'Avg Latency: {total_ms/1000:.1f}s<br>'
            f'Queries: {queries}'
            f'</div>',
            unsafe_allow_html=True,
        )

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

    st.subheader("Latency Breakdown (Mean)")
    stages = ["query_processing_ms", "retrieval_ms", "reranking_ms", "generation_ms", "hallucination_check_ms"]
    stage_labels = ["Query Processing", "Retrieval", "Reranking", "Generation", "Hallucination Check"]

    fig_lat = go.Figure()
    for sys_name in systems:
        data = metrics[sys_name]
        color = SYSTEM_COLORS.get(sys_name, "#666")
        values = [data.get(f"lat_{stage}_mean", 0) / 1000 for stage in stages]
        fig_lat.add_trace(go.Bar(name=sys_name, x=stage_labels, y=values, marker_color=color))
    fig_lat.update_layout(barmode="group", yaxis_title="Seconds", height=400)
    st.plotly_chart(fig_lat, use_container_width=True)

    st.subheader("Total Latency Percentiles")
    fig_perc = go.Figure()
    for sys_name in systems:
        data = metrics[sys_name]
        color = SYSTEM_COLORS.get(sys_name, "#666")
        values = [data.get("lat_total_ms_p50", 0) / 1000, data.get("lat_total_ms_p95", 0) / 1000]
        fig_perc.add_trace(go.Bar(name=sys_name, x=["P50", "P95"], y=values, marker_color=color))
    fig_perc.update_layout(barmode="group", yaxis_title="Seconds", height=350)
    st.plotly_chart(fig_perc, use_container_width=True)

    st.subheader("Detailed Metrics")
    rows = []
    for sys_name in systems:
        data = metrics[sys_name]
        n_eff = data.get("hall_n_effective")
        n_tot = data.get("hall_n_total")
        n_suffix = f" (n={n_eff}/{n_tot})" if n_eff is not None and n_tot is not None else ""
        rows.append({
            "System": sys_name,
            "Queries": data.get("total_queries", 0),
            "Errors": data.get("errors", 0),
            "Faithfulness": (
                f"{data.get('hall_faithfulness_mean', 0):.3f} +/- "
                f"{data.get('hall_faithfulness_std', 0):.3f}{n_suffix}"
            ),
            "Hallucination Rate": f"{data.get('hall_hallucination_rate_mean', 0):.3f}",
            "Latency P50 (s)": f"{data.get('lat_total_ms_p50', 0)/1000:.1f}",
            "Latency P95 (s)": f"{data.get('lat_total_ms_p95', 0)/1000:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_generic_dashboard(exp_id: str, metrics: dict):
    import pandas as pd
    import plotly.graph_objects as go

    systems = list(metrics.keys())
    if not systems:
        st.warning(f"No data available for {exp_id}.")
        return

    st.subheader("Configurations")
    st.write(f"{len(systems)} configurations evaluated")

    all_metric_keys = set()
    for data in metrics.values():
        all_metric_keys.update(data.keys())
    metric_keys = sorted([key for key in all_metric_keys if key.endswith("_mean") and not key.startswith("lat_")])

    if metric_keys:
        st.subheader("Metrics Comparison")
        selected_metric = st.selectbox("Select Metric", metric_keys)

        fig = go.Figure()
        values = [metrics[config].get(selected_metric, 0) for config in systems]
        std_key = selected_metric.replace("_mean", "_std")
        stds = [metrics[config].get(std_key, 0) for config in systems]

        fig.add_trace(go.Bar(
            x=systems,
            y=values,
            error_y=dict(type="data", array=stds, visible=True),
            marker_color="#1B3A5C",
        ))
        fig.update_layout(yaxis_title=selected_metric, height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    lat_keys = sorted([key for key in all_metric_keys if key.startswith("lat_") and key.endswith("_mean")])
    if lat_keys:
        st.subheader("Latency")
        rows = []
        for config in systems:
            row = {"Config": config}
            for lat_key in lat_keys:
                label = lat_key.replace("lat_", "").replace("_mean", "")
                row[label] = f"{metrics[config].get(lat_key, 0):.1f}ms"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("All Metrics")
    rows = []
    for config in systems:
        row = {"Config": config}
        for key, value in sorted(metrics[config].items()):
            row[key] = f"{value:.4f}" if isinstance(value, float) else value
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render():
    """Render the Metrics Dashboard page."""
    st.header("Metrics Dashboard")

    available = _get_available_experiments()
    if not available:
        st.info("No experiments have been run yet. Use the Experiment Runner page to run experiments.")
        return

    exp_descriptions = {
        "exp1": "Chunking Strategy Comparison",
        "exp2": "Embedding Model Comparison",
        "exp3": "Retrieval Method Comparison",
        "exp4": "Reranker Comparison",
        "exp5": "LLM Comparison",
        "exp6": "Ablation Study",
        "exp7": "Cross-Cloud Normalization",
        "exp8": "End-to-End System Comparison (Llama baseline)",
        "exp8b": "End-to-End System Comparison (Official Mistral baseline)",
    }

    exp_options = [f"{exp_id} - {exp_descriptions.get(exp_id, exp_id)}" for exp_id in available]
    default_idx = available.index("exp8b") if "exp8b" in available else 0
    selected_idx = st.selectbox(
        "Select Experiment",
        range(len(exp_options)),
        index=default_idx,
        format_func=lambda idx: exp_options[idx],
    )
    selected_exp = available[selected_idx]

    metrics = _load_aggregated_metrics(selected_exp)
    full_results = _load_full_results(selected_exp)

    if not metrics:
        st.warning(f"No aggregated metrics found for {selected_exp}.")
        return

    existing_figs = []
    if FIGURES_DIR.exists():
        for fig_file in FIGURES_DIR.iterdir():
            if fig_file.suffix == ".png":
                existing_figs.append(fig_file)

    if selected_exp in {"exp8", "exp8b"}:
        _render_exp8_dashboard(metrics, full_results, selected_exp)
    else:
        _render_generic_dashboard(selected_exp, metrics)

    if existing_figs:
        st.divider()
        st.subheader("Pre-generated Figures")
        for fig_path in sorted(existing_figs):
            st.image(str(fig_path), caption=fig_path.stem, use_container_width=True)
