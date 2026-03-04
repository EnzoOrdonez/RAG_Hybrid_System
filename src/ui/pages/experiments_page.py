"""Experiment Runner page for executing experiments from the UI."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def _get_experiment_status() -> dict:
    """Check which experiments have been run."""
    from experiments.experiment_configs import EXPERIMENT_CONFIGS

    status = {}
    for exp_id, config in EXPERIMENT_CONFIGS.items():
        agg_path = RESULTS_DIR / exp_id / "aggregated_metrics.json"
        checkpoint_files = list((RESULTS_DIR / exp_id).glob("checkpoint_*.json")) if (RESULTS_DIR / exp_id).exists() else []

        if agg_path.exists():
            status[exp_id] = {
                "state": "completed",
                "name": config.name,
                "description": config.description,
                "hypothesis": config.hypothesis,
                "num_configs": len(config.pipeline_configs),
                "max_queries": config.max_queries,
            }
        elif checkpoint_files:
            status[exp_id] = {
                "state": "partial",
                "name": config.name,
                "description": config.description,
                "hypothesis": config.hypothesis,
                "num_configs": len(config.pipeline_configs),
                "max_queries": config.max_queries,
                "checkpoints": len(checkpoint_files),
            }
        else:
            status[exp_id] = {
                "state": "pending",
                "name": config.name,
                "description": config.description,
                "hypothesis": config.hypothesis,
                "num_configs": len(config.pipeline_configs),
                "max_queries": config.max_queries,
            }

    return status


def render():
    """Render the Experiment Runner page."""
    st.header("Experiment Runner")

    # Get experiment status
    try:
        status = _get_experiment_status()
    except Exception as e:
        st.error(f"Error loading experiment configs: {e}")
        return

    # Overview table
    st.subheader("Experiment Status")

    state_icons = {"completed": "done", "partial": "warning", "pending": "schedule"}

    for exp_id in sorted(status.keys()):
        info = status[exp_id]
        state = info["state"]

        if state == "completed":
            icon = "✅"
        elif state == "partial":
            icon = "⚠️"
        else:
            icon = "⏳"

        with st.expander(f"{icon} {exp_id} — {info['name']}", expanded=(state != "completed")):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Hypothesis:** {info['hypothesis']}")
            st.write(f"**Configs:** {info['num_configs']} | **Max Queries:** {info['max_queries']}")

            if state == "completed":
                st.success("Experiment completed. View results in Metrics Dashboard.")
            elif state == "partial":
                st.warning(f"Partially complete ({info.get('checkpoints', 0)} checkpoints found). Can resume.")

    st.divider()

    # Run experiment section
    st.subheader("Run Experiment")

    from src.ui.components.index_loader import check_ollama

    ollama_ok = check_ollama()
    if not ollama_ok:
        st.warning(
            "Ollama is not running. Experiments requiring LLM generation (exp5, exp6, exp7, exp8) "
            "will only compute retrieval metrics."
        )

    # Experiment selector
    exp_options = [f"{eid} - {info['name']}" for eid, info in sorted(status.items())]
    exp_ids = sorted(status.keys())
    selected_idx = st.selectbox(
        "Select Experiment",
        range(len(exp_options)),
        format_func=lambda i: exp_options[i],
        key="run_exp_select",
    )
    selected_exp = exp_ids[selected_idx]

    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio("Mode", ["Quick (20 queries)", "Full (200 queries)"], horizontal=True)
    with col2:
        resume = st.checkbox("Resume from checkpoint", value=True)

    max_queries = 20 if "Quick" in mode else 200

    # Check for running experiment
    if "experiment_running" not in st.session_state:
        st.session_state.experiment_running = False

    if st.session_state.experiment_running:
        st.info("An experiment is currently running. Please wait...")
        return

    if st.button("Run Experiment", type="primary", disabled=st.session_state.experiment_running):
        st.session_state.experiment_running = True

        try:
            from experiments.experiment_configs import get_experiment
            from src.evaluation.benchmark_runner import BenchmarkRunner
            from src.evaluation.test_queries import load_queries

            exp_config = get_experiment(selected_exp)

            # Load queries
            queries_path = PROJECT_ROOT / "data" / "evaluation" / "test_queries.json"
            progress_bar = st.progress(0, text="Loading queries...")

            try:
                queries = load_queries(str(queries_path))
            except Exception:
                st.warning("Test queries not found. Generating...")
                from src.evaluation.test_queries import generate_all_queries, save_queries
                queries = generate_all_queries(count=200)
                save_queries(queries, str(queries_path))

            progress_bar.progress(10, text="Initializing benchmark runner...")

            runner = BenchmarkRunner(
                results_dir=str(RESULTS_DIR),
                checkpoint_interval=5,
            )

            progress_bar.progress(20, text=f"Running {selected_exp}...")

            results = runner.run_experiment(
                exp_config,
                queries,
                resume=resume,
                max_queries=max_queries,
            )

            progress_bar.progress(100, text="Complete!")

            # Show summary
            st.success(f"Experiment {selected_exp} completed!")

            for config_name, query_results in results.items():
                valid = [r for r in query_results if not r.error]
                errors = len(query_results) - len(valid)
                st.write(f"**{config_name}:** {len(valid)} queries OK, {errors} errors")

            st.info("View detailed results in the Metrics Dashboard.")

        except Exception as e:
            st.error(f"Experiment failed: {e}")
        finally:
            st.session_state.experiment_running = False
