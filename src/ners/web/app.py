"""Entrypoint for the local experiment app."""

from __future__ import annotations

import streamlit as st

from ners.config import ResearchConfig
from ners.web.components import (
    render_dataset,
    render_experiment_launcher,
    render_overview,
    render_results,
)


def render_app() -> None:
    """Render the experiment, result, and dataset tabs."""

    st.set_page_config(
        page_title="CongoNames model study",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    config = ResearchConfig()

    st.title("CongoNames model study")
    st.caption("Run model templates and compare held-out metrics from names.csv.")
    st.info(
        "In this project, sex means the f or m marker in the source records, not gender "
        "identity."
    )

    with st.sidebar:
        st.header("Project paths")
        st.write("Dataset", config.dataset_path)
        st.write("Templates", config.templates_path)
        st.write("Results", config.experiments_dir)
        st.write("Models", config.experiment_models_dir)
        st.divider()
        st.caption(
            "Paths and defaults come from ners.config. This interface never writes a "
            "derived dataset."
        )

    overview_tab, run_tab, results_tab, dataset_tab = st.tabs(
        ["Experiments", "Run experiment", "Results", "Dataset"],
        default="Experiments",
        key="research_tabs",
        on_change="rerun",
    )
    if overview_tab.open:
        with overview_tab:
            render_overview(config)
    elif run_tab.open:
        with run_tab:
            render_experiment_launcher(config)
    elif results_tab.open:
        with results_tab:
            render_results(config)
    elif dataset_tab.open:
        with dataset_tab:
            render_dataset(config)


if __name__ == "__main__":  # pragma: no cover
    render_app()
