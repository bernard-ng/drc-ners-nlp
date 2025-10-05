#!.venv/bin/python3
import os

import streamlit as st

from ners.core.config import setup_config, PipelineConfig
from ners.core.utils.data_loader import DataLoader
from ners.processing.monitoring.pipeline_monitor import PipelineMonitor
from ners.research.experiment.experiment_runner import ExperimentRunner
from ners.research.experiment.experiment_tracker import ExperimentTracker

# Page configuration
st.set_page_config(
    page_title="DRC NERS Platform",
    page_icon="🇨🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state(config: PipelineConfig):
    """Initialize session state variables"""
    if "config" not in st.session_state:
        st.session_state.config = config
    if "data_loader" not in st.session_state:
        st.session_state.data_loader = DataLoader(config)
    if "experiment_tracker" not in st.session_state:
        st.session_state.experiment_tracker = ExperimentTracker(config)
    if "experiment_runner" not in st.session_state:
        st.session_state.experiment_runner = ExperimentRunner(config)
    if "pipeline_monitor" not in st.session_state:
        st.session_state.pipeline_monitor = PipelineMonitor()
    if "current_experiment" not in st.session_state:
        st.session_state.current_experiment = None
    if "experiment_results" not in st.session_state:
        st.session_state.experiment_results = {}


class StreamlitApp:
    def __init__(self, config: PipelineConfig):
        self.config = config
        initialize_session_state(config)

    @classmethod
    def run(cls):
        st.title("🇨🇩 DRC NERS Platform")
        st.markdown(
            "A Culturally-Aware NLP System for Congolese Name Analysis and Gender Inference"
        )
        st.markdown(
            """
            ## Overview
            Despite the growing success of gender inference models in Natural Language Processing (NLP), these tools often
            underperform when applied to culturally diverse African contexts due to the lack of culturally-representative training
            data.
            This project introduces a comprehensive pipeline for Congolese name analysis with a large-scale dataset of over 5
            million names from the Democratic Republic of Congo (DRC) annotated with gender and demographic metadata.
            """
        )


# Initialize app using environment variables when launched via Typer
_config_path = os.environ.get("NERS_CONFIG")
_env = os.environ.get("NERS_ENV", "development")
_cfg = setup_config(_config_path, env=_env)
_app = StreamlitApp(_cfg)
_app.run()
