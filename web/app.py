#!.venv/bin/python3
import argparse
import sys
from pathlib import Path

import streamlit as st

# Add parent directory to Python path to access core modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core.config import setup_config, PipelineConfig
from core.utils.data_loader import DataLoader
from processing.monitoring.pipeline_monitor import PipelineMonitor
from research.experiment.experiment_runner import ExperimentRunner
from research.experiment.experiment_tracker import ExperimentTracker

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


def main():
    parser = argparse.ArgumentParser(
        description="DRC NERS Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="development", help="Environment name")
    args = parser.parse_args()

    config = setup_config(args.config, env=args.env)
    app = StreamlitApp(config)
    app.run()


if __name__ == "__main__":
    main()
