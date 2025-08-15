#!.venv/bin/python3
import argparse

import streamlit as st

from core.config import setup_config, PipelineConfig
from core.utils.data_loader import DataLoader
from interface.configuration import Configuration
from interface.dashboard import Dashboard
from interface.data_overview import DataOverview
from interface.data_processing import DataProcessing
from interface.experiments import Experiments
from interface.predictions import Predictions
from interface.results_analysis import ResultsAnalysis
from processing.monitoring.pipeline_monitor import PipelineMonitor
from research.experiment.experiment_runner import ExperimentRunner
from research.experiment.experiment_tracker import ExperimentTracker

# Page configuration
st.set_page_config(
    page_title="DRC Names NLP Pipeline",
    page_icon="🇨🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)


class StreamlitApp:
    """Main Streamlit application class"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader(self.config)
        self.experiment_tracker = ExperimentTracker(self.config)
        self.experiment_runner = ExperimentRunner(self.config)
        self.pipeline_monitor = PipelineMonitor()

        # Initialize interface components
        self.dashboard = Dashboard(self.config, self.experiment_tracker, self.experiment_runner)
        self.data_overview = DataOverview(self.config)
        self.data_processing = DataProcessing(self.config, self.pipeline_monitor)
        self.experiments = Experiments(self.config, self.experiment_tracker, self.experiment_runner)
        self.results_analysis = ResultsAnalysis(
            self.config, self.experiment_tracker, self.experiment_runner
        )
        self.predictions = Predictions(self.config, self.experiment_tracker, self.experiment_runner)
        self.configuration = Configuration(self.config)

        # Initialize session state
        if "current_experiment" not in st.session_state:
            st.session_state.current_experiment = None
        if "experiment_results" not in st.session_state:
            st.session_state.experiment_results = {}

    def run(self):
        st.title("🇨🇩 DRC NERS Pipeline")
        st.markdown("A comprehensive tool for Congolese name analysis and gender prediction")

        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            [
                "Dashboard",
                "Dataset Overview",
                "Data Processing",
                "Experiments",
                "Results & Analysis",
                "Predictions",
                "Configuration",
            ],
        )

        # Route to appropriate page
        page_map = {
            "Dashboard": self.dashboard.index,
            "Dataset Overview": self.data_overview.index,
            "Data Processing": self.data_processing.index,
            "Experiments": self.experiments.index,
            "Results & Analysis": self.results_analysis.index,
            "Predictions": self.predictions.index,
            "Configuration": self.configuration.index,
        }
        page_map.get(page, lambda: None)()


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
