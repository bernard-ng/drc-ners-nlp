from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from ners.core.utils.data_loader import OPTIMIZED_DTYPES
from ners.web.interfaces.log_reader import LogReader


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk using optimized dtypes, with a safe fallback.

    The OPTIMIZED_DTYPES mapping is supported by pandas at runtime but is not
    fully recognized by static type checkers (Pyright). A targeted type ignore
    is used to avoid false-positive errors.

    Args:
        file_path: Path to the CSV file to load.

    Returns:
        A pandas DataFrame containing the dataset, or an empty DataFrame on error.
    """
    try:
        return pd.read_csv(
            file_path,
            dtype=OPTIMIZED_DTYPES,  # type: ignore[arg-type]
        )
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


class DataProcessing:
    """
    Streamlit interface for monitoring the data processing pipeline.

    Displays overall progress, step-level metrics, and recent log entries
    with filtering and basic statistics.
    """

    def __init__(self, config: Any, pipeline_monitor: Any):
        """
        Initialize the DataProcessing view.

        Args:
            config: Application configuration object providing paths.
            pipeline_monitor: Object exposing pipeline status information.
        """
        self.config = config
        self.pipeline_monitor = pipeline_monitor

    def index(self) -> None:
        """
        Render the Data Processing dashboard.

        Shows pipeline progress, per-step statistics, recent processing logs,
        and aggregated log-level metrics with a visualization.
        """
        st.title("Data Processing")
        status = self.pipeline_monitor.get_pipeline_status()

        # Overall progress
        overall_progress = (status.get("overall_completion", 0) or 0) / 100
        st.progress(overall_progress)
        st.write(f"Overall Progress: {status.get('overall_completion', 0):.1f}%")

        # Step details
        for step_name, step_status in status.get("steps", {}).items():
            with st.expander(
                f"{step_name.replace('_', ' ').title()} - {step_status.get('status', 'unknown')}"
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Processed Batches",
                        int(step_status.get("processed_batches", 0) or 0),
                    )

                with col2:
                    st.metric(
                        "Total Batches",
                        int(step_status.get("total_batches", 0) or 0),
                    )

                with col3:
                    st.metric(
                        "Failed Batches",
                        int(step_status.get("failed_batches", 0) or 0),
                    )

                completion_pct = step_status.get("completion_percentage", 0) or 0
                if completion_pct > 0:
                    st.progress(completion_pct / 100)

        # Read actual log entries from the log file
        st.subheader("Recent Processing Logs")
        try:
            log_file_path = self.config.paths.logs_dir / "pipeline.development.log"
            log_reader = LogReader(log_file_path)

            # Options for filtering logs
            col1, col2 = st.columns(2)
            with col1:
                log_level_filter = st.selectbox(
                    "Filter by Level",
                    ["All", "INFO", "WARNING", "ERROR", "DEBUG", "CRITICAL"],
                    key="log_level_filter",
                )

            with col2:
                num_entries = st.number_input(
                    "Number of entries",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key="num_log_entries",
                )

            # Get log entries based on filter
            if log_level_filter == "All":
                log_entries = log_reader.read_last_entries(num_entries)
            else:
                log_entries = log_reader.read_entries_by_level(
                    log_level_filter, num_entries
                )

            if log_entries:
                for entry in log_entries:
                    timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    message = f"[{timestamp}] {entry.level}: {entry.message}"

                    if entry.level == "ERROR":
                        st.error(message)
                    elif entry.level == "WARNING":
                        st.warning(message)
                    elif entry.level == "INFO":
                        st.info(message)
                    else:
                        st.text(message)

                # Show log statistics
                st.subheader("Log Statistics")
                log_stats = log_reader.get_log_stats()

                if log_stats:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Lines", int(log_stats.get("total_lines", 0) or 0))
                    with col2:
                        st.metric("INFO", int(log_stats.get("INFO", 0) or 0))
                    with col3:
                        st.metric("WARNING", int(log_stats.get("WARNING", 0) or 0))
                    with col4:
                        st.metric("ERROR", int(log_stats.get("ERROR", 0) or 0))

                    # Log level distribution chart
                    levels = ["INFO", "WARNING", "ERROR", "DEBUG", "CRITICAL"]
                    counts = [int(log_stats.get(level, 0) or 0) for level in levels]

                    if sum(counts) > 0:
                        fig = px.bar(
                            x=levels,
                            y=counts,
                            title="Log Entries by Level",
                            color=levels,
                            color_discrete_map={
                                "INFO": "blue",
                                "WARNING": "orange",
                                "ERROR": "red",
                                "DEBUG": "gray",
                                "CRITICAL": "darkred",
                            },
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No log entries found or log file is empty.")

        except Exception as e:
            st.error(f"Error reading log file: {e}")
