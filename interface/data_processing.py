import pandas as pd
import plotly.express as px
import streamlit as st

from interface.log_reader import LogReader


def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


class DataProcessing:
    def __init__(self, config, pipeline_monitor):
        self.config = config
        self.pipeline_monitor = pipeline_monitor

    def index(self):
        st.header("Data Processing Pipeline")
        status = self.pipeline_monitor.get_pipeline_status()

        # Overall progress
        overall_progress = status["overall_completion"] / 100
        st.progress(overall_progress)
        st.write(f"Overall Progress: {status['overall_completion']:.1f}%")

        # Step details
        for step_name, step_status in status["steps"].items():
            with st.expander(f"{step_name.replace('_', ' ').title()} - {step_status['status']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Processed Batches", step_status["processed_batches"])

                with col2:
                    st.metric("Total Batches", step_status["total_batches"])

                with col3:
                    st.metric("Failed Batches", step_status["failed_batches"])

                if step_status["completion_percentage"] > 0:
                    st.progress(step_status["completion_percentage"] / 100)

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
                    key="log_level_filter"
                )

            with col2:
                num_entries = st.number_input(
                    "Number of entries",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key="num_log_entries"
                )

            # Get log entries based on filter
            if log_level_filter == "All":
                log_entries = log_reader.read_last_entries(num_entries)
            else:
                log_entries = log_reader.read_entries_by_level(log_level_filter, num_entries)

            if log_entries:
                for entry in log_entries:
                    if entry.level == "ERROR":
                        st.error(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {entry.level}: {entry.message}")
                    elif entry.level == "WARNING":
                        st.warning(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {entry.level}: {entry.message}")
                    elif entry.level == "INFO":
                        st.info(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {entry.level}: {entry.message}")
                    else:
                        st.text(f"[{entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {entry.level}: {entry.message}")

                # Show log statistics
                st.subheader("Log Statistics")
                log_stats = log_reader.get_log_stats()

                if log_stats:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Lines", log_stats.get('total_lines', 0))
                    with col2:
                        st.metric("INFO", log_stats.get('INFO', 0))
                    with col3:
                        st.metric("WARNING", log_stats.get('WARNING', 0))
                    with col4:
                        st.metric("ERROR", log_stats.get('ERROR', 0))

                    # Log level distribution chart
                    levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG', 'CRITICAL']
                    counts = [log_stats.get(level, 0) for level in levels]

                    if sum(counts) > 0:
                        fig = px.bar(
                            x=levels,
                            y=counts,
                            title="Log Entries by Level",
                            color=levels,
                            color_discrete_map={
                                'INFO': 'blue',
                                'WARNING': 'orange',
                                'ERROR': 'red',
                                'DEBUG': 'gray',
                                'CRITICAL': 'darkred'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No log entries found or log file is empty.")

        except Exception as e:
            st.error(f"Error reading log file: {e}")
