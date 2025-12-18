import pandas as pd
import streamlit as st
from typing import Any

from ners.core.utils.data_loader import OPTIMIZED_DTYPES


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk using optimized dtypes, with a safe fallback.

    This function attempts to read a CSV using `OPTIMIZED_DTYPES` to reduce memory
    usage. If reading fails for any reason, it reports the error to Streamlit and
    returns an empty DataFrame so the dashboard can continue to run without crashing.

    Args:
        file_path: Path to the CSV file to load.

    Returns:
        A pandas DataFrame with the loaded data, or an empty DataFrame on error.
    """
    try:
        return pd.read_csv(
            file_path,
            dtype=OPTIMIZED_DTYPES,  # type: ignore[arg-type]
        )
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


class Dashboard:
    """
    Dashboard renderer for the Streamlit web UI.

    Responsible for presenting dataset statistics and recent experiment summaries.
    """

    def __init__(self, config: Any, experiment_tracker: Any, experiment_runner: Any):
        """
        Initialize the Dashboard with configuration and experiment components.

        Args:
            config: Application configuration object providing paths and data info.
            experiment_tracker: Object that exposes `list_experiments()` for recent runs.
            experiment_runner: Runner object (kept for future use in the UI).
        """
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.experiment_runner = experiment_runner

    def index(self) -> None:
        """
        Render the main dashboard view.

        Displays total number of rows (names), annotated count, provinces, gender ratio,
        annotation rate, and a table of recent experiments. All numeric conversions are
        guarded to avoid passing `None`/`Unknown` to `int()` which fixes the static
        typing errors raised by Pyright.
        """
        st.title("Dashboard")
        col1, col2, col3, col4, col5 = st.columns(5)

        try:
            data_path = self.config.paths.get_data_path(
                self.config.data.output_files["featured"]
            )

            if not data_path.exists():
                st.warning("No processed data found. Please run data processing first.")
                return

            df = load_dataset(str(data_path))
            total_rows = int(len(df) or 0)

            with col1:
                st.metric("Total Names", f"{total_rows:,}")

            with col2:
                # guard .sum() result to avoid passing None/Unknown to int()
                annotated_sum = (df.get("annotated", 0) == 1).sum() if df is not None else 0
                annotated_count = int(annotated_sum or 0)
                st.metric("Annotated Names", f"{annotated_count:,}")

            with col3:
                provinces_unique = df["province"].nunique() if "province" in df.columns else 0
                provinces = int(provinces_unique or 0)
                st.metric("Provinces", provinces)

            with col4:
                if "sex" in df.columns:
                    # value_counts() may return numpy scalars or missing keys -> guard with `or 0`
                    gender_dist = df["sex"].value_counts()
                    females = int(gender_dist.get("f") or 0)
                    males = int(gender_dist.get("m") or 0)
                    # avoid division by zero; use max(males, 1) to produce a sensible ratio
                    ratio = females / max(males, 1)
                    st.metric("F/M Rate", f"{ratio:.2%}")

            with col5:
                if total_rows > 0 and "annotated" in df.columns:
                    ratio = (annotated_count / total_rows) if total_rows else 0.0
                    st.metric("Annotation Rate", f"{ratio:.2%}")

        except Exception as e:
            st.error(f"Error loading dashboard data: {e}")

        # Recent experiments section
        st.subheader("Recent Experiments")
        experiments = self.experiment_tracker.list_experiments()[:5]

        if not experiments:
            st.info(
                "No experiments found. Create your first experiment in the Experiments tab!"
            )
            return

        exp_data: list[dict[str, Any]] = []
        for exp in experiments:
            exp_data.append(
                {
                    "Name": exp.config.name,
                    "Model": exp.config.model_type,
                    "Status": exp.status.value,
                    "Accuracy": (
                        f"{exp.test_metrics.get('accuracy', 0):.3f}"
                        if exp.test_metrics
                        else "N/A"
                    ),
                    "Date": exp.start_time.strftime("%Y-%m-%d %H:%M"),
                }
            )

        st.dataframe(pd.DataFrame(exp_data), use_container_width=True)
