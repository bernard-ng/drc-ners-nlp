import pandas as pd
import streamlit as st

from ners.core.utils.data_loader import OPTIMIZED_DTYPES


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, dtype=OPTIMIZED_DTYPES)  # type: ignore
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


class Dashboard:
    def __init__(self, config, experiment_tracker, experiment_runner):
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.experiment_runner = experiment_runner

    def index(self):
        st.title("Dashboard")
        col1, col2, col3, col4, col5 = st.columns(5)

        # Load basic statistics
        try:
            data_path = self.config.paths.get_data_path(
                self.config.data.output_files["featured"]
            )
            if data_path.exists():
                df = load_dataset(str(data_path))

                with col1:
                    st.metric("Total Names", str(len(df)))

                with col2:
                    annotated = len(df[df["annotated"] == 1]) if "annotated" in df.columns else 0
                    st.metric("Annotated Names", str(annotated))

                with col3:
                    provinces = (
                        df["province"].nunique() if "province" in df.columns else 0
                    )
                    st.metric("Provinces", str(provinces))

                with col4:
                    if "sex" in df.columns:
                        gender_dist = df["sex"].value_counts()
                        f_count = int(gender_dist.get("f", 0) or 0)
                        m_count = int(gender_dist.get("m", 0) or 0)

                        if m_count == 0:
                            if f_count == 0:
                                ratio_display = "N/A (no data)"
                            else:
                                ratio_display = "∞ (no males)"
                        else:
                            ratio = f_count / m_count
                            ratio_display = f"{ratio:.2f}"

                        st.metric("F/M Rate", ratio_display)
                with col5:
                    if "annotated" in df.columns:
                        annotated = len(df[df["annotated"] == 1])
                        ratio = float(annotated) / len(df) if len(df) > 0 else 0.0
                        st.metric("Annotation Rate", f"{ratio:.2%}")
            else:
                st.warning("No processed data found. Please run data processing first.")

        except Exception as e:
            st.error(f"Error loading dashboard data: {e}")

        # Recent experiments
        st.subheader("Recent Experiments")
        experiments = self.experiment_tracker.list_experiments()[:5]

        if experiments:
            exp_data = []
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
        else:
            st.info(
                "No experiments found. Create your first experiment in the Experiments tab!"
            )
