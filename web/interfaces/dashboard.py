import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.utils.data_loader import OPTIMIZED_DTYPES


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, dtype=OPTIMIZED_DTYPES)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


class Dashboard:
    def __init__(self, config, experiment_tracker, experiment_runner):
        self.config = config
        self.experiment_tracker = experiment_tracker
        self.experiment_runner = experiment_runner

    def _create_gender_distribution_chart(self, df: pd.DataFrame):
        """Create gender distribution pie chart"""
        if "sex" in df.columns:
            gender_counts = df["sex"].value_counts()
            fig = px.pie(
                values=gender_counts.values,
                names=gender_counts.index,
                title="Gender Distribution",
                color_discrete_map={"m": "#3498db", "f": "#e74c3c"},
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            return fig
        return None

    def _create_province_distribution_chart(self, df: pd.DataFrame):
        """Create province distribution bar chart"""
        if "province" in df.columns:
            province_counts = df["province"].value_counts().head(15)  # Top 15 provinces
            fig = px.bar(
                x=province_counts.index,
                y=province_counts.values,
                title="Top 15 Provinces by Name Count",
                labels={"x": "Province", "y": "Number of Names"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        return None

    def _create_name_length_distribution(self, df: pd.DataFrame):
        """Create name length distribution histogram"""
        if "length" in df.columns:
            fig = px.histogram(
                df,
                x="length",
                title="Name Length Distribution",
                labels={"length": "Name Length (characters)", "count": "Frequency"},
                nbins=30,
            )
            fig.update_layout(bargap=0.1)
            return fig
        return None

    def _create_annotation_progress_chart(self, df: pd.DataFrame):
        """Create annotation progress chart"""
        if "annotated" in df.columns and "ner_tagged" in df.columns:
            annotation_data = {
                "Not Annotated": (df["annotated"] == 0).sum(),
                "Annotated": (df["annotated"] == 1).sum(),
                "NER Tagged": (df["ner_tagged"] == 1).sum(),
            }

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(annotation_data.keys()),
                        y=list(annotation_data.values()),
                        marker_color=["#95a5a6", "#2ecc71", "#9b59b6"],
                    )
                ]
            )
            fig.update_layout(
                title="Annotation Progress",
                xaxis_title="Status",
                yaxis_title="Number of Names",
            )
            return fig
        return None

    def _create_regional_analysis(self, df: pd.DataFrame):
        """Create regional analysis chart"""
        if "region" in df.columns and "sex" in df.columns:
            regional_gender = pd.crosstab(df["region"], df["sex"])
            fig = px.bar(
                regional_gender,
                title="Gender Distribution by Region",
                labels={"value": "Count", "index": "Region"},
            )
            fig.update_layout(xaxis_tickangle=-45)
            return fig
        return None

    def _create_words_distribution(self, df: pd.DataFrame):
        """Create word count distribution"""
        if "words" in df.columns:
            fig = px.box(
                df,
                y="words",
                title="Word Count Distribution in Names",
                labels={"words": "Number of Words"},
            )
            return fig
        return None

    def index(self):
        st.title("Dashboard")

        # Load basic statistics
        try:
            data_path = self.config.paths.get_data_path(self.config.data.output_files["featured"])
            if data_path.exists():
                df = load_dataset(str(data_path))

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Names", f"{len(df):,}")

                with col2:
                    annotated = (df.get("annotated", 0) == 1).sum()
                    st.metric("Annotated Names", f"{annotated:,}")

                with col3:
                    provinces = df["province"].nunique() if "province" in df.columns else 0
                    st.metric("Provinces", provinces)

                with col4:
                    if "sex" in df.columns:
                        gender_dist = df["sex"].value_counts()
                        ratio = gender_dist.get("f", 0) / max(gender_dist.get("m", 1), 1)
                        st.metric("F/M Ratio", f"{ratio:.2f}")

                # First row of charts
                col1, col2 = st.columns(2)

                with col1:
                    gender_chart = self._create_gender_distribution_chart(df)
                    if gender_chart:
                        st.plotly_chart(gender_chart, use_container_width=True)

                with col2:
                    annotation_chart = self._create_annotation_progress_chart(df)
                    if annotation_chart:
                        st.plotly_chart(annotation_chart, use_container_width=True)

                # Second row of charts
                col1, col2 = st.columns(2)

                with col1:
                    length_chart = self._create_name_length_distribution(df)
                    if length_chart:
                        st.plotly_chart(length_chart, use_container_width=True)

                with col2:
                    words_chart = self._create_words_distribution(df)
                    if words_chart:
                        st.plotly_chart(words_chart, use_container_width=True)

                # Full-width charts
                province_chart = self._create_province_distribution_chart(df)
                if province_chart:
                    st.plotly_chart(province_chart, use_container_width=True)

                regional_chart = self._create_regional_analysis(df)
                if regional_chart:
                    st.plotly_chart(regional_chart, use_container_width=True)

                # Data insights section
                st.header("🔍 Key Insights")
                insights_col1, insights_col2 = st.columns(2)

                with insights_col1:
                    st.subheader("Dataset Overview")
                    total_names = len(df)
                    unique_provinces = df["province"].nunique() if "province" in df.columns else 0
                    avg_length = df["length"].mean() if "length" in df.columns else 0

                    st.write(f"• **{total_names:,}** total names in the dataset")
                    st.write(f"• **{unique_provinces}** provinces represented")
                    if avg_length > 0:
                        st.write(f"• Average name length: **{avg_length:.1f}** characters")

                with insights_col2:
                    st.subheader("Processing Status")
                    if "annotated" in df.columns:
                        annotated_pct = (df["annotated"] == 1).mean() * 100
                        st.write(f"• **{annotated_pct:.1f}%** of names are annotated")

                    if "ner_tagged" in df.columns:
                        ner_pct = (df["ner_tagged"] == 1).mean() * 100
                        st.write(f"• **{ner_pct:.1f}%** of names have NER tags")

            else:
                st.warning("No processed data found. Please run data processing first.")

        except Exception as e:
            st.error(f"Error loading dashboard data: {e}")

        # Recent experiments
        st.header("Recent Experiments")
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
            st.info("No experiments found. Create your first experiment in the Experiments tab!")
