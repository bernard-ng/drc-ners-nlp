from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from core.utils.data_loader import OPTIMIZED_DTYPES


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, dtype=OPTIMIZED_DTYPES)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


class DataOverview:
    def __init__(self, config):
        self.config = config

    def index(self):
        st.title("Data Overview")
        data_files = {
            "Names": self.config.data.input_file,
            "Featured Dataset": self.config.data.output_files["featured"],
            "Evaluation Dataset": self.config.data.output_files["evaluation"],
            "Male Names": self.config.data.output_files["males"],
            "Female Names": self.config.data.output_files["females"],
        }

        selected_file = st.selectbox("Select Dataset", list(data_files.keys()))
        file_path = self.config.paths.get_data_path(data_files[selected_file])

        if not file_path.exists():
            st.warning(f"Dataset not found: {file_path}")
            st.warning("Please run data processing first to generate datasets.")
            return

        # Load and display data
        df = load_dataset(str(file_path))

        if df.empty:
            st.error("Failed to load dataset")
            return

        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(df):,}")

        with col2:
            if "annotated" in df.columns:
                annotated_pct = (df["annotated"] == 1).mean() * 100
                st.metric("Annotated", f"{annotated_pct:.1f}%")

        with col3:
            if "words" in df.columns:
                avg_words = df["words"].mean()
                st.metric("Avg Words", f"{avg_words:.1f}")

        with col4:
            if "length" in df.columns:
                avg_length = df["length"].mean()
                st.metric("Avg Length", f"{avg_length:.0f}")

        # Data quality analysis
        st.subheader("Data Quality Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(
                    x=missing_data.index, y=missing_data.values, title="Missing Values by Column"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found")

        with col2:
            # Gender distribution
            if "sex" in df.columns:
                gender_counts = df["sex"].value_counts()
                fig = px.pie(
                    values=gender_counts.values,
                    names=gender_counts.index,
                    title="Gender Distribution",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Word count distribution
        if "words" in df.columns:
            st.subheader("Name Structure Analysis")

            col1, col2 = st.columns(2)

            with col1:
                word_dist = df["words"].value_counts().sort_index()
                fig = px.bar(
                    x=word_dist.index,
                    y=word_dist.values,
                    title="Distribution of Word Count in Names",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Province distribution
                if "province" in df.columns:
                    province_counts = df["province"].value_counts().head(10)
                    fig = px.bar(
                        x=province_counts.values,
                        y=province_counts.index,
                        orientation="h",
                        title="Top 10 Provinces by Name Count",
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

        # Sample data
        st.subheader("Sample Data")

        # Display columns selector
        if not df.empty:
            columns_to_show = st.multiselect(
                "Select columns to display",
                df.columns.tolist(),
                default=(
                    ["name", "sex", "province", "words"]
                    if all(col in df.columns for col in ["name", "sex", "province", "words"])
                    else df.columns[:5].tolist()
                ),
            )

            if columns_to_show:
                sample_size = st.slider("Number of rows to display", 10, min(1000, len(df)), 50)
                st.dataframe(df[columns_to_show].head(sample_size), use_container_width=True)

        # Data export
        st.subheader("Export Data")
        if st.button("Download as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{selected_file.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
