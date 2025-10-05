from datetime import datetime

import pandas as pd
import streamlit as st

from ners.core.utils.data_loader import OPTIMIZED_DTYPES


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

        st.write("Available Data Files:")
        for name, rel_path in data_files.items():
            file_path = self.config.paths.get_data_path(rel_path)
            exists = file_path.exists()
            size = file_path.stat().st_size if exists else 0
            stats = (
                f"Size: {size / (1024 * 1024):.1f} MB, Last Modified: {datetime.fromtimestamp(file_path.stat().st_mtime)}"
                if exists
                else "Not found"
            )
            st.write(f"- {name}: {file_path} ({stats})")

        # Preview featured dataset if available
        data_path = self.config.paths.get_data_path(
            self.config.data.output_files["featured"]
        )
        if data_path.exists():
            df = load_dataset(str(data_path))
            st.subheader("Featured Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Rows: {len(df):,}")
