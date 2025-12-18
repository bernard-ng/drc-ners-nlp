from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from ners.core.utils.data_loader import OPTIMIZED_DTYPES


@st.cache_data
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from disk using optimized dtypes, with a safe fallback.

    The `OPTIMIZED_DTYPES` mapping is intentionally passed with a type ignore
    to satisfy static type checkers (Pyright), as pandas supports dict-based
    dtype specifications at runtime.

    Args:
        file_path: Path to the CSV file.

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


class DataOverview:
    """
    Streamlit view that provides an overview of available datasets.

    Displays file existence, size, last modification date, and a preview
    of the featured dataset when available.
    """

    def __init__(self, config: Any):
        """
        Initialize the DataOverview with application configuration.

        Args:
            config: Application configuration object providing data paths.
        """
        self.config = config

    def index(self) -> None:
        """
        Render the Data Overview page.

        Lists all known data files with basic filesystem statistics and
        shows a preview of the featured dataset if it exists.
        """
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

            if exists:
                stat = file_path.stat()
                size_mb = stat.st_size / (1024 * 1024)
                last_modified = datetime.fromtimestamp(stat.st_mtime)
                stats = f"Size: {size_mb:.1f} MB, Last Modified: {last_modified}"
            else:
                stats = "Not found"

            st.write(f"- {name}: {file_path} ({stats})")

        # Preview featured dataset if available
        data_path = self.config.paths.get_data_path(
            self.config.data.output_files["featured"]
        )

        if data_path.exists():
            df = load_dataset(str(data_path))
            st.subheader("Featured Dataset Preview")
            st.dataframe(df.head(), use_container_width=True)
            st.write(f"Rows: {int(len(df) or 0):,}")
