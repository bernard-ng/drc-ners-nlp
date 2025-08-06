import logging
from typing import Dict

import pandas as pd


class DatasetAnalyzer:
    """Analyze dataset statistics and quality"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None

    def load_data(self) -> bool:
        """Load dataset for analysis"""
        try:
            self.df = pd.read_csv(self.filepath)
            return True
        except Exception as e:
            logging.error(f"Failed to load {self.filepath}: {e}")
            return False

    def analyze_completion(self) -> Dict:
        """Analyze annotation completion status"""
        if self.df is None:
            return {}

        total_rows = len(self.df)

        # Check annotation status
        if "annotated" in self.df.columns:
            annotated_count = (self.df["annotated"] == 1).sum()
            unannotated_count = (self.df["annotated"] == 0).sum()
        else:
            annotated_count = 0
            unannotated_count = total_rows

        # Analyze name completeness
        complete_names = 0
        if "identified_name" in self.df.columns and "identified_surname" in self.df.columns:
            complete_names = (
                (self.df["identified_name"].notna()) & (self.df["identified_surname"].notna())
            ).sum()

        return {
            "total_rows": total_rows,
            "annotated_rows": annotated_count,
            "unannotated_rows": unannotated_count,
            "annotation_percentage": (annotated_count / total_rows * 100) if total_rows > 0 else 0,
            "complete_names": complete_names,
            "completeness_percentage": (complete_names / total_rows * 100) if total_rows > 0 else 0,
        }
