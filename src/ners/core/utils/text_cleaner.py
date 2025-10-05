from typing import Optional, Dict

import pandas as pd


class TextCleaner:
    """Reusable text cleaning utilities"""

    def __init__(self, patterns: Optional[Dict[str, str]] = None):
        self.patterns = patterns or {
            "null_bytes": "\x00",
            "non_breaking_spaces": "\u00a0",
            "multiple_spaces": r" +",
            "extra_whitespace": r"\s+",
        }

    def clean_text_series(self, series: pd.Series) -> pd.Series:
        """Clean a pandas Series of text data"""
        cleaned = series.astype(str)

        # Apply cleaning patterns
        for pattern_name, pattern in self.patterns.items():
            if pattern_name == "multiple_spaces":
                cleaned = cleaned.str.replace(pattern, " ", regex=True)
            else:
                cleaned = cleaned.str.replace(pattern, " ", regex=False)

        return cleaned.str.strip().str.lower()

    def clean_dataframe_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean all text columns in a DataFrame"""
        df = df.copy()
        columns = df.select_dtypes(include=["object", "string"]).columns
        for col in columns:
            df[col] = self.clean_text_series(df[col])

        return df
