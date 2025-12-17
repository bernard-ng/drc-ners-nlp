import logging
from enum import Enum
from typing import Dict, Any, cast

import pandas as pd


class NameCategory(Enum):
    SIMPLE = "SIMPLE"
    COMPOSE = "COMPOSE"
    UNKNOWN = "UNKNOWN"


class FeatureExtractionStep:
    """
    Extracts features and performs initial tagging on the raw name data.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def _compute_numeric_features(self, series: pd.Series) -> Dict[str, Any]:
        """Compute basic numeric features from the name string."""
        name = str(series.get("full_name", ""))
        words = name.split()
        return {"name_length": len(name), "word_count": len(words)}

    def _assign_identified_category(self, series: pd.Series) -> str:
        """Categorize names based on word count."""
        val = series.get("word_count", 0)
        try:
            count = int(cast(Any, val))
        except (ValueError, TypeError):
            count = 0

        if count == 0:
            return str(NameCategory.UNKNOWN.value)
        return (
            str(NameCategory.SIMPLE.value)
            if count <= 2
            else str(NameCategory.COMPOSE.value)
        )

    def _normalize_gender(self, series: pd.Series) -> str:
        """Normalize gender strings to a standard format."""
        gender = str(series.get("gender", "")).upper()
        if gender in ["M", "MALE", "H"]:
            return "M"
        if gender in ["F", "FEMALE"]:
            return "F"
        return "U"

    def tag_name(self, name: str, probable_native: str, probable_surname: str) -> str:
        """Logic for tagging components (placeholder for actual implementation)."""
        if not name:
            return "O"
        return "B-PER"

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the feature extraction pipeline on the dataframe."""
        self.logger.info("Starting feature extraction...")

        working_df = df.copy()

        # Numeric features
        features = working_df.apply(
            lambda x: pd.Series(self._compute_numeric_features(cast(pd.Series, x))),
            axis=1,
        )
        working_df = pd.concat([working_df, features], axis=1)

        # Category assignment
        working_df["identified_category"] = working_df.apply(
            lambda x: self._assign_identified_category(cast(pd.Series, x)), axis=1
        )

        # Fixed numeric conversion using explicit cast to pd.Series for Pyright
        if "word_count" in working_df.columns:
            converted = pd.to_numeric(working_df["word_count"], errors="coerce")
            working_df["word_count"] = cast(pd.Series, converted).fillna(0).astype(int)

        # Gender normalization
        working_df["gender_norm"] = working_df.apply(
            lambda x: self._normalize_gender(cast(pd.Series, x)), axis=1
        )

        # Name tagging
        working_df["ner_tag"] = working_df.apply(
            lambda x: self.tag_name(
                str(cast(pd.Series, x).get("full_name", "")),
                str(cast(pd.Series, x).get("native_part", "")),
                str(cast(pd.Series, x).get("surname_part", "")),
            ),
            axis=1,
        )

        return working_df
