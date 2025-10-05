import gc
import logging
from enum import Enum
from typing import Dict, Any

import pandas as pd

from ners.core.config.pipeline_config import PipelineConfig
from ners.core.utils.region_mapper import RegionMapper
from ners.processing.ner.name_tagger import NameTagger
from ners.processing.steps import PipelineStep


class Gender(Enum):
    MALE = "m"
    FEMALE = "f"


class NameCategory(Enum):
    SIMPLE = "simple"
    COMPOSE = "compose"


class FeatureExtractionStep(PipelineStep):
    """Configuration-driven feature extraction step"""

    def __init__(self, pipeline_config: PipelineConfig):
        super().__init__("feature_extraction", pipeline_config)
        self.region_mapper = RegionMapper()
        self.name_tagger = NameTagger()

    @classmethod
    def requires_batch_mutation(cls) -> bool:
        """This step creates new columns, so mutation is required"""
        return True

    @classmethod
    def validate_gender(cls, gender: str) -> Gender:
        """Validate and normalize gender value"""
        gender_lower = str(gender).lower().strip()
        if gender_lower in ["m", "male", "homme", "masculin"]:
            return Gender.MALE
        elif gender_lower in ["f", "female", "femme", "féminin"]:
            return Gender.FEMALE
        else:
            raise ValueError(f"Unknown gender: {gender}")

    @classmethod
    def get_name_category(cls, word_count: int) -> NameCategory:
        """Determine name category based on word count"""
        return NameCategory.SIMPLE if word_count == 3 else NameCategory.COMPOSE

    def process_batch(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Extract features from names in batch"""
        logging.info(f"Extracting features for batch {batch_id} with {len(batch)} rows")

        result = batch.copy()
        numeric_features = self._compute_numeric_features(result["name"])
        result = result.assign(**numeric_features)

        # Initialize features columns with optimal dtypes
        features_columns = self._initialize_features_columns(len(result))
        result = result.assign(**features_columns)

        self._assign_probable_names(result)
        self._process_simple_names(result)
        result["identified_category"] = self._assign_identified_category(
            result["words"]
        )

        if "year" in result.columns:
            result["year"] = pd.to_numeric(result["year"], errors="coerce").astype(
                "Int16"
            )

        if "region" in result.columns:
            result["province"] = self.region_mapper.map(result["region"]).str.lower()
            result["province"] = result["province"].astype("category")

        if "sex" in result.columns:
            result["sex"] = self._normalize_gender(result["sex"])

        # Apply final dtype optimizations
        result = self._optimize_dtypes(result)

        # Cleanup
        del numeric_features, features_columns
        if batch_id % 10 == 0:  # Periodic cleanup
            gc.collect()

        return result

    @classmethod
    def _compute_numeric_features(cls, series: pd.Series) -> Dict[str, pd.Series]:
        """Calculate basic features in vectorized manner"""
        return {
            "words": (series.str.count(" ") + 1).astype("Int8"),
            "length": series.str.len().astype("Int16"),
        }

    @classmethod
    def _initialize_features_columns(cls, size: int) -> Dict[str, Any]:
        """Initialize new columns with optimal dtypes"""
        return {
            "probable_native": pd.Series([None] * size, dtype="string"),
            "probable_surname": pd.Series([None] * size, dtype="string"),
            "identified_name": pd.Series([None] * size, dtype="string"),
            "identified_surname": pd.Series([None] * size, dtype="string"),
            "ner_entities": pd.Series([None] * size, dtype="string"),
            "ner_tagged": pd.Series([0] * size, dtype="Int8"),
            "annotated": pd.Series([0] * size, dtype="Int8"),
        }

    @classmethod
    def _assign_probable_names(cls, df: pd.DataFrame) -> None:
        """Assign probable native and surname names efficiently"""

        name_splits = df["name"].str.split()
        mask = name_splits.str.len() >= 2

        df.loc[mask, "probable_native"] = name_splits[mask].apply(
            lambda x: " ".join(x[:-1]) if isinstance(x, list) else None
        )
        df.loc[mask, "probable_surname"] = name_splits[mask].apply(
            lambda x: x[-1] if isinstance(x, list) else None
        )

    def _assign_identified_category(self, series: pd.Series) -> pd.Series:
        """Assign identified category based on word count"""
        return series.map(lambda x: self.get_name_category(x).value).astype("category")

    def _process_simple_names(self, df: pd.DataFrame) -> None:
        """Process 3-word names efficiently with vectorized operations"""
        mask = pd.Series(df["words"] == 3)

        if not mask.any():
            return

        df.loc[mask, "identified_name"] = df.loc[mask, "probable_native"]
        df.loc[mask, "identified_surname"] = df.loc[mask, "probable_surname"]
        df.loc[mask, "annotated"] = 1

        # NER tagging for 3-word names
        three_word_rows = df[mask]
        for idx, row in three_word_rows.iterrows():
            try:
                entity = self.name_tagger.tag_name(
                    row["name"], row["identified_name"], row["identified_surname"]
                )

                if entity:
                    df.at[idx, "ner_entities"] = str(entity["entities"])
                    df.at[idx, "ner_tagged"] = 1
            except Exception as e:
                logging.warning(f"NER tagging failed for row {idx}: {e}")

    @classmethod
    def _normalize_gender(cls, series: pd.Series) -> pd.Series:
        gender_mapping = {
            "m": "m",
            "male": "m",
            "homme": "m",
            "masculin": "m",
            "f": "f",
            "female": "f",
            "femme": "f",
            "féminin": "f",
        }

        # Apply mapping with error handling
        normalized = series.astype(str).str.lower().str.strip().map(gender_mapping)
        return normalized.astype("category")

    @classmethod
    def _optimize_dtypes(cls, df: pd.DataFrame) -> pd.DataFrame:
        categories = ["province", "identified_category", "sex"]

        for col in categories:
            if col in df.columns and df[col].dtype != "category":
                df[col] = df[col].astype("category")

        # Ensure string columns are proper string dtype
        string_cols = [
            "name",
            "probable_native",
            "probable_surname",
            "identified_name",
            "identified_surname",
            "ner_entities",
        ]

        for col in string_cols:
            if col in df.columns and df[col].dtype == "object":
                df[col] = df[col].astype("string")

        return df
