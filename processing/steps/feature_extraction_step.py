import logging
from enum import Enum

import pandas as pd

from core.config.pipeline_config import PipelineConfig
from core.utils.region_mapper import RegionMapper
from processing.steps import PipelineStep


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

    @classmethod
    def validate_gender(cls, gender: str) -> Gender:
        """Validate and normalize gender value"""
        gender_lower = gender.lower().strip()
        if gender_lower in ["m", "male", "homme", "masculin"]:
            return Gender.MALE
        elif gender_lower in ["f", "female", "femme", "féminin"]:
            return Gender.FEMALE
        else:
            raise ValueError(f"Unknown gender: {gender}")

    @classmethod
    def get_name_category(cls, word_count: int) -> NameCategory:
        """Determine name category based on word count"""
        if word_count <= 3:
            return NameCategory.SIMPLE
        else:
            return NameCategory.COMPOSE

    def process_batch(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Extract features from names in batch"""
        logging.info(f"Extracting features for batch {batch_id} with {len(batch)} rows")

        batch = batch.copy()

        # Basic features
        batch["words"] = batch["name"].str.count(" ") + 1
        batch["length"] = batch["name"].str.replace(" ", "", regex=False).str.len()

        # Handle year column
        if "year" in batch.columns:
            batch["year"] = pd.to_numeric(batch["year"], errors="coerce").astype("Int64")

        # Initialize new columns
        batch["probable_native"] = None
        batch["probable_surname"] = None
        batch["identified_name"] = None
        batch["identified_surname"] = None
        batch["annotated"] = 0

        # Vectorized category assignment
        batch["identified_category"] = batch["words"].apply(
            lambda x: self.get_name_category(x).value
        )

        # Assign probable_native and probable_surname for all names
        name_splits = batch["name"].str.split()
        batch["probable_native"] = name_splits.apply(
            lambda x: " ".join(x[:-1]) if isinstance(x, list) and len(x) >= 2 else None
        )
        batch["probable_surname"] = name_splits.apply(
            lambda x: x[-1] if isinstance(x, list) and len(x) >= 2 else None
        )

        # Auto-assign for 3-word names
        three_word_mask = batch["words"] == 3
        batch.loc[three_word_mask, "identified_name"] = batch.loc[
            three_word_mask, "probable_native"
        ]
        batch.loc[three_word_mask, "identified_surname"] = batch.loc[
            three_word_mask, "probable_surname"
        ]
        batch.loc[three_word_mask, "annotated"] = 1

        # Map regions to provinces
        batch["province"] = self.region_mapper.map_regions_vectorized(batch["region"])

        # Normalize gender
        if "sex" in batch.columns:
            batch["sex"] = batch["sex"].apply(lambda x: self.validate_gender(str(x)).value)

        return batch
