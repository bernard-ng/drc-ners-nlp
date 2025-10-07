from enum import Enum
from typing import List, Dict, Any, Union, Optional

import pandas as pd


class FeatureType(Enum):
    """Types of features that can be extracted from names"""

    FULL_NAME = "full_name"
    NATIVE_NAME = "native_name"
    SURNAME = "surname"
    FIRST_WORD = "first_word"
    LAST_WORD = "last_word"
    NAME_LENGTH = "name_length"
    WORD_COUNT = "word_count"
    PROVINCE = "province"
    CHAR_NGRAMS = "char_ngrams"
    WORD_NGRAMS = "word_ngrams"
    NAME_ENDINGS = "name_endings"
    NAME_BEGINNINGS = "name_beginnings"


class FeatureExtractor:
    """Extract different types of features from name data"""

    def __init__(
        self,
        feature_types: List[FeatureType],
        feature_params: Optional[Dict[str, Any]] = None,
    ):
        self.feature_types = feature_types
        self.feature_params = feature_params or {}

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all configured features"""
        features_df = pd.DataFrame(index=df.index)

        for feature_type in self.feature_types:
            feature_data = self._extract_single_feature(df, feature_type)

            if isinstance(feature_data, pd.DataFrame):
                features_df = pd.concat([features_df, feature_data], axis=1)
            else:
                features_df[feature_type.value] = feature_data

        return features_df

    def _extract_single_feature(
        self, df: pd.DataFrame, feature_type: FeatureType
    ) -> Union[pd.Series, pd.DataFrame]:
        """Extract a single type of feature"""
        if feature_type == FeatureType.FULL_NAME:
            return df["name"].fillna("")

        elif feature_type == FeatureType.NATIVE_NAME:
            return df["identified_name"].fillna(df["probable_native"]).fillna("")

        elif feature_type == FeatureType.SURNAME:
            return df["identified_surname"].fillna(df["probable_surname"]).fillna("")

        elif feature_type == FeatureType.FIRST_WORD:
            return df["name"].str.split().str[0].fillna("")

        elif feature_type == FeatureType.LAST_WORD:
            return df["name"].str.split().str[-1].fillna("")

        elif feature_type == FeatureType.NAME_LENGTH:
            return df["name"].str.len().fillna(0)

        elif feature_type == FeatureType.WORD_COUNT:
            return df["words"].fillna(1)

        elif feature_type == FeatureType.PROVINCE:
            return df["province"].fillna("unknown")

        elif feature_type == FeatureType.NAME_ENDINGS:
            n = self.feature_params.get("ending_length", 3)
            return df["name"].str[-n:].fillna("")

        elif feature_type == FeatureType.NAME_BEGINNINGS:
            n = self.feature_params.get("beginning_length", 3)
            return df["name"].str[:n].fillna("")

        elif feature_type == FeatureType.CHAR_NGRAMS:
            # This will be handled by the model's vectorizer
            return df["name"].fillna("")

        elif feature_type == FeatureType.WORD_NGRAMS:
            # This will be handled by the model's vectorizer
            return df["name"].fillna("")

        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
