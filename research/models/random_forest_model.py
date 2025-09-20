from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from research.traditional_model import TraditionalModel


class RandomForestModel(TraditionalModel):
    """Random Forest with engineered features"""

    def __init__(self, config):
        super().__init__(config)
        # Persist encoders so categorical mappings stay consistent.
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def build_model(self) -> BaseEstimator:

        params = self.config.model_params

        # Tree ensemble is robust to mixed numeric/categorical encodings; parallelize
        # across trees for speed. Keep depth moderate for generalisation.
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=self.config.random_seed,
            verbose=2,
            n_jobs=params.get("n_jobs", -1),
        )

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        features = []

        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                column = X[feature_type.value]

                # Handle different feature types
                if feature_type.value in ["name_length", "word_count"]:
                    # Numerical features
                    features.append(column.fillna(0).values.reshape(-1, 1))
                else:
                    # Categorical features (encode them persistently)
                    feature_key = f"encoder_{feature_type.value}"

                    if feature_key not in self.label_encoders:
                        self.label_encoders[feature_key] = LabelEncoder()
                        encoded = self.label_encoders[feature_key].fit_transform(
                            column.fillna("unknown").astype(str)
                        )
                    else:
                        encoder = self.label_encoders[feature_key]
                        column_clean = column.fillna("unknown").astype(str)
                        known_classes = set(encoder.classes_)
                        default_class = "unknown" if "unknown" in known_classes else encoder.classes_[0]
                        column_mapped = column_clean.apply(
                            lambda value: value if value in known_classes else default_class
                        )
                        encoded = encoder.transform(column_mapped)

                    features.append(encoded.reshape(-1, 1))

        return np.hstack(features) if features else np.array([]).reshape(len(X), 0)
