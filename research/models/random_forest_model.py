import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from research.traditional_model import TraditionalModel


class RandomForestModel(TraditionalModel):
    """Random Forest with engineered features"""

    def build_model(self) -> BaseEstimator:

        params = self.config.model_params

        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=self.config.random_seed,
            verbose=2,
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
                    # Categorical features (encode them)
                    le = LabelEncoder()
                    encoded = le.fit_transform(column.fillna("unknown").astype(str))
                    features.append(encoded.reshape(-1, 1))

        return np.hstack(features) if features else np.array([]).reshape(len(X), 0)
