import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from research.traditional_model import TraditionalModel


class XGBoostModel(TraditionalModel):
    """XGBoost with engineered features and character embeddings"""

    def __init__(self, config):
        super().__init__(config)
        # Store vectorizers and encoders to ensure consistent feature space
        self.vectorizers = {}
        self.label_encoders = {}

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params

        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            random_state=self.config.random_seed,
            eval_metric="logloss",
            verbosity=2
        )

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        features = []

        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                column = X[feature_type.value]

                if feature_type.value in ["name_length", "word_count"]:
                    # Numerical features
                    features.append(column.fillna(0).values.reshape(-1, 1))
                elif feature_type.value in ["full_name", "native_name", "surname"]:
                    # Character-level features for names
                    feature_key = f"vectorizer_{feature_type.value}"

                    if feature_key not in self.vectorizers:
                        # First time - create and fit vectorizer
                        self.vectorizers[feature_key] = CountVectorizer(
                            analyzer="char", ngram_range=(2, 3), max_features=100
                        )
                        char_features = self.vectorizers[feature_key].fit_transform(
                            column.fillna("").astype(str)
                        ).toarray()
                    else:
                        # Subsequent times - use existing vectorizer
                        char_features = self.vectorizers[feature_key].transform(
                            column.fillna("").astype(str)
                        ).toarray()

                    features.append(char_features)
                else:
                    # Categorical features
                    feature_key = f"encoder_{feature_type.value}"

                    if feature_key not in self.label_encoders:
                        # First time - create and fit encoder
                        self.label_encoders[feature_key] = LabelEncoder()
                        encoded = self.label_encoders[feature_key].fit_transform(
                            column.fillna("unknown").astype(str)
                        )
                    else:
                        # Subsequent times - use existing encoder
                        # Handle unseen labels by mapping them to a default value
                        column_clean = column.fillna("unknown").astype(str)

                        # Get the classes the encoder was trained on
                        known_classes = set(self.label_encoders[feature_key].classes_)

                        # Map unseen values to "unknown" if it exists, otherwise to the first class
                        if "unknown" in known_classes:
                            default_class = "unknown"
                        else:
                            default_class = self.label_encoders[feature_key].classes_[0]

                        # Replace unseen values with default
                        column_mapped = column_clean.apply(
                            lambda x: x if x in known_classes else default_class
                        )

                        encoded = self.label_encoders[feature_key].transform(column_mapped)

                    features.append(encoded.reshape(-1, 1))

        return np.hstack(features) if features else np.array([]).reshape(len(X), 0)
