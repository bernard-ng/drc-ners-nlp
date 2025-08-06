import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

from research.traditional_model import TraditionalModel


class XGBoostModel(TraditionalModel):
    """XGBoost with engineered features and character embeddings"""

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
                    vectorizer = CountVectorizer(
                        analyzer="char", ngram_range=(2, 3), max_features=100
                    )
                    char_features = vectorizer.fit_transform(
                        column.fillna("").astype(str)
                    ).toarray()
                    features.append(char_features)
                else:
                    # Categorical features
                    le = LabelEncoder()
                    encoded = le.fit_transform(column.fillna("unknown").astype(str))
                    features.append(encoded.reshape(-1, 1))

        return np.hstack(features) if features else np.array([]).reshape(len(X), 0)
