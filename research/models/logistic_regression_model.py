import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from research.traditional_model import TraditionalModel


class LogisticRegressionModel(TraditionalModel):
    """Logistic Regression with character n-grams"""

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        # Character n-grams are strong signals for names; (2,5) balances
        # capturing prefixes/suffixes with tractable feature size.
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=params.get("ngram_range", (2, 5)),
            max_features=params.get("max_features", 10000),
        )

        # liblinear handles sparse, small-to-medium problems well; n_jobs parallelizes
        # OvR across classes (no effect for binary). class_weight can mitigate imbalance.
        classifier = LogisticRegression(
            max_iter=params.get("max_iter", 1000),
            random_state=self.config.random_seed,
            verbose=2,
            solver=params.get("solver", "liblinear"),
            n_jobs=params.get("n_jobs", -1),
            class_weight=params.get("class_weight", None),
        )

        return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        text_features = []

        # Collect text-based features from the extracted features DataFrame
        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                text_features.append(X[feature_type.value].astype(str))

        # Combine text features
        if len(text_features) == 1:
            return text_features[0].values
        else:
            # Concatenate multiple text features with separator
            combined = text_features[0].astype(str)
            for feature in text_features[1:]:
                combined = combined + " " + feature.astype(str)
            return combined.values
