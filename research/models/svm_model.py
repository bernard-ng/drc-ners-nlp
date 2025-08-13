import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from research.traditional_model import TraditionalModel


class SVMModel(TraditionalModel):
    """Support Vector Machine with character n-grams and RBF kernel"""

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=params.get("ngram_range", (2, 4)),
            max_features=params.get("max_features", 5000),
        )

        classifier = SVC(
            kernel=params.get("kernel", "rbf"),
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            probability=True,  # Enable probability prediction
            random_state=self.config.random_seed,
            verbose=2,
        )

        return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        text_features = []

        for feature_type in self.config.features:
            if feature_type.value in X.columns:
                text_features.append(X[feature_type.value].astype(str))

        if len(text_features) == 1:
            return text_features[0].values
        else:
            combined = text_features[0].astype(str)
            for feature in text_features[1:]:
                combined = combined + " " + feature.astype(str)
            return combined.values
