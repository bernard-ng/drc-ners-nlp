import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ners.research.traditional_model import TraditionalModel


class NaiveBayesModel(TraditionalModel):
    """Multinomial Naive Bayes with character n-grams"""

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        # Bag-of-character-ngrams aligns with Multinomial NB assumptions; (1,4)
        # includes unigrams for coverage and higher n for suffix/prefix cues.
        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=params.get("ngram_range", (2, 5)),
            max_features=params.get("max_features", 8000),
        )

        # Laplace smoothing (alpha) counters zero counts for rare n-grams.
        classifier = MultinomialNB(alpha=params.get("alpha", 1.0))

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
