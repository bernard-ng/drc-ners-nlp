import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from drc_names_classifier.models.sklearn import SklearnModel
from drc_names_classifier.utils import full_name_array


class NaiveBayesModel(SklearnModel):
    """Multinomial Naive Bayes full-name classifier."""

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        ngram_range = params.get("ngram_range", (2, 4))
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)

        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=params.get("max_features", 8000),
        )

        classifier = MultinomialNB(alpha=params.get("alpha", 1.0))

        return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])

    def prepare_features(self, X: pl.DataFrame) -> np.ndarray:
        return full_name_array(X)
