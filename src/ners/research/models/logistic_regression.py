import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from ners.research.models.sklearn import SklearnModel
from ners.utils import full_name_array


class LogisticRegressionModel(SklearnModel):
    """Logistic regression full-name classifier with character n-grams."""

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        ngram_range = params.get("ngram_range", (2, 4))
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)

        vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=params.get("max_features", 10000),
            min_df=int(params.get("min_df", 1)),
            binary=bool(params.get("binary", True)),
        )

        classifier = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 1000)),
            tol=float(params.get("tol", 1e-4)),
            random_state=self.config.random_seed,
            verbose=params.get("verbose", 0),
            solver=params.get("solver", "saga"),
            class_weight=params.get("class_weight", "balanced"),
        )

        return Pipeline(
            [
                ("vectorizer", vectorizer),
                ("scale", MaxAbsScaler(copy=False)),
                ("classifier", classifier),
            ]
        )

    def prepare_features(self, X: pl.DataFrame) -> np.ndarray:
        return full_name_array(X)
