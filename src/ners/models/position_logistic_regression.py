"""Position-aware sparse linear model for ordered three-token names."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ners.models.sklearn import SklearnModel
from ners.utils import full_name_array


class TokenPositionSelector(TransformerMixin, BaseEstimator):
    """Select one token and expose its position to scikit-learn cloning."""

    def __init__(self, position: int | Literal["sequence"]) -> None:
        self.position = position

    def fit(self, X: Any, y: Any = None) -> TokenPositionSelector:
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=str).reshape(-1)
        position = self.position
        if isinstance(position, str):
            return values

        selected: list[str] = []
        for value in values:
            tokens = value.split()
            selected.append(tokens[position] if position < len(tokens) else "<missing>")
        return np.asarray(selected)


class PositionAwareLogisticRegressionModel(SklearnModel):
    """Logistic regression with separate sequence and token-position channels.

    The third channel contains the surname in the controlled three-token cohort. The
    native-only variant has two tokens, so this channel receives a constant missing value.
    """

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        sequence_ngram_range = self._ngram_range("ngram_range", (2, 5))
        token_ngram_range = self._ngram_range("token_ngram_range", (2, 5))
        sequence_features = int(params.get("sequence_max_features", 60_000))
        token_features = int(params.get("token_max_features", 20_000))
        min_df = int(params.get("min_df", 2))

        channels = FeatureUnion(
            [
                (
                    "sequence",
                    self._channel(
                        "sequence",
                        sequence_ngram_range,
                        sequence_features,
                        min_df,
                    ),
                ),
                (
                    "token_1",
                    self._channel(0, token_ngram_range, token_features, min_df),
                ),
                (
                    "token_2",
                    self._channel(1, token_ngram_range, token_features, min_df),
                ),
                (
                    "token_3_surname",
                    self._channel(2, token_ngram_range, token_features, min_df),
                ),
            ]
        )
        classifier = LogisticRegression(
            C=float(params.get("C", 2.0)),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=int(params.get("max_iter", 700)),
            random_state=self.config.random_seed,
            solver=str(params.get("solver", "saga")),
            tol=float(params.get("tol", 1e-4)),
        )
        return Pipeline([("features", channels), ("classifier", classifier)])

    def prepare_features(self, X: pl.DataFrame) -> np.ndarray:
        return full_name_array(X)

    @staticmethod
    def _channel(
        position: int | Literal["sequence"],
        ngram_range: tuple[int, int],
        max_features: int,
        min_df: int,
    ) -> Pipeline:
        return Pipeline(
            [
                ("select", TokenPositionSelector(position)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=ngram_range,
                        max_features=max_features,
                        min_df=min_df,
                        sublinear_tf=True,
                        dtype=np.float32,  # pyright: ignore[reportArgumentType]
                    ),
                ),
            ]
        )

    def _ngram_range(self, key: str, default: tuple[int, int]) -> tuple[int, int]:
        value = self.config.model_params.get(key, default)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{key} must contain exactly two integers")
        return int(value[0]), int(value[1])
