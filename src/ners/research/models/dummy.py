"""Non-informative control model for interpreting study metrics."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier

from ners.research.models.sklearn import SklearnModel


class DummyBaselineModel(SklearnModel):
    """Predict the training-set class prior without looking at the name."""

    def build_model(self) -> BaseEstimator:
        return DummyClassifier(strategy="prior", random_state=self.config.random_seed)

    def prepare_features(self, X: pl.DataFrame) -> np.ndarray:
        return np.ones((X.height, 1), dtype=np.float32)
