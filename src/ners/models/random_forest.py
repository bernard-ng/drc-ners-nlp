from __future__ import annotations

from typing import Any

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from ners.models.features import SparseTextFeatures
from ners.models.sklearn import SklearnModel


class RandomForestModel(SparseTextFeatures, SklearnModel):
    """Random-forest full-name classifier over character TF-IDF."""

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth", 24),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            max_features=params.get("tree_max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced_subsample"),
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=self.config.random_seed,
        )

    def prepare_features(self, X: pl.DataFrame) -> Any:
        return self.prepare_sparse_text(X)
