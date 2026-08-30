# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import polars as pl

from ners.research.models.features import SparseTextFeatures
from ners.research.models.sklearn import SklearnModel


class LightGBMModel(SparseTextFeatures, SklearnModel):
    """LightGBM full-name classifier over sparse character TF-IDF."""

    def build_model(self) -> Any:
        params = self.config.model_params
        return lgb.LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 63)),
            max_depth=int(params.get("max_depth", -1)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            class_weight=params.get("class_weight", "balanced"),
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=self.config.random_seed,
            verbosity=int(params.get("verbosity", -1)),
        )

    def prepare_features(self, X: pl.DataFrame) -> Any:
        return self.prepare_sparse_text(X)
