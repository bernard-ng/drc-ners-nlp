# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import Any

import polars as pl
import xgboost as xgb
from sklearn.base import BaseEstimator

from drc_names_classifier.models.features import SparseTextFeatures
from drc_names_classifier.models.sklearn import SklearnModel


class XGBoostModel(SparseTextFeatures, SklearnModel):
    """XGBoost full-name classifier over sparse character TF-IDF."""

    def build_model(self) -> BaseEstimator:
        params = self.config.model_params
        return xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 8)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.8)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            tree_method=str(params.get("tree_method", "hist")),
            eval_metric="logloss",
            n_jobs=int(params.get("n_jobs", -1)),
            random_state=self.config.random_seed,
            verbosity=int(params.get("verbosity", 0)),
        )

    def prepare_features(self, X: pl.DataFrame) -> Any:
        return self.prepare_sparse_text(X)
