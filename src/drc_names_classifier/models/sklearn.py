"""scikit-learn training and validation."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from drc_names_classifier.experiments.metrics import calculate_metrics
from drc_names_classifier.models.base import ExperimentModel
from drc_names_classifier.utils import native_group_array


class SklearnModel(ExperimentModel):
    """Shared lifecycle for scikit-learn compatible models."""

    @abstractmethod
    def build_model(self) -> Any:
        """Create an unfitted estimator."""
        pass

    def fit(self, X: pl.DataFrame, y: pl.Series) -> ExperimentModel:
        """Fit the estimator and its label encoder."""
        logging.info("Training %s", self.__class__.__name__)

        model = self.model
        if model is None:
            model = self.build_model()
            self.model = model

        X_prepared = self.prepare_features(self.select_input_view(X))

        label_encoder = self.label_encoder
        if label_encoder is None:
            label_encoder = LabelEncoder()
            self.label_encoder = label_encoder
        if not hasattr(label_encoder, "classes_"):
            label_encoder.fit(y.to_numpy())
        y_encoded = label_encoder.transform(y.to_numpy())

        if len(X_prepared.shape) == 1:
            logging.info("Fitting model with %d text samples", X_prepared.shape[0])
        else:
            logging.info(
                "Fitting model with %d samples and %d features",
                X_prepared.shape[0],
                X_prepared.shape[1],
            )
        logging.info("Model parameters: %s", self.config.model_params)

        model.fit(X_prepared, y_encoded)
        self.is_fitted = True
        return self

    def cross_validate(
        self, X: pl.DataFrame, y: pl.Series, cv_folds: int = 5
    ) -> dict[str, float]:
        if self.label_encoder is None or self.model is None:
            raise ValueError("Train the model before cross-validation")
        labels = y.to_numpy()
        groups = native_group_array(X)
        cv = StratifiedGroupKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.config.random_seed
        )
        fold_metrics: dict[str, list[float]] = {metric: [] for metric in self.config.metrics}
        for train_indices, validation_indices in cv.split(X, labels, groups):
            fold_model = self.__class__(self.config)
            fold_model.fit(X[train_indices], y[train_indices])
            predictions = fold_model.predict(X[validation_indices])
            values = calculate_metrics(
                labels[validation_indices],
                predictions,
                self.config.metrics,
            )
            for metric, value in values.items():
                fold_metrics[metric].append(value)

        results: dict[str, float] = {}
        for metric, values in fold_metrics.items():
            if not values:
                continue
            results[metric] = float(np.mean(values))
            results[f"{metric}_std"] = float(np.std(values))
        return results
