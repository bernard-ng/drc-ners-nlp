"""scikit-learn training, validation, and learning curves."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedGroupKFold, learning_curve
from sklearn.preprocessing import LabelEncoder

from ners.research.experiment.metrics import calculate_metrics
from ners.research.models.base import ResearchModel
from ners.utils import native_group_array


class SklearnModel(ResearchModel):
    """Shared lifecycle for scikit-learn compatible models."""

    @property
    def architecture(self) -> str:
        return "sklearn"

    @abstractmethod
    def build_model(self) -> BaseEstimator:
        """Create an unfitted estimator."""
        pass

    def fit(self, X: pl.DataFrame, y: pl.Series) -> ResearchModel:
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
        # scikit-learn estimators do not return an epoch history.
        self.training_history = {}

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

    def generate_learning_curve(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        train_sizes: list[float] | None = None,
    ) -> dict[str, Any]:
        """Measure accuracy at several training sizes with three-fold validation."""
        logging.info("Generating learning curve for %s", self.__class__.__name__)

        if train_sizes is None:
            train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]

        X_prepared = self.prepare_features(self.select_input_view(X))

        label_encoder = self.label_encoder
        if label_encoder is None:
            label_encoder = LabelEncoder()
            self.label_encoder = label_encoder
        if not hasattr(label_encoder, "classes_"):
            label_encoder.fit(y.to_numpy())
        y_encoded = label_encoder.transform(y.to_numpy())

        try:
            curve = learning_curve(
                self.build_model(),
                X_prepared,
                y_encoded,
                train_sizes=np.asarray(train_sizes, dtype=np.float64),
                cv=3,
                scoring="accuracy",
                shuffle=True,
                random_state=self.config.random_seed,
                n_jobs=int(self.config.model_params.get("n_jobs", -1)),
            )
            train_sizes_abs, train_scores, val_scores = curve[:3]

            learning_curve_data = {
                "train_sizes": train_sizes_abs.tolist(),
                "train_scores": train_scores.mean(axis=1).tolist(),
                "val_scores": val_scores.mean(axis=1).tolist(),
                "train_scores_std": train_scores.std(axis=1).tolist(),
                "val_scores_std": val_scores.std(axis=1).tolist(),
            }
        except (TypeError, ValueError) as error:
            logging.warning("Could not generate learning curve: %s", error)
            return {}

        self.learning_curve_data = learning_curve_data
        return learning_curve_data
