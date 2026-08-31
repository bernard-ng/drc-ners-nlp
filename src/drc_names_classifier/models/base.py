"""Shared model contract and versioned artifact persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import polars as pl

from drc_names_classifier.config import ExperimentConfig
from drc_names_classifier.utils import name_view_expression, normalize_name_expression

if TYPE_CHECKING:
    from sklearn.preprocessing import LabelEncoder


EXPERIMENT_MODEL_FORMAT_VERSION = 1


class ExperimentModel(ABC):
    """Interface for trainable experiment models and their saved artifacts."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model: Any | None = None
        self.label_encoder: "LabelEncoder | None" = None
        self.tokenizer: Any | None = None  # For neural models
        self.is_fitted: bool = False

    @abstractmethod
    def prepare_features(self, X: pl.DataFrame) -> Any:
        """Convert full names to the input expected by the estimator."""
        pass

    @abstractmethod
    def fit(self, X: pl.DataFrame, y: pl.Series) -> ExperimentModel:
        """Fit the estimator to full names and sex labels."""
        pass

    @abstractmethod
    def cross_validate(
        self, X: pl.DataFrame, y: pl.Series, cv_folds: int = 5
    ) -> dict[str, float]:
        """Return mean cross-validation scores for the configured metrics."""
        pass

    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Predict sex for each full name."""
        if not self.is_fitted:
            raise ValueError("Train the model before making predictions")

        if self.model is None or self.label_encoder is None:
            raise ValueError("Model is not fully initialized for prediction")

        X_prepared = self.prepare_features(self.select_input_view(X))

        predictions: Any = self.model.predict(X_prepared)

        if hasattr(predictions, "shape") and len(predictions.shape) > 1:
            predictions = predictions.argmax(axis=1)

        return self.label_encoder.inverse_transform(predictions)

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return estimator-specific feature scores when available."""

        model = self.model
        if model is None:
            return None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = self._get_feature_names()
            return {
                str(name): float(value)
                for name, value in zip(feature_names, importances, strict=False)
            }

        elif hasattr(model, "coef_"):
            coefficients = model.coef_[0]
            feature_names = self._get_feature_names()
            top_indices = np.argsort(np.abs(coefficients))[-50:]
            return self._coefficient_map(feature_names, coefficients, top_indices)

        elif hasattr(model, "named_steps") and "classifier" in model.named_steps:
            classifier = model.named_steps["classifier"]
            if hasattr(classifier, "coef_"):
                coefficients = classifier.coef_[0]
                try:
                    feature_names = model[:-1].get_feature_names_out()
                except (AttributeError, ValueError):
                    feature_names = self._pipeline_feature_names(model)
                if len(feature_names) != len(coefficients):
                    return None
                top_indices = np.argsort(np.abs(coefficients))[-50:]
                return self._coefficient_map(feature_names, coefficients, top_indices)

        return None

    def select_input_view(self, X: pl.DataFrame) -> pl.DataFrame:
        """Enforce the artifact's saved name view at every estimator boundary."""

        normalized = X.select(normalize_name_expression("name").alias("name"))
        return normalized.select(name_view_expression(self.config.input_view).alias("name"))

    @staticmethod
    def _coefficient_map(
        feature_names: Any,
        coefficients: Any,
        indices: Any,
    ) -> dict[str, float]:
        """Keep signed coefficients; positive supports M and negative supports F."""

        return {str(feature_names[index]): float(coefficients[index]) for index in indices}

    @staticmethod
    def _pipeline_feature_names(model: Any) -> list[str]:
        """Resolve names from a position-aware FeatureUnion pipeline."""

        features = model.named_steps.get("features")
        transformers = getattr(features, "transformer_list", ())
        names: list[str] = []
        for channel_name, transformer in transformers:
            named_steps = getattr(transformer, "named_steps", {})
            vectorizer = named_steps.get("tfidf")
            if vectorizer is None or not hasattr(vectorizer, "get_feature_names_out"):
                continue
            names.extend(
                f"{channel_name}__{value}" for value in vectorizer.get_feature_names_out()
            )
        return names

    def _get_feature_names(self) -> list[str]:
        """Return vectorizer names or deterministic fallback names."""
        model = self.model
        vectorizer = getattr(self, "vectorizer", None)
        if vectorizer is not None and hasattr(vectorizer, "get_feature_names_out"):
            return [str(value) for value in vectorizer.get_feature_names_out()]
        if model is not None and hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        return [f"feature_{i}" for i in range(100)]

    def save(self, path: str | Path) -> Path:
        """Write the fitted estimator and its metadata to a joblib artifact."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "format_version": EXPERIMENT_MODEL_FORMAT_VERSION,
            "model": self.model,
            "label_encoder": self.label_encoder,
            "tokenizer": self.tokenizer,
            "config": self.config.to_dict(),
            "is_fitted": self.is_fitted,
        }
        for attribute in ("vectorizer",):
            if hasattr(self, attribute):
                model_data[attribute] = getattr(self, attribute)
        joblib.dump(model_data, destination, compress=3)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ExperimentModel:
        """Restore an artifact written by `save`."""
        model_data = joblib.load(Path(path))
        if not isinstance(model_data, dict):
            raise ValueError("Research model artifact must contain a dictionary")
        if model_data.get("format_version") != EXPERIMENT_MODEL_FORMAT_VERSION:
            raise ValueError("Unsupported experiment model artifact version")

        config = ExperimentConfig.from_dict(model_data["config"])
        instance = cls(config)

        instance.model = model_data["model"]
        instance.label_encoder = model_data["label_encoder"]
        instance.tokenizer = model_data.get("tokenizer")
        instance.is_fitted = model_data["is_fitted"]
        for attribute in ("vectorizer",):
            if attribute in model_data:
                setattr(instance, attribute, model_data[attribute])

        return instance
