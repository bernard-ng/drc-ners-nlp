from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from ners.dataset import DatasetBatch, LABELS


MODEL_FORMAT_VERSION = 1


class ModelArtifactError(ValueError):
    """Raised when a saved model artifact is missing or incompatible."""


class NameSexClassifier:
    """Online classifier for the corpus sex field."""

    def __init__(
        self,
        *,
        n_features: int,
        ngram_range: tuple[int, int],
        alpha: float,
        random_seed: int,
        class_weight: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            lowercase=True,
            dtype=np.float32,  # pyright: ignore[reportArgumentType]
            norm="l2",
        )
        self.classifier = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            learning_rate="optimal",
            average=True,
            class_weight=class_weight,
            random_state=random_seed,
        )
        self.metadata = metadata or {}

    @property
    def is_fitted(self) -> bool:
        return hasattr(self.classifier, "classes_")

    @property
    def classes(self) -> tuple[str, ...]:
        if not self.is_fitted:
            return LABELS
        classes = getattr(self.classifier, "classes_", LABELS)
        return tuple(str(label) for label in classes)

    def partial_fit(self, batch: DatasetBatch) -> None:
        matrix = self.vectorizer.transform(batch.names)
        self.classifier.partial_fit(matrix, batch.labels, classes=np.asarray(LABELS))

    def predict(self, names: np.ndarray | list[str]) -> np.ndarray:
        self._require_fitted()
        matrix = self.vectorizer.transform(names)
        return self.classifier.predict(matrix)

    def predict_proba(self, names: np.ndarray | list[str]) -> np.ndarray:
        self._require_fitted()
        matrix = self.vectorizer.transform(names)
        return self.classifier.predict_proba(matrix)

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "metadata": self.metadata,
            },
            destination,
            compress=3,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> NameSexClassifier:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Model artifact not found: {source}")

        artifact = joblib.load(source)
        if not isinstance(artifact, dict):
            raise ModelArtifactError("Model artifact is not a dictionary")
        if artifact.get("format_version") != MODEL_FORMAT_VERSION:
            raise ModelArtifactError("Unsupported model artifact version")

        vectorizer = artifact.get("vectorizer")
        classifier = artifact.get("classifier")
        metadata = artifact.get("metadata", {})
        if not isinstance(vectorizer, HashingVectorizer):
            raise ModelArtifactError("Artifact has no compatible vectorizer")
        if not isinstance(classifier, SGDClassifier):
            raise ModelArtifactError("Artifact has no compatible classifier")

        instance = cls.__new__(cls)
        instance.vectorizer = vectorizer
        instance.classifier = classifier
        instance.metadata = metadata if isinstance(metadata, dict) else {}
        return instance

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
