"""Shared TensorFlow experiment model lifecycle."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import set_random_seed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from ners.experiments.metrics import calculate_metrics
from ners.models.base import ExperimentModel
from ners.utils import (
    configure_tensorflow,
    full_name_series,
    native_group_array,
    stable_text_buckets,
)


class NeuralNetworkModel(ExperimentModel):
    """Shared TensorFlow lifecycle for the neural models."""

    @abstractmethod
    def build_model(self, vocab_size: int, **kwargs) -> Any:
        """Create an untrained Keras model for the fitted vocabulary."""
        pass

    def fit(self, X: pl.DataFrame, y: pl.Series) -> ExperimentModel:
        """Fit the tokenizer, label encoder, and Keras model."""
        logging.info("Training %s", self.__class__.__name__)
        configure_tensorflow(
            self.config.model_params,
            random_seed=self.config.random_seed,
        )

        X_prepared = self.prepare_features(self.select_input_view(X))
        X_prepared = self._sanitize_sequences(X_prepared)

        label_encoder = self.label_encoder
        if label_encoder is None:
            label_encoder = LabelEncoder()
            self.label_encoder = label_encoder
        if not hasattr(label_encoder, "classes_"):
            label_encoder.fit(y.to_numpy())
        y_encoded = np.asarray(
            label_encoder.transform(y.to_numpy()),
            dtype=np.int64,
        )

        vocab_size = len(self.tokenizer.word_index) + 1 if self.tokenizer else 1000
        logging.info("Vocabulary size: %d", vocab_size)

        model = self.build_model(vocab_size=vocab_size, **self.config.model_params)
        self.model = model

        logging.info(
            "Fitting model with %d samples and %d features",
            X_prepared.shape[0],
            X_prepared.shape[1],
        )
        logging.info("Model parameters: %s", self.config.model_params)

        validation_fraction = float(self.config.model_params.get("validation_split", 0.1))
        validation_buckets = stable_text_buckets(
            pl.Series(native_group_array(X)),
            namespace=b"ners-neural-val",
            bucket_count=10_000,
        )
        validation_mask = validation_buckets < round(validation_fraction * 10_000)
        training_mask = ~validation_mask
        if not self._contains_all_labels(y_encoded[training_mask], y_encoded):
            raise ValueError("Grouped neural training partition is missing a class")
        if not self._contains_all_labels(y_encoded[validation_mask], y_encoded):
            raise ValueError("Grouped neural validation partition is missing a class")

        encoded_classes = np.unique(y_encoded)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=encoded_classes,
            y=y_encoded[training_mask],
        )
        class_weight = {
            int(label): float(weight)
            for label, weight in zip(encoded_classes, weights, strict=True)
        }
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=int(self.config.model_params.get("early_stopping_patience", 2)),
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=1,
                min_lr=1e-5,
            ),
        ]
        model.fit(
            X_prepared[training_mask],
            y_encoded[training_mask],
            epochs=self.config.model_params.get("epochs", 10),
            batch_size=self.config.model_params.get("batch_size", 64),
            validation_data=(X_prepared[validation_mask], y_encoded[validation_mask]),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )

        self.is_fitted = True
        return self

    @staticmethod
    def _contains_all_labels(values: np.ndarray, reference: np.ndarray) -> bool:
        return set(np.unique(values)) == set(np.unique(reference))

    def _sanitize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Clamp invalid token indices to OOV and return an int32 array.

        Negative or out-of-vocabulary indices can fail during GPU embedding updates.
        """
        try:
            if sequences is None:
                return sequences
            arr = np.asarray(sequences)
            if not np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.int64, copy=False)

            if self.tokenizer is not None and hasattr(self.tokenizer, "word_index"):
                if self.tokenizer.word_index:
                    max_idx = max(self.tokenizer.word_index.values())
                else:
                    max_idx = 0
                oov_index = self.tokenizer.word_index.get(
                    getattr(self.tokenizer, "oov_token", "<OOV>"), 1
                )
                invalid_mask = (arr < 0) | (arr > max_idx)
                invalid_mask &= arr != 0
                if invalid_mask.any():
                    arr[invalid_mask] = oov_index

                # Keras recurrent masks require padding after the final nonzero token.
                try:
                    nz = arr != 0
                    if nz.ndim == 2 and arr.shape[1] > 0:
                        has_nz = nz.any(axis=1)
                        indices = np.arange(arr.shape[1], dtype=np.int64)
                        last_pos = (nz * indices).max(axis=1)
                        last_pos = np.where(has_nz, last_pos, -1)
                        left_region = indices <= last_pos[:, None]
                        zero_inside = (~nz) & left_region
                        if zero_inside.any():
                            arr[zero_inside] = oov_index
                except Exception:
                    pass

            return arr.astype(np.int32, copy=False)
        except (TypeError, ValueError) as error:
            logging.debug("Sequence sanitization skipped: %s", error)
            return sequences

    def _collect_text_corpus(self, X: pl.DataFrame) -> list[str]:
        """Return the sole configured predictor as a Python string sequence."""

        return full_name_series(X).to_list()

    def cross_validate(
        self, X: pl.DataFrame, y: pl.Series, cv_folds: int = 5
    ) -> dict[str, float]:
        if self.label_encoder is None:
            raise ValueError("Train the model before cross-validation")
        configure_tensorflow(
            self.config.model_params,
            random_seed=self.config.random_seed,
        )
        labels = y.to_numpy()
        groups = native_group_array(X)
        cv = StratifiedGroupKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.config.random_seed
        )

        fold_metrics: dict[str, list[float]] = {metric: [] for metric in self.config.metrics}

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, labels, groups)):
            set_random_seed(self.config.random_seed + fold)
            fold_model = self.__class__(self.config)
            fold_model.fit(X[train_idx], y[train_idx])
            y_pred = fold_model.predict(X[val_idx])

            values = calculate_metrics(
                labels[val_idx],
                y_pred,
                self.config.metrics,
            )
            for metric, value in values.items():
                fold_metrics[metric].append(value)

        results: dict[str, float] = {}
        for metric, values in fold_metrics.items():
            results[metric] = float(np.mean(values))
            results[f"{metric}_std"] = float(np.std(values))
        return results
