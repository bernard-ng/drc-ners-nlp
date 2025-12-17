import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from ners.research.base_model import BaseModel
from ners.research.experiment.feature_extractor import FeatureExtractor


class NeuralNetworkModel(BaseModel):
    """Base class for neural network models (TensorFlow/Keras)"""

    @property
    def architecture(self) -> str:
        return "neural_network"

    @abstractmethod
    def build_model(self, vocab_size: int, **kwargs) -> Any:
        """Build neural network model with known vocabulary size"""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the neural network model with deferred building"""
        logging.info(f"Training {self.__class__.__name__}")

        # Best-effort GPU configuration for TensorFlow when available
        try:
            requested_gpu = bool(self.config.model_params.get("use_gpu", False))
            enable_mixed = bool(self.config.model_params.get("mixed_precision", False))

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass

                if enable_mixed:
                    try:
                        tf.keras.mixed_precision.set_global_policy("mixed_float16")
                        logging.info("Enabled TensorFlow mixed precision (float16)")
                    except Exception as e:
                        logging.warning(f"Could not enable mixed precision: {e}")
            else:
                if requested_gpu:
                    logging.warning(
                        "Requested GPU but no TensorFlow GPU device is available."
                    )
        except Exception as e:
            logging.debug(f"TensorFlow GPU setup skipped: {e}")

        # Setup feature extraction
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.config.features, self.config.feature_params
            )

        # Pyright fix: Access via local variable to ensure non-None type
        fe = self.feature_extractor
        assert fe is not None
        features_df = fe.extract_features(X)
        
        X_prepared = self.prepare_features(features_df)
        X_prepared = self._sanitize_sequences(X_prepared)

        # Encode labels
        # Pyright fix: Use local variable and cast to avoid reportOptionalMemberAccess
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            le = cast(LabelEncoder, self.label_encoder)
            y_encoded = le.fit_transform(y)
        else:
            le = cast(LabelEncoder, self.label_encoder)
            y_encoded = le.transform(y)

        # Now we can build the model with known vocab size
        vocab_size = len(self.tokenizer.word_index) + 1 if self.tokenizer else 1000
        logging.info(f"Vocabulary size: {vocab_size}")

        # Get additional model parameters
        self.model = self.build_model(vocab_size=vocab_size, **self.config.model_params)
        
        # Pyright fix: Explicitly check that model is built before training
        if self.model is None:
            raise ValueError("Model building failed: self.model is None")

        # Train the neural network
        logging.info(
            f"Fitting model with {X_prepared.shape[0]} samples and {X_prepared.shape[1]} features"
        )
        
        history = self.model.fit(
            X_prepared,
            y_encoded,
            epochs=self.config.model_params.get("epochs", 10),
            batch_size=self.config.model_params.get("batch_size", 64),
            validation_split=self.config.model_params.get("validation_split", 0.1),
            verbose=2,
        )

        # Store training history
        self.training_history = {
            "accuracy": history.history["accuracy"],
            "loss": history.history["loss"],
            "val_accuracy": history.history.get("val_accuracy", []),
            "val_loss": history.history.get("val_loss", []),
        }

        self.is_fitted = True
        return self

    def _sanitize_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Clamp invalid token indices to OOV and ensure int32 dtype."""
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
        except Exception as e:
            logging.debug(f"Sequence sanitization skipped due to: {e}")
            return sequences

    def _collect_text_corpus(self, X: pd.DataFrame) -> List[str]:
        """Combine configured textual features into one string per record."""
        column_names = [
            feature.value
            for feature in self.config.features
            if feature.value in X.columns
        ]
        if not column_names:
            raise ValueError(
                "No configured text features found in the provided DataFrame."
            )

        text_frame = X[column_names].fillna("").astype(str)

        if len(column_names) == 1:
            return text_frame.iloc[:, 0].tolist()

        combined_rows = []
        for row in text_frame.itertuples(index=False):
            tokens = [value for value in row if value]
            combined_rows.append(" ".join(tokens))

        return combined_rows

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
    ) -> dict[str, np.floating[Any]]:
        """Run cross-validation on the neural network model"""
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

        # Pyright fix: Explicitly verify and capture components to avoid None errors
        if self.feature_extractor is None or self.label_encoder is None:
            raise ValueError("Required components (FeatureExtractor/LabelEncoder) must be initialized.")
            
        fe, le = self.feature_extractor, self.label_encoder
        features_df = fe.extract_features(X)
        X_prepared = self.prepare_features(features_df)
        X_prepared = self._sanitize_sequences(X_prepared)
        y_encoded = le.transform(y)

        cv = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.config.random_seed
        )

        accuracies, precisions, recalls, f1_scores = [], [], [], []

        vocab_size = len(self.tokenizer.word_index) + 1 if self.tokenizer else 1000
        max_len = self.config.model_params.get("max_len", 6)

        for train_idx, val_idx in cv.split(X_prepared, y_encoded):
            fold_model = self.build_model(
                vocab_size=vocab_size, max_len=max_len, **self.config.model_params
            )

            if hasattr(fold_model, "fit"):
                fold_model.fit(
                    X_prepared[train_idx],
                    y_encoded[train_idx],
                    epochs=self.config.model_params.get("epochs", 10),
                    batch_size=self.config.model_params.get("batch_size", 32),
                    verbose=0,
                )

            y_pred = fold_model.predict(X_prepared[val_idx])
            if len(y_pred.shape) > 1:
                y_pred = y_pred.argmax(axis=1)

            acc = accuracy_score(y_encoded[val_idx], y_pred)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_encoded[val_idx], y_pred, average="weighted"
            )

            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)

        return {
            "accuracy": np.mean(accuracies),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1_scores),
        }

    def generate_learning_curve(
        self, X: pd.DataFrame, y: pd.Series, train_sizes: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Generate learning curve data for the model"""
        logging.info(f"Generating learning curve for {self.__class__.__name__}")

        # Pyright fix: Handle Optional train_sizes by assigning to a local non-optional list
        actual_train_sizes = train_sizes if train_sizes is not None else [0.1, 0.3, 0.5, 0.7, 1.0]

        learning_curve_data = {
            "train_sizes": [],
            "train_scores": [],
            "val_scores": [],
            "train_scores_std": [],
            "val_scores_std": [],
        }

        # Pyright fix: Capture components to local variables to ensure type safety
        if self.feature_extractor is None or self.label_encoder is None:
            raise ValueError("Required components (FeatureExtractor/LabelEncoder) must be initialized.")
            
        fe, le = self.feature_extractor, self.label_encoder
        features_df = fe.extract_features(X)
        X_prepared = self.prepare_features(features_df)
        X_prepared = self._sanitize_sequences(X_prepared)
        y_encoded = le.transform(y)

        vocab_size = len(self.tokenizer.word_index) + 1 if self.tokenizer else 1000
        max_len = self.config.model_params.get("max_len", 6)

        X_train_full, X_val, y_train_full, y_val = train_test_split(
            X_prepared,
            y_encoded,
            test_size=0.2,
            random_state=self.config.random_seed,
            stratify=y_encoded,
        )

        for size in actual_train_sizes:
            train_size = int(len(X_train_full) * size)
            if train_size < 10:
                continue

            indices = np.random.choice(len(X_train_full), train_size, replace=False)
            X_train_subset = X_train_full[indices]
            y_train_subset = y_train_full[indices]

            train_scores, val_scores = [], []

            for seed in range(3):
                model = self.build_model(
                    vocab_size=vocab_size, max_len=max_len, **self.config.model_params
                )

                if hasattr(model, "fit"):
                    model.fit(
                        X_train_subset,
                        y_train_subset,
                        epochs=self.config.model_params.get("epochs", 10),
                        batch_size=self.config.model_params.get("batch_size", 32),
                        validation_data=(X_val, y_val),
                        verbose=0,
                    )

                train_pred = model.predict(X_train_subset)
                val_pred = model.predict(X_val)

                train_acc = accuracy_score(y_train_subset, train_pred.argmax(axis=1))
                val_acc = accuracy_score(y_val, val_pred.argmax(axis=1))

                train_scores.append(train_acc)
                val_scores.append(val_acc)

            learning_curve_data["train_sizes"].append(train_size)
            learning_curve_data["train_scores"].append(np.mean(train_scores))
            learning_curve_data["val_scores"].append(np.mean(val_scores))
            learning_curve_data["train_scores_std"].append(np.std(train_scores))
            learning_curve_data["val_scores_std"].append(np.std(val_scores))

        self.learning_curve_data = learning_curve_data
        return learning_curve_data
