import logging
from abc import abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from research.base_model import BaseModel
from research.experiment.feature_extractor import FeatureExtractor


class NeuralNetworkModel(BaseModel):
    """Base class for neural network models (TensorFlow/Keras)"""

    @property
    def architecture(self) -> str:
        return "neural_network"

    @abstractmethod
    def build_model_with_vocab(self, vocab_size: int, **kwargs) -> Any:
        """Build neural network model with known vocabulary size"""
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the neural network model with deferred building"""
        logging.info(f"Training {self.__class__.__name__}")

        # Setup feature extraction
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.config.features, self.config.feature_params
            )

        # Extract and prepare features (this will also initialize tokenizer)
        features_df = self.feature_extractor.extract_features(X)
        X_prepared = self.prepare_features(features_df)

        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)

        # Now we can build the model with known vocab size
        vocab_size = len(self.tokenizer.word_index) + 1 if self.tokenizer else 1000

        # Get additional model parameters
        max_len = self.config.model_params.get("max_len", 6)

        self.model = self.build_model_with_vocab(
            vocab_size=vocab_size, max_len=max_len, **self.config.model_params
        )

        # Train the neural network
        history = self.model.fit(
            X_prepared,
            y_encoded,
            epochs=self.config.model_params.get("epochs", 10),
            batch_size=self.config.model_params.get("batch_size", 64),
            validation_split=0.1,
            verbose=1,
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

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
    ) -> dict[str, np.floating[Any]]:
        features_df = self.feature_extractor.extract_features(X)
        X_prepared = self.prepare_features(features_df)
        y_encoded = self.label_encoder.transform(y)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.random_seed)

        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_prepared, y_encoded)):
            # Create fresh model for each fold
            fold_model = self.build_model()

            # Train on fold
            if hasattr(fold_model, "fit"):
                fold_model.fit(
                    X_prepared[train_idx],
                    y_encoded[train_idx],
                    epochs=self.config.model_params.get("epochs", 10),
                    batch_size=self.config.model_params.get("batch_size", 32),
                    verbose=0,
                )

            # Predict on validation
            y_pred = fold_model.predict(X_prepared[val_idx])
            if len(y_pred.shape) > 1:
                y_pred = y_pred.argmax(axis=1)

            # Calculate metrics
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
            "accuracy_std": np.std(accuracies),
            "precision": np.mean(precisions),
            "precision_std": np.std(precisions),
            "recall": np.mean(recalls),
            "recall_std": np.std(recalls),
            "f1": np.mean(f1_scores),
            "f1_std": np.std(f1_scores),
        }

    def generate_learning_curve(
        self, X: pd.DataFrame, y: pd.Series, train_sizes: List[float] = None
    ) -> Dict[str, Any]:
        """Generate learning curve data for the model"""
        logging.info(f"Generating learning curve for {self.__class__.__name__}")

        learning_curve_data = {
            "train_sizes": [],
            "train_scores": [],
            "val_scores": [],
            "train_scores_std": [],
            "val_scores_std": [],
        }

        # Split data once for validation
        X_train_full, X_val, y_train_full, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.config.random_seed, stratify=y
        )

        for size in train_sizes:
            train_size = int(len(X_train_full) * size)
            if train_size < 10:  # Minimum training size
                continue

            # Sample training data
            indices = np.random.choice(len(X_train_full), train_size, replace=False)
            X_train_subset = X_train_full[indices]
            y_train_subset = y_train_full[indices]

            # Train multiple models for variance estimation
            train_scores = []
            val_scores = []

            for seed in range(3):  # 3 runs for variance
                # Build fresh model
                model = self.build_model()

                # Train model
                if hasattr(model, "fit"):
                    history = model.fit(
                        X_train_subset,
                        y_train_subset,
                        epochs=self.config.model_params.get("epochs", 10),
                        batch_size=self.config.model_params.get("batch_size", 32),
                        validation_data=(X_val, y_val),
                        verbose=0,
                    )

                # Evaluate
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
