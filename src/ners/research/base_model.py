import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ners.research.experiment import ExperimentConfig

if TYPE_CHECKING:
    from ners.research.experiment.feature_extractor import FeatureExtractor
    from sklearn.preprocessing import LabelEncoder


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model: Any | None = None
        self.feature_extractor: "FeatureExtractor | None" = None
        self.label_encoder: "LabelEncoder | None" = None
        self.tokenizer: Any | None = None  # For neural models
        self.is_fitted: bool = False
        self.training_history: Dict[str, Any] = {}  # For learning curves
        self.learning_curve_data: Dict[str, Any] = {}

    @property
    @abstractmethod
    def architecture(self) -> str:
        """Return the architecture type: 'neural_network', 'traditional', or 'ensemble'"""
        pass

    @abstractmethod
    def prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare features for training/prediction"""
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model - implemented differently for each architecture"""
        pass

    @abstractmethod
    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5
    ) -> Dict[str, float] | dict[str, np.floating[Any]]:
        """Perform cross-validation and return average scores"""
        pass

    @abstractmethod
    def generate_learning_curve(
        self, X: pd.DataFrame, y: pd.Series, train_sizes: List[float] = []
    ) -> Dict[str, Any]:
        """Generate learning curve data for the model"""
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if (
            self.feature_extractor is None
            or self.model is None
            or self.label_encoder is None
        ):
            raise ValueError("Model is not fully initialized for prediction")

        features_df = self.feature_extractor.extract_features(X)
        X_prepared = self.prepare_features(features_df)

        predictions: Union[np.ndarray, Any] = self.model.predict(X_prepared)

        # Handle different prediction formats
        if hasattr(predictions, "shape") and len(predictions.shape) > 1:
            # Neural network outputs (probabilities)
            predictions = predictions.argmax(axis=1)

        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities if supported"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.feature_extractor is None or self.model is None:
            raise ValueError("Model is not fully initialized for prediction")

        features_df = self.feature_extractor.extract_features(X)
        X_prepared = self.prepare_features(features_df)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_prepared)
        elif hasattr(self.model, "predict"):
            # For neural networks that return probabilities directly
            probabilities = self.model.predict(X_prepared)
            if (
                hasattr(probabilities, "shape")
                and len(probabilities.shape) == 2
                and probabilities.shape[1] > 1
            ):
                return probabilities

        raise NotImplementedError("Model does not support probability predictions")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if supported by the model"""

        model = self.model
        if model is None:
            return None

        if hasattr(model, "feature_importances_"):
            # For tree-based models
            importances = model.feature_importances_
            feature_names = self._get_feature_names()
            return dict(zip(feature_names, importances))

        elif hasattr(model, "coef_"):
            # For linear models
            coefficients = np.abs(model.coef_[0])
            feature_names = self._get_feature_names()
            return dict(zip(feature_names, coefficients))

        elif hasattr(model, "named_steps") and "classifier" in model.named_steps:
            # For sklearn pipelines (like LogisticRegression with vectorizer)
            classifier = model.named_steps["classifier"]
            if hasattr(classifier, "coef_"):
                coefficients = np.abs(classifier.coef_[0])
                if hasattr(model.named_steps["vectorizer"], "get_feature_names_out"):
                    feature_names = model.named_steps[
                        "vectorizer"
                    ].get_feature_names_out()
                    # Take top features to avoid too many n-grams
                    top_indices = np.argsort(coefficients)[-20:]
                    return dict(
                        zip(feature_names[top_indices], coefficients[top_indices])
                    )

        return None

    def _get_feature_names(self) -> List[str]:
        """Get feature names (override in subclasses if needed)"""
        model = self.model
        if model is not None and hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        return [f"feature_{i}" for i in range(100)]  # Default fallback

    def save(self, path: str):
        """Save the complete model with training history"""

        model_data = {
            "model": self.model,
            "feature_extractor": self.feature_extractor,
            "label_encoder": self.label_encoder,
            "tokenizer": self.tokenizer,
            "config": self.config.to_dict(),
            "is_fitted": self.is_fitted,
            "training_history": self.training_history,
            "learning_curve_data": self.learning_curve_data,
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load a saved model with training history"""
        model_data = joblib.load(path)

        # Recreate the model instance
        from ners.research.experiment import ExperimentConfig

        config = ExperimentConfig.from_dict(model_data["config"])
        instance = cls(config)

        # Restore state
        instance.model = model_data["model"]
        instance.feature_extractor = model_data["feature_extractor"]
        instance.label_encoder = model_data["label_encoder"]
        instance.tokenizer = model_data.get("tokenizer")
        instance.is_fitted = model_data["is_fitted"]
        instance.training_history = model_data.get("training_history", {})
        instance.learning_curve_data = model_data.get("learning_curve_data", {})

        return instance

    def plot_learning_curve(self, save_path: Optional[str] = None) -> str:
        """Plot and save learning curve"""

        if not self.learning_curve_data:
            logging.warning("No learning curve data available")
            return ""

        plt.figure(figsize=(10, 6))

        data = self.learning_curve_data
        train_sizes = data["train_sizes"]
        train_scores = data["train_scores"]
        val_scores = data["val_scores"]
        train_std = data.get("train_scores_std", [0] * len(train_sizes))
        val_std = data.get("val_scores_std", [0] * len(train_sizes))

        # Plot learning curves
        plt.plot(train_sizes, train_scores, "o-", color="blue", label="Training Score")
        plt.fill_between(
            train_sizes,
            np.array(train_scores) - np.array(train_std),
            np.array(train_scores) + np.array(train_std),
            alpha=0.1,
            color="blue",
        )

        plt.plot(train_sizes, val_scores, "o-", color="red", label="Validation Score")
        plt.fill_between(
            train_sizes,
            np.array(val_scores) - np.array(val_std),
            np.array(val_scores) + np.array(val_std),
            alpha=0.1,
            color="red",
        )

        plt.xlabel("Training Set Size")
        plt.ylabel("Accuracy Score")
        plt.title(f"Learning Curve - {self.__class__.__name__}")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""

    def plot_training_history(self, save_path: Optional[str] = None) -> str:
        """Plot training history for neural networks"""
        if not self.training_history:
            logging.warning("No training history available")
            return ""

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        if "accuracy" in self.training_history:
            axes[0].plot(self.training_history["accuracy"], label="Training Accuracy")
            if "val_accuracy" in self.training_history:
                axes[0].plot(
                    self.training_history["val_accuracy"], label="Validation Accuracy"
                )
            axes[0].set_title("Model Accuracy")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # Plot loss
        if "loss" in self.training_history:
            axes[1].plot(self.training_history["loss"], label="Training Loss")
            if "val_loss" in self.training_history:
                axes[1].plot(self.training_history["val_loss"], label="Validation Loss")
            axes[1].set_title("Model Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            return save_path
        else:
            plt.show()
            return ""
