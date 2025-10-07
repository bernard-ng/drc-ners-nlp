from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .feature_extractor import FeatureType


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""

    # Experiment metadata
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Model configuration
    model_type: str = (
        "logistic_regression"  # logistic_regression, lstm, transformer, etc.
    )
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Feature configuration
    features: List[FeatureType] = field(default_factory=lambda: [FeatureType.FULL_NAME])
    feature_params: Dict[str, Any] = field(default_factory=dict)

    # Data configuration
    train_data_filter: Optional[Dict[str, Any]] = (
        None  # Filter criteria for training data
    )
    test_data_filter: Optional[Dict[str, Any]] = None
    target_column: str = "sex"

    # Training configuration
    test_size: float = 0.2
    random_seed: int = 42
    cross_validation_folds: int = 5

    # Evaluation configuration
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1"]
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert enums to strings
        result["features"] = [f.value for f in self.features]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary"""
        if "features" in data:
            data["features"] = [FeatureType(f) for f in data["features"]]
        return cls(**data)


class ExperimentStatus(Enum):
    """Experiment execution status"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Calculate specified metrics"""

    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]

    results = {}

    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(y_true, y_pred)

    if any(m in metrics for m in ["precision", "recall", "f1"]):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        if "precision" in metrics:
            results["precision"] = precision
        if "recall" in metrics:
            results["recall"] = recall
        if "f1" in metrics:
            results["f1"] = f1

    return results
