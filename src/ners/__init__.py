"""Public API for CongoNames sex model training."""

from ners.dataset import DatasetBatch, DatasetPartition, DatasetSchemaError, NameDataset
from ners.model import NameSexClassifier
from ners.training import EvaluationResult, evaluate_model, train_model

__all__ = [
    "DatasetBatch",
    "DatasetPartition",
    "DatasetSchemaError",
    "EvaluationResult",
    "NameDataset",
    "NameSexClassifier",
    "evaluate_model",
    "train_model",
]
__version__ = "0.2.0"
