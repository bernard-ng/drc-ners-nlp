import argparse
import logging
from dataclasses import dataclass
from typing import Optional

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format=">> %(message)s")


def evaluate_proba(y_true, y_proba, threshold, class_names):
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)

    logging.info(f"Accuracy: {acc:.4f} | Precision: {pr:.4f} | Recall: {rc:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))


@dataclass
class BaseConfig:
    """
    Represents the base configuration for a dataset and its associated parameters.

    This class serves as a foundational configuration handler to encapsulate
    dataset-related parameters and options. It allows customization of dataset
    behavior, including threshold values, size, cross-validation settings, and
    whether to save derived configurations. It can also manage configurations
    for balanced datasets if necessary.
    """
    dataset_path: str = "names_featured.csv"
    size: Optional[int] = None
    threshold: float = 0.5
    cv: Optional[int] = None
    save: bool = False
    balanced: bool = False

    epochs: int = 10
    test_size: float = 0.2
    random_state: int = 42


def load_config(description: str) -> BaseConfig:
    """
    Parses command-line arguments and loads the configuration for the logistic regression model.

    This function sets up an argument parser for various command-line options including
    the dataset path, dataset size, dataset balancing, classification threshold,
    cross-validation folds, and saving the model and its associated artifacts. Once parsed,
    it transfers the configurations to a ``BaseConfig`` instance and returns it.
    """
    parser = argparse.ArgumentParser(description)

    parser.add_argument("--dataset", type=str, default="names_featured.csv", help="Path to the dataset file")
    parser.add_argument("--size", type=int, help="Number of rows to load from the dataset")
    parser.add_argument("--balanced", action="store_true", help="Load balanced dataset")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for classification")
    parser.add_argument("--cv", type=int, help="Number of folds for cross-validation")
    parser.add_argument("--save", action="store_true", help="Save the model and artifacts after training")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    return BaseConfig(
        dataset_path=args.dataset,
        size=args.size,
        threshold=args.threshold,
        cv=args.cv,
        save=args.save,
        balanced=args.balanced,
        epochs=args.epochs,
        test_size=args.test_size,
        random_state=args.random_state
    )
