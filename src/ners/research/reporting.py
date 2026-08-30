from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ners.research.models.base import ResearchModel


def plot_learning_curve(model: ResearchModel, output_path: str | Path) -> Path:
    """Render persisted learning-curve data without adding plotting to model classes."""

    if not model.learning_curve_data:
        raise ValueError("Model has no learning-curve data")

    data = model.learning_curve_data
    train_sizes = data["train_sizes"]
    train_scores = np.asarray(data["train_scores"])
    validation_scores = np.asarray(data["val_scores"])
    train_std = np.asarray(data.get("train_scores_std", np.zeros(len(train_sizes))))
    validation_std = np.asarray(data.get("val_scores_std", np.zeros(len(train_sizes))))

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(train_sizes, train_scores, "o-", label="Training")
    axis.fill_between(
        train_sizes, train_scores - train_std, train_scores + train_std, alpha=0.1
    )
    axis.plot(train_sizes, validation_scores, "o-", label="Validation")
    axis.fill_between(
        train_sizes,
        validation_scores - validation_std,
        validation_scores + validation_std,
        alpha=0.1,
    )
    axis.set(xlabel="Training set size", ylabel="Accuracy", title=type(model).__name__)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_training_history(model: ResearchModel, output_path: str | Path) -> Path:
    """Render neural training history as accuracy and loss panels."""

    if not model.training_history:
        raise ValueError("Model has no training history")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    history = model.training_history

    axes[0].plot(history.get("accuracy", []), label="Training")
    axes[0].plot(history.get("val_accuracy", []), label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Model accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.get("loss", []), label="Training")
    axes[1].plot(history.get("val_loss", []), label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Model loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return destination
