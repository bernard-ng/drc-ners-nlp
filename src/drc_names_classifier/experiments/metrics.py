from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Calculate the study's consistently defined classification metrics."""

    requested = set(metrics or ("accuracy", "precision", "recall", "f1"))
    results: dict[str, float] = {}

    if "accuracy" in requested:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "balanced_accuracy" in requested:
        results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    weighted = {"precision", "recall", "f1"} & requested
    if weighted:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
        values = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        results.update({name: values[name] for name in weighted})

    if "macro_f1" in requested:
        _, _, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,  # pyright: ignore[reportArgumentType]
        )
        results["macro_f1"] = float(macro_f1)

    if "mcc" in requested:
        results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    return results
