"""Polars tables shared by the Streamlit experiment app."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

import polars as pl

from drc_names_classifier.experiments import ExperimentResult
from drc_names_classifier.models import ModelSpec


_METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
)


def model_availability_frame(registry: Mapping[str, ModelSpec]) -> pl.DataFrame:
    """Create the model/dependency table shown in the experiment overview."""

    rows = [
        {
            "model": name,
            "family": spec.family.value,
            "available": spec.available,
            "dependency": spec.dependency or "included",
            "reason": spec.availability_error,
        }
        for name, spec in registry.items()
    ]
    return pl.DataFrame(
        rows,
        schema={
            "model": pl.String,
            "family": pl.String,
            "available": pl.Boolean,
            "dependency": pl.String,
            "reason": pl.String,
        },
    ).sort("model")


def experiment_results_frame(results: Iterable[ExperimentResult]) -> pl.DataFrame:
    """Flatten tracked results into a consistent comparison table."""

    rows = []
    for result in results:
        duration = _duration_seconds(result.start_time, result.end_time)
        row: dict[str, object] = {
            "experiment_id": result.experiment_id,
            "name": result.config.name,
            "model": result.config.model_type,
            "status": result.status.value,
            "started_at": result.start_time,
            "duration_seconds": duration,
            "sample_fraction": result.config.sample_fraction,
            "train_rows": result.train_size,
            "test_rows": result.test_size,
            "tags": ", ".join(result.config.tags),
        }
        row.update(
            {f"test_{metric}": result.test_metrics.get(metric) for metric in _METRIC_NAMES}
        )
        rows.append(row)

    schema = {
        "experiment_id": pl.String,
        "name": pl.String,
        "model": pl.String,
        "status": pl.String,
        "started_at": pl.Datetime,
        "duration_seconds": pl.Float64,
        "sample_fraction": pl.Float64,
        "train_rows": pl.Int64,
        "test_rows": pl.Int64,
        "tags": pl.String,
        **{f"test_{metric}": pl.Float64 for metric in _METRIC_NAMES},
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort("started_at", descending=True)


def metric_frame(result: ExperimentResult) -> pl.DataFrame:
    """Align train, test, and cross-validation metrics for one result."""

    metric_names = sorted(
        set(result.train_metrics) | set(result.test_metrics) | set(result.cv_metrics)
    )
    return pl.DataFrame(
        [
            {
                "metric": metric,
                "train": result.train_metrics.get(metric),
                "test": result.test_metrics.get(metric),
                "cross_validation": result.cv_metrics.get(metric),
            }
            for metric in metric_names
        ],
        schema={
            "metric": pl.String,
            "train": pl.Float64,
            "test": pl.Float64,
            "cross_validation": pl.Float64,
        },
    )


def confusion_matrix_frame(result: ExperimentResult) -> pl.DataFrame:
    """Return the fixed f/m confusion matrix as a long Polars frame."""

    matrix = result.confusion_matrix
    if matrix is None:
        return pl.DataFrame(
            schema={"actual": pl.String, "predicted": pl.String, "rows": pl.Int64}
        )
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise ValueError("Expected a 2x2 confusion matrix ordered as f, m")
    return pl.DataFrame(
        {
            "actual": ["f", "f", "m", "m"],
            "predicted": ["f", "m", "f", "m"],
            "rows": [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]],
        },
        schema={"actual": pl.String, "predicted": pl.String, "rows": pl.Int64},
    )


def feature_importance_frame(result: ExperimentResult, *, limit: int = 20) -> pl.DataFrame:
    """Sort the estimator's stored feature scores for display."""

    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    values = result.feature_importance or {}
    return (
        pl.DataFrame(
            {
                "feature": list(values),
                "importance": list(values.values()),
            },
            schema={"feature": pl.String, "importance": pl.Float64},
        )
        .sort("importance", descending=True)
        .head(limit)
    )


def _duration_seconds(start: datetime, end: datetime | None) -> float | None:
    if end is None:
        return None
    return max(0.0, (end - start).total_seconds())
