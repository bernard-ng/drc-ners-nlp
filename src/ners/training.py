from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ners.config import TrainingConfig
from ners.dataset import LABELS, NameDataset
from ners.model import NameSexClassifier
from ners.utils import write_json


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    rows: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    log_loss: float
    labels: tuple[str, ...]
    confusion_matrix: list[list[int]]
    class_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dataset(config: TrainingConfig) -> NameDataset:
    return NameDataset(
        config.dataset_path,
        chunk_size=config.chunk_size,
        test_fraction=config.test_fraction,
        sample_fraction=config.sample_fraction,
        name_column=config.name_column,
        target_column=config.target_column,
    )


def train_model(config: TrainingConfig) -> EvaluationResult:
    """Train from names.csv in bounded memory and persist the model and metrics."""

    dataset = build_dataset(config)
    class_weight, train_counts = _resolve_class_weights(dataset, config.balanced)
    model = NameSexClassifier(
        n_features=config.n_features,
        ngram_range=(config.ngram_min, config.ngram_max),
        alpha=config.alpha,
        random_seed=config.random_seed,
        class_weight=class_weight,
    )

    trained_rows = 0
    batches = 0
    for epoch in range(config.epochs):
        logging.info("Training epoch %d/%d", epoch + 1, config.epochs)
        epoch_rows = 0
        for batch in dataset.iter_split("train"):
            if epoch == 0 and not config.balanced:
                labels, frequencies = np.unique(batch.labels, return_counts=True)
                for label, frequency in zip(labels, frequencies, strict=True):
                    train_counts[str(label)] += int(frequency)
            model.partial_fit(batch)
            epoch_rows += len(batch)
            batches += 1
        if epoch_rows == 0:
            raise ValueError("The configured split contains no training rows")
        trained_rows = epoch_rows
        logging.info("Completed epoch %d with %d rows", epoch + 1, epoch_rows)

    evaluation = evaluate_model(model, dataset)
    metadata = {
        "target": "sex",
        "labels": list(LABELS),
        "created_at": datetime.now(UTC).isoformat(),
        "training_rows": trained_rows,
        "training_batches": batches,
        "training_class_counts": train_counts,
        "training_config": config.to_dict(),
        "evaluation": evaluation.to_dict(),
        "responsible_use": (
            "In this project, sex means the f or m marker in the source records, not "
            "gender identity. Do not use this model for profiling or consequential "
            "decisions about individuals."
        ),
    }
    model.metadata = metadata
    model.save(config.model_path)
    write_json(config.resolved_metrics_path, metadata)
    return evaluation


def evaluate_model(model: NameSexClassifier, dataset: NameDataset) -> EvaluationResult:
    """Evaluate a fitted model against the deterministic held-out name groups."""

    labels = model.classes
    label_to_index = {label: index for index, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    class_counts = {label: 0 for label in labels}
    negative_log_likelihood = 0.0
    rows = 0

    for batch in dataset.iter_split("test"):
        predictions = model.predict(batch.names)
        probabilities = model.predict_proba(batch.names)
        true_indices = np.fromiter(
            (label_to_index[str(label)] for label in batch.labels),
            dtype=np.int64,
            count=len(batch),
        )
        predicted_indices = np.fromiter(
            (label_to_index[str(label)] for label in predictions),
            dtype=np.int64,
            count=len(batch),
        )
        np.add.at(confusion, (true_indices, predicted_indices), 1)
        chosen_probabilities = probabilities[np.arange(len(batch)), true_indices]
        negative_log_likelihood -= float(
            np.log(np.clip(chosen_probabilities, 1e-15, 1.0)).sum()
        )

        for label, count in zip(*np.unique(batch.labels, return_counts=True), strict=True):
            class_counts[str(label)] += int(count)
        rows += len(batch)

    if rows == 0:
        raise ValueError("The configured split contains no test rows")

    true_totals = confusion.sum(axis=1)
    predicted_totals = confusion.sum(axis=0)
    correct = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        correct,
        predicted_totals,
        out=np.zeros_like(correct),
        where=predicted_totals != 0,
    )
    recall = np.divide(
        correct,
        true_totals,
        out=np.zeros_like(correct),
        where=true_totals != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(correct),
        where=(precision + recall) != 0,
    )

    return EvaluationResult(
        rows=rows,
        accuracy=float(correct.sum() / rows),
        balanced_accuracy=float(recall.mean()),
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        log_loss=negative_log_likelihood / rows,
        labels=labels,
        confusion_matrix=confusion.tolist(),
        class_counts=class_counts,
    )


def _resolve_class_weights(
    dataset: NameDataset, balanced: bool
) -> tuple[dict[str, float] | None, dict[str, int]]:
    if not balanced:
        return None, {label: 0 for label in LABELS}

    counts = dataset.count_labels("train")
    if any(count == 0 for count in counts.values()):
        raise ValueError("Balanced training requires both labels in the training split")

    total = sum(counts.values())
    weights = {label: total / (len(LABELS) * count) for label, count in counts.items()}
    return weights, counts
