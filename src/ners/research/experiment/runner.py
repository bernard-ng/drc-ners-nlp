from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
from sklearn.metrics import confusion_matrix

from ners.config import ExperimentConfig, ResearchConfig
from ners.research.data import ExperimentDatasetStore
from ners.research.experiment.metrics import calculate_metrics
from ners.research.experiment.result import ExperimentStatus
from ners.research.experiment.tracker import ExperimentTracker
from ners.research.model_registry import MODEL_REGISTRY, ModelRegistry
from ners.research.models.base import ResearchModel


class ExperimentRunner:
    """Run configured models against shared deterministic full-name splits."""

    def __init__(
        self,
        config: ResearchConfig,
        *,
        registry: ModelRegistry = MODEL_REGISTRY,
        tracker: ExperimentTracker | None = None,
        datasets: ExperimentDatasetStore | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.tracker = tracker or ExperimentTracker(config)
        self.datasets = datasets or ExperimentDatasetStore(config)

    def run(self, config: ExperimentConfig) -> str:
        experiment_id = self.tracker.create(config)
        self.tracker.update(experiment_id, status=ExperimentStatus.RUNNING)

        try:
            split = self.datasets.load(
                sample_fraction=config.sample_fraction,
                test_fraction=config.test_fraction,
                required_token_count=config.required_token_count,
                split_group_view=config.split_group_view,
            )
            X_train = split.train.drop("sex")
            y_train = split.train.get_column("sex")
            X_test = split.test.drop("sex")
            y_test = split.test.get_column("sex")

            model = self.registry.create(config)
            model.fit(X_train, y_train)
            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)
            y_train_array = y_train.to_numpy()
            y_test_array = y_test.to_numpy()

            train_metrics = calculate_metrics(y_train_array, train_predictions, config.metrics)
            test_metrics = calculate_metrics(y_test_array, test_predictions, config.metrics)
            cv_metrics: dict[str, float] = {}
            if config.cross_validation_folds > 1:
                cv_metrics = {
                    key: float(value)
                    for key, value in model.cross_validate(
                        X_train, y_train, config.cross_validation_folds
                    ).items()
                }

            model_path = self._save_model(model, experiment_id)
            class_distribution = {
                str(label): int(count)
                for label, count in split.train.group_by("sex").len().iter_rows()
            }
            self.tracker.update(
                experiment_id,
                status=ExperimentStatus.COMPLETED,
                end_time=datetime.now(),
                model_path=str(model_path),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                cv_metrics=cv_metrics,
                confusion_matrix=confusion_matrix(
                    y_test_array, test_predictions, labels=["f", "m"]
                ).tolist(),
                feature_importance=model.get_feature_importance(),
                train_size=X_train.height,
                test_size=X_test.height,
                class_distribution=class_distribution,
            )
            logging.info(
                "Experiment %s completed with test macro-F1 %.4f",
                experiment_id,
                test_metrics.get("macro_f1", test_metrics.get("f1", 0.0)),
            )
            return experiment_id
        except Exception as error:
            self.tracker.update(
                experiment_id,
                status=ExperimentStatus.FAILED,
                end_time=datetime.now(),
                error_message=str(error),
            )
            raise

    def run_batch(self, experiments: list[ExperimentConfig]) -> list[str]:
        experiment_ids: list[str] = []
        for config in experiments:
            try:
                experiment_ids.append(self.run(config))
            except Exception:
                logging.exception("Experiment failed: %s", config.name)
        return experiment_ids

    def load_model(self, experiment_id: str) -> ResearchModel | None:
        experiment = self.tracker.get(experiment_id)
        if experiment is None or experiment.model_path is None:
            return None
        model_class = self.registry.model_class(experiment.config.model_type)
        return model_class.load(experiment.model_path)

    def compare(self, experiment_ids: list[str], metric: str = "macro_f1") -> pl.DataFrame:
        comparison = self.tracker.compare(experiment_ids)
        column = f"test_{metric}"
        if column in comparison.columns:
            return comparison.sort(column, descending=True)
        return comparison

    def _save_model(self, model: ResearchModel, experiment_id: str) -> Path:
        model_dir = self.config.experiment_models_dir / experiment_id
        model_dir.mkdir(parents=True, exist_ok=True)
        return Path(model.save(str(model_dir / "model.joblib")))
