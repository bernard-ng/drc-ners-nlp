from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

import polars as pl

from ners.config import ExperimentConfig, ResearchConfig
from ners.research.experiment.result import ExperimentResult, ExperimentStatus
from ners.utils import read_json, write_json


class ExperimentTracker:
    """Persist, query, compare, and export experiment results."""

    def __init__(self, config: ResearchConfig | None = None) -> None:
        self.config = config or ResearchConfig()
        self.experiments_dir = self.config.experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.experiments_dir / "experiments.json"
        self._results: dict[str, ExperimentResult] = {}
        self._load_results()

    def create(self, config: ExperimentConfig) -> str:
        config_hash = hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        experiment_id = f"{config.name}_{timestamp}_{config_hash}"
        self._results[experiment_id] = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            start_time=datetime.now(),
        )
        self._save_results()
        return experiment_id

    def update(self, experiment_id: str, **updates: Any) -> ExperimentResult:
        try:
            result = self._results[experiment_id]
        except KeyError as error:
            raise KeyError(f"Unknown experiment: {experiment_id}") from error

        for key, value in updates.items():
            if not hasattr(result, key):
                raise ValueError(f"Unknown experiment result field: {key}")
            setattr(result, key, value)
        self._save_results()
        return result

    def get(self, experiment_id: str) -> ExperimentResult | None:
        return self._results.get(experiment_id)

    def list(
        self,
        *,
        status: ExperimentStatus | None = None,
        tags: tuple[str, ...] | list[str] | None = None,
        model_type: str | None = None,
    ) -> list[ExperimentResult]:
        results = list(self._results.values())
        if status is not None:
            results = [result for result in results if result.status == status]
        if tags:
            results = [
                result for result in results if any(tag in result.config.tags for tag in tags)
            ]
        if model_type is not None:
            results = [result for result in results if result.config.model_type == model_type]
        return sorted(results, key=lambda result: result.start_time, reverse=True)

    def best(
        self,
        metric: str = "macro_f1",
        *,
        dataset: str = "test",
        model_type: str | None = None,
    ) -> ExperimentResult | None:
        candidates = self.list(status=ExperimentStatus.COMPLETED, model_type=model_type)
        scored = []
        for result in candidates:
            values = result.test_metrics if dataset == "test" else result.train_metrics
            if metric in values:
                scored.append((result, values[metric]))
        return max(scored, key=lambda item: item[1])[0] if scored else None

    def compare(self, experiment_ids: list[str]) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for experiment_id in experiment_ids:
            result = self.get(experiment_id)
            if result is None:
                continue
            row: dict[str, Any] = {
                "experiment_id": experiment_id,
                "name": result.config.name,
                "model_type": result.config.model_type,
                "input_view": result.config.input_view,
                "split_group_view": result.config.split_group_view,
                "required_token_count": result.config.required_token_count,
                "status": result.status.value,
                "train_size": result.train_size,
                "test_size": result.test_size,
            }
            row.update({f"test_{key}": value for key, value in result.test_metrics.items()})
            row.update({f"cv_{key}": value for key, value in result.cv_metrics.items()})
            rows.append(row)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def _load_results(self) -> None:
        if not self.results_path.exists():
            return
        try:
            payload = read_json(self.results_path)
            if not isinstance(payload, dict):
                raise ValueError("experiment database must contain an object")
            self._results = {
                experiment_id: ExperimentResult.from_dict(value)
                for experiment_id, value in payload.items()
            }
        except (OSError, TypeError, ValueError) as error:
            logging.warning("Could not load experiment results: %s", error)

    def _save_results(self) -> None:
        write_json(
            self.results_path,
            {key: value.to_dict() for key, value in self._results.items()},
        )
