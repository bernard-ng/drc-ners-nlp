import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd

from ners.core.config import PipelineConfig, get_config
from ners.research.experiment import ExperimentConfig, ExperimentStatus
from ners.research.experiment.experiement_result import ExperimentResult


class ExperimentTracker:
    """Tracks and manages experiments"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
        self.experiments_dir = self.config.paths.outputs_dir / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.results_db_path = self.experiments_dir / "experiments.json"
        self._results: Dict[str, ExperimentResult] = {}
        self._load_results()

    def _load_results(self):
        """Load existing experiment results"""
        if self.results_db_path.exists():
            try:
                with open(self.results_db_path, "r") as f:
                    data = json.load(f)

                for exp_id, exp_data in data.items():
                    self._results[exp_id] = ExperimentResult.from_dict(exp_data)
            except Exception as e:
                print(f"Warning: Failed to load experiment results: {e}")

    def _save_results(self):
        """Save experiment results to disk"""
        data = {exp_id: result.to_dict() for exp_id, result in self._results.items()}

        with open(self.results_db_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment and return its ID"""
        # Generate experiment ID
        config_hash = hashlib.md5(
            json.dumps(config.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{config.name}_{timestamp}_{config_hash}"

        # Create result object
        result = ExperimentResult(
            experiment_id=experiment_id, config=config, start_time=datetime.now()
        )

        self._results[experiment_id] = result
        self._save_results()

        return experiment_id

    def update_experiment(self, experiment_id: str, **updates):
        """Update an experiment's results"""
        if experiment_id in self._results:
            result = self._results[experiment_id]

            for key, value in updates.items():
                if hasattr(result, key):
                    setattr(result, key, value)

            self._save_results()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment by ID"""
        return self._results.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        tags: Optional[List[str]] = None,
        model_type: Optional[str] = None,
    ) -> List[ExperimentResult]:
        """List experiments with optional filtering"""
        results = list(self._results.values())

        if status:
            results = [r for r in results if r.status == status]

        if tags:
            results = [r for r in results if any(tag in r.config.tags for tag in tags)]

        if model_type:
            results = [r for r in results if r.config.model_type == model_type]

        return sorted(results, key=lambda x: x.start_time, reverse=True)

    def get_best_experiment(
        self,
        metric: str = "accuracy",
        dataset: str = "test",
        filters: Optional[Dict] = None,
    ) -> Optional[ExperimentResult]:
        """Get the best experiment based on a metric"""
        experiments = self.list_experiments()

        if filters:
            # Apply additional filters
            if "model_type" in filters:
                experiments = [
                    e
                    for e in experiments
                    if e.config.model_type == filters["model_type"]
                ]
            if "features" in filters:
                experiments = [
                    e
                    for e in experiments
                    if any(f in e.config.features for f in filters["features"])
                ]

        valid_experiments = []
        for exp in experiments:
            if exp.status == ExperimentStatus.COMPLETED:
                metrics_dict = (
                    exp.test_metrics if dataset == "test" else exp.train_metrics
                )
                if metric in metrics_dict:
                    valid_experiments.append((exp, metrics_dict[metric]))

        if not valid_experiments:
            return None

        return max(valid_experiments, key=lambda x: x[1])[0]

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments in a DataFrame"""
        rows = []

        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                row = {
                    "experiment_id": exp_id,
                    "name": exp.config.name,
                    "model_type": exp.config.model_type,
                    "features": ",".join([f.value for f in exp.config.features]),
                    "status": exp.status.value,
                    "train_size": exp.train_size,
                    "test_size": exp.test_size,
                }

                # Add metrics
                for metric, value in exp.test_metrics.items():
                    row[f"test_{metric}"] = value

                for metric, value in exp.cv_metrics.items():
                    row[f"cv_{metric}"] = value

                rows.append(row)

        return pd.DataFrame(rows)

    def export_results(self, output_path: Optional[Path] = None) -> Path:
        """Export all results to CSV"""
        if output_path is None:
            output_path = (
                self.experiments_dir
                / f"experiments_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

        rows = []
        for exp in self._results.values():
            row = {
                "experiment_id": exp.experiment_id,
                "name": exp.config.name,
                "description": exp.config.description,
                "model_type": exp.config.model_type,
                "features": ",".join([f.value for f in exp.config.features]),
                "status": exp.status.value,
                "start_time": exp.start_time.isoformat(),
                "end_time": exp.end_time.isoformat() if exp.end_time else None,
                "train_size": exp.train_size,
                "test_size": exp.test_size,
            }

            # Add all metrics
            for metric, value in exp.test_metrics.items():
                row[f"test_{metric}"] = value

            for metric, value in exp.cv_metrics.items():
                row[f"cv_{metric}"] = value

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        return output_path
