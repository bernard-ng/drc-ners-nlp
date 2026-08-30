from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any

import yaml

from ners.config import ExperimentConfig, ResearchConfig
from ners.utils import NameView


_SECTIONS = {
    "baseline": "baseline_experiments",
    "advanced": "advanced_experiments",
    "feature_study": "feature_studies",
    "tuning": "hyperparameter_tuning",
}


class ExperimentBuilder:
    """Load and validate reproducible model definitions."""

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config

    def load_templates(self, templates: str | Path | None = None) -> dict[str, Any]:
        path = Path(templates) if templates is not None else self.config.templates_path
        if not path.is_absolute() and not path.is_file():
            path = self.config.templates_path.parent / path
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Template file must contain a mapping: {path}")
        return loaded

    def build(
        self,
        name: str,
        *,
        experiment_type: str = "baseline",
        sample_fraction: float | None = None,
        test_fraction: float | None = None,
    ) -> ExperimentConfig:
        """Build one experiment from the configured template collection."""

        template = self.find_template(self.load_templates(), name, experiment_type)
        return self.from_template(
            template,
            sample_fraction=(
                self.config.sample_fraction if sample_fraction is None else sample_fraction
            ),
            test_fraction=(
                self.config.test_fraction if test_fraction is None else test_fraction
            ),
        )

    @staticmethod
    def find_template(
        templates: dict[str, Any],
        name: str,
        experiment_type: str = "baseline",
    ) -> dict[str, Any]:
        section_name = _SECTIONS.get(experiment_type)
        if section_name is None:
            available_types = ", ".join(_SECTIONS)
            raise ValueError(
                f"Unknown experiment type '{experiment_type}'. Available: {available_types}"
            )

        experiments = templates.get(section_name, [])
        for experiment in experiments:
            if experiment.get("name") == name:
                return experiment
        available = [experiment.get("name", "unknown") for experiment in experiments]
        raise ValueError(f"Experiment '{name}' not found. Available: {available}")

    def templates(self, experiment_type: str = "baseline") -> list[dict[str, Any]]:
        section_name = _SECTIONS.get(experiment_type)
        if section_name is None:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        values = self.load_templates().get(section_name, [])
        if not isinstance(values, list):
            raise ValueError(f"Template section '{section_name}' must be a list")
        return values

    def name_view_pair(
        self,
        template: dict[str, Any],
        *,
        sample_fraction: float,
        test_fraction: float,
    ) -> tuple[ExperimentConfig, ExperimentConfig]:
        """Build the controlled three-token full/native ablation pair."""

        base = self.from_template(
            template,
            sample_fraction=sample_fraction,
            test_fraction=test_fraction,
        )
        shared = {
            "split_group_view": NameView.NATIVE_ONLY.value,
            "required_token_count": 3,
        }
        return (
            replace(
                base,
                name=f"{base.name}__full",
                input_view=NameView.FULL.value,
                tags=(*base.tags, "surname_ablation", "surname_included"),
                **shared,
            ),
            replace(
                base,
                name=f"{base.name}__native_only",
                input_view=NameView.NATIVE_ONLY.value,
                tags=(*base.tags, "surname_ablation", "native_only"),
                **shared,
            ),
        )

    @staticmethod
    def from_template(
        template: dict[str, Any],
        *,
        sample_fraction: float,
        test_fraction: float,
    ) -> ExperimentConfig:
        features = template.get("features", ["full_name"])
        if features != ["full_name"]:
            raise ValueError("Research templates may use only the published full name")

        return ExperimentConfig(
            name=str(template.get("name", template.get("model_type", "experiment"))),
            description=str(template.get("description", "")),
            model_type=str(template.get("model_type", "logistic_regression")),
            model_params=dict(template.get("model_params", {})),
            tags=tuple(template.get("tags", [])),
            test_fraction=test_fraction,
            sample_fraction=sample_fraction,
            cross_validation_folds=int(template.get("cross_validation_folds", 0)),
            input_view=str(template.get("input_view", NameView.FULL.value)),
            split_group_view=str(template.get("split_group_view", NameView.FULL.value)),
            required_token_count=template.get("required_token_count"),
        )
