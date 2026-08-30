from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ners.config.defaults import DEFAULT_METRICS
from ners.utils import NameView, coerce_name_view


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Definition of one comparable full-name experiment."""

    name: str
    model_type: str = "logistic_regression"
    description: str = ""
    tags: tuple[str, ...] = ()
    model_params: dict[str, Any] = field(default_factory=dict)
    sample_fraction: float = 1.0
    test_fraction: float = 0.2
    random_seed: int = 42
    cross_validation_folds: int = 0
    metrics: tuple[str, ...] = DEFAULT_METRICS
    input_view: str = NameView.FULL
    split_group_view: str = NameView.FULL
    required_token_count: int | None = None

    def __post_init__(self) -> None:
        name = self.name.strip()
        model_type = self.model_type.strip()
        if not name:
            raise ValueError("Experiment name must not be empty")
        if not model_type:
            raise ValueError("Experiment model_type must not be empty")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be between zero and one")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")
        if self.cross_validation_folds == 1 or self.cross_validation_folds < 0:
            raise ValueError("cross_validation_folds must be zero or at least two")
        if not self.metrics:
            raise ValueError("At least one evaluation metric is required")
        if self.required_token_count is not None and self.required_token_count <= 0:
            raise ValueError("required_token_count must be positive when configured")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "input_view", coerce_name_view(self.input_view).value)
        object.__setattr__(
            self,
            "split_group_view",
            coerce_name_view(self.split_group_view).value,
        )
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "model_params", dict(self.model_params))
        object.__setattr__(self, "metrics", tuple(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "description": self.description,
            "tags": list(self.tags),
            "model_params": dict(self.model_params),
            "sample_fraction": self.sample_fraction,
            "test_fraction": self.test_fraction,
            "random_seed": self.random_seed,
            "cross_validation_folds": self.cross_validation_folds,
            "metrics": list(self.metrics),
            "input_view": self.input_view,
            "split_group_view": self.split_group_view,
            "required_token_count": self.required_token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        values = dict(data)
        if "tags" in values:
            values["tags"] = tuple(values["tags"])
        if "metrics" in values:
            values["metrics"] = tuple(values["metrics"])
        return cls(**values)
