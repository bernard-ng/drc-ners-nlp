from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drc_names_classifier.config import ExperimentConfig
    from drc_names_classifier.models.base import ExperimentModel


class ModelFamily(StrEnum):
    LINEAR = "linear"
    PROBABILISTIC = "probabilistic"
    TREE = "tree"
    BOOSTING = "boosting"
    NEURAL = "neural"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Import path and dependency status for one configured model."""

    name: str
    module: str
    class_name: str
    family: ModelFamily
    dependency: str | None = None

    @property
    def available(self) -> bool:
        return self.availability_error is None

    @property
    def availability_error(self) -> str | None:
        if self.dependency is None:
            return None
        return _dependency_error(self.dependency)


class ModelRegistry(Mapping[str, ModelSpec]):
    """Read-only registry that lazily loads optional model dependencies."""

    def __init__(self, specs: tuple[ModelSpec, ...]) -> None:
        duplicates = {
            spec.name for spec in specs if sum(s.name == spec.name for s in specs) > 1
        }
        if duplicates:
            raise ValueError(f"Duplicate model registrations: {sorted(duplicates)}")
        self._specs = MappingProxyType({spec.name: spec for spec in specs})

    def __getitem__(self, name: str) -> ModelSpec:
        return self._specs[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def model_class(self, name: str) -> type[ExperimentModel]:
        try:
            spec = self[name]
        except KeyError as error:
            available = ", ".join(self._specs)
            raise ValueError(f"Unknown model type '{name}'. Available: {available}") from error
        if not spec.available:
            raise RuntimeError(f"Model '{name}' is unavailable: {spec.availability_error}")
        module = import_module(spec.module)
        return getattr(module, spec.class_name)

    def create(self, config: ExperimentConfig) -> ExperimentModel:
        return self.model_class(config.model_type)(config)

    def names(self, *, available_only: bool = False) -> tuple[str, ...]:
        if not available_only:
            return tuple(self._specs)
        return tuple(name for name, spec in self._specs.items() if spec.available)


@lru_cache(maxsize=None)
def _dependency_error(dependency: str) -> str | None:
    if find_spec(dependency) is None:
        return f"Python package '{dependency}' is not installed"
    try:
        import_module(dependency)
    except Exception as error:
        details = next(
            (line.strip() for line in str(error).splitlines() if line.strip()),
            "dependency could not be imported",
        )
        return f"{type(error).__name__}: {details}"
    return None


MODEL_REGISTRY = ModelRegistry(
    (
        ModelSpec(
            "bigru",
            "drc_names_classifier.models.bigru",
            "BiGRUModel",
            ModelFamily.NEURAL,
            "tensorflow",
        ),
        ModelSpec(
            "cnn",
            "drc_names_classifier.models.cnn",
            "CNNModel",
            ModelFamily.NEURAL,
            "tensorflow",
        ),
        ModelSpec(
            "dummy",
            "drc_names_classifier.models.dummy",
            "DummyBaselineModel",
            ModelFamily.PROBABILISTIC,
        ),
        ModelSpec(
            "ensemble",
            "drc_names_classifier.models.ensemble",
            "EnsembleModel",
            ModelFamily.ENSEMBLE,
        ),
        ModelSpec(
            "lightgbm",
            "drc_names_classifier.models.lightgbm",
            "LightGBMModel",
            ModelFamily.BOOSTING,
            "lightgbm",
        ),
        ModelSpec(
            "logistic_regression",
            "drc_names_classifier.models.logistic_regression",
            "LogisticRegressionModel",
            ModelFamily.LINEAR,
        ),
        ModelSpec(
            "lstm",
            "drc_names_classifier.models.lstm",
            "LSTMModel",
            ModelFamily.NEURAL,
            "tensorflow",
        ),
        ModelSpec(
            "naive_bayes",
            "drc_names_classifier.models.naive_bayes",
            "NaiveBayesModel",
            ModelFamily.PROBABILISTIC,
        ),
        ModelSpec(
            "position_logistic_regression",
            "drc_names_classifier.models.position_logistic_regression",
            "PositionAwareLogisticRegressionModel",
            ModelFamily.LINEAR,
        ),
        ModelSpec(
            "random_forest",
            "drc_names_classifier.models.random_forest",
            "RandomForestModel",
            ModelFamily.TREE,
        ),
        ModelSpec(
            "transformer",
            "drc_names_classifier.models.transformer",
            "TransformerModel",
            ModelFamily.NEURAL,
            "tensorflow",
        ),
        ModelSpec(
            "xgboost",
            "drc_names_classifier.models.xgboost",
            "XGBoostModel",
            ModelFamily.BOOSTING,
            "xgboost",
        ),
    )
)
