"""Public registry and shared contract for experiment models."""

from drc_names_classifier.models.base import ExperimentModel
from drc_names_classifier.models.registry import (
    MODEL_REGISTRY,
    ModelFamily,
    ModelRegistry,
    ModelSpec,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelFamily",
    "ModelRegistry",
    "ModelSpec",
    "ExperimentModel",
]
