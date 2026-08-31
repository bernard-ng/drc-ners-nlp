"""Central configuration API for the model experiments."""

from drc_names_classifier.config.defaults import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METRICS,
    DEFAULT_EXPERIMENT_MODELS_DIR,
    DEFAULT_EXPERIMENT_OUTPUTS_DIR,
    DEFAULT_EXPERIMENT_TEMPLATES_PATH,
)
from drc_names_classifier.config.experiment import ExperimentConfig
from drc_names_classifier.config.experiment_settings import ExperimentSettings

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_METRICS",
    "DEFAULT_EXPERIMENT_MODELS_DIR",
    "DEFAULT_EXPERIMENT_OUTPUTS_DIR",
    "DEFAULT_EXPERIMENT_TEMPLATES_PATH",
    "ExperimentConfig",
    "ExperimentSettings",
]
