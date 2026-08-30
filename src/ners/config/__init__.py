"""Central configuration API for training and comparative research."""

from ners.config.defaults import (
    DEFAULT_DATASET_PATH,
    DEFAULT_METRICS,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESEARCH_MODELS_DIR,
    DEFAULT_RESEARCH_OUTPUTS_DIR,
    DEFAULT_RESEARCH_TEMPLATES_PATH,
)
from ners.config.experiment import ExperimentConfig
from ners.config.research import ResearchConfig
from ners.config.training import TrainingConfig

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_METRICS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_RESEARCH_MODELS_DIR",
    "DEFAULT_RESEARCH_OUTPUTS_DIR",
    "DEFAULT_RESEARCH_TEMPLATES_PATH",
    "ExperimentConfig",
    "ResearchConfig",
    "TrainingConfig",
]
