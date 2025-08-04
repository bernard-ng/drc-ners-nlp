import logging
from contextlib import contextmanager
from pathlib import Path

from core.config import get_config, PipelineConfig


@contextmanager
def temporary_config_override(**overrides):
    """Context manager for temporarily overriding configuration"""
    config = get_config()
    original_values = {}

    # Store original values and apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            original_values[key] = getattr(config, key)
            setattr(config, key, value)

    try:
        yield config
    finally:
        # Restore original values
        for key, value in original_values.items():
            setattr(config, key, value)


def ensure_directories(config: PipelineConfig) -> None:
    """Ensure all required directories exist"""
    directories = [
        config.paths.data_dir,
        config.paths.models_dir,
        config.paths.outputs_dir,
        config.paths.logs_dir,
        config.paths.configs_dir,
        config.paths.checkpoints_dir,
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logging.info("Ensured all required directories exist")


def get_data_file_path(filename: str, config: PipelineConfig) -> Path:
    """Get full path for a data file"""
    return config.paths.data_dir / filename


def get_model_file_path(filename: str, config: PipelineConfig) -> Path:
    """Get full path for a model file"""
    return config.paths.models_dir / filename


def get_output_file_path(filename: str, config: PipelineConfig) -> Path:
    """Get full path for an output file"""
    return config.paths.outputs_dir / filename
