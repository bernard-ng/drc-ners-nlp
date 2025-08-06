import logging
from pathlib import Path
from typing import Optional, Union

from core.config.config_manager import ConfigManager
from core.config.logging_config import LoggingConfig
from core.config.pipeline_config import PipelineConfig

config_manager = ConfigManager()


def get_config() -> PipelineConfig:
    """Get the global configuration instance"""
    return config_manager.get_config()


def load_config(config_path: Optional[Union[str, Path]] = None) -> PipelineConfig:
    """Load configuration from specified path"""
    if config_path:
        return config_manager.load_config(Path(config_path))
    return config_manager.get_config()


def setup_config(config_path: Optional[Path] = None, env: str = "development") -> PipelineConfig:
    """
    Unified configuration loading and logging setup for all entrypoint scripts.

    Args:
        config_path: Direct path to config file (takes precedence over env)
        env: Environment name (defaults to "development")

    Returns:
        Loaded configuration object
    """
    # Determine config path
    if config_path is None:
        config_path = Path("config") / f"pipeline.{env}.yaml"

    # Load configuration
    config = ConfigManager(config_path).load_config()

    # Setup logging
    setup_logging(config)

    # Ensure required directories exist
    from core.utils import ensure_directories
    ensure_directories(config)

    logging.info(f"Loaded configuration: {config.name} v{config.version}")
    logging.info(f"Environment: {config.environment}")
    logging.info(f"Config file: {config_path}")

    return config


def setup_logging(config: PipelineConfig):
    """Setup logging based on configuration"""

    # Create logs directory
    log_dir = config.paths.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging configuration
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(config.logging.format)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.logging.console_logging:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if config.logging.file_logging:
        from logging.handlers import RotatingFileHandler

        log_file_path = log_dir / config.logging.log_file
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=config.logging.max_log_size,
            backupCount=config.logging.backup_count,
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
