import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

import yaml

from core.config.pipeline_config import PipelineConfig
from core.config.project_paths import ProjectPaths


class ConfigManager:
    """Centralized configuration management"""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[PipelineConfig] = None
        self._setup_default_paths()

    @classmethod
    def _find_config_file(cls) -> Path:
        """Find configuration file in standard locations"""
        possible_paths = [
            Path.cwd() / "config" / "pipeline.yaml",
            Path.cwd() / "config" / "pipeline.yml",
            Path.cwd() / "pipeline.yaml",
            Path(__file__).parent.parent.parent / "config" / "pipeline.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Return default path if none found
        return Path.cwd() / "config" / "pipeline.yaml"

    def _setup_default_paths(self):
        """Setup default project paths"""
        root_dir = Path(__file__).parent.parent.parent
        self.default_paths = ProjectPaths(
            root_dir=root_dir,
            configs_dir=root_dir / "config",
            data_dir=root_dir / "data" / "dataset",
            models_dir=root_dir / "data" / "models",
            outputs_dir=root_dir / "data" / "outputs",
            logs_dir=root_dir / "data" / "logs",
            checkpoints_dir=root_dir / "data" / "checkpoints",
        )

    def load_config(self, config_path: Optional[Path] = None) -> PipelineConfig:
        """Load configuration from file"""
        if config_path:
            self.config_path = config_path

        if not self.config_path.exists():
            logging.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return self._create_default_config()

        try:
            with open(self.config_path, "r") as f:
                if self.config_path.suffix.lower() in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            # Ensure paths are properly set
            if "paths" not in config_data:
                config_data["paths"] = self.default_paths.model_dump()

            self._config = PipelineConfig(**config_data)
            return self._config

        except Exception as e:
            logging.error(f"Failed to load config from {self.config_path}: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> PipelineConfig:
        """Create default configuration"""
        return PipelineConfig(paths=self.default_paths)

    def save_config(self, config: PipelineConfig, path: Optional[Path] = None):
        """Save configuration to file"""
        save_path = path or self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump()

        # Convert Path objects to strings for serialization
        if "paths" in config_dict:
            for key, value in config_dict["paths"].items():
                if isinstance(value, Path):
                    config_dict["paths"][key] = str(value)

        try:
            with open(save_path, "w") as f:
                if save_path.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)

            logging.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logging.error(f"Failed to save config to {save_path}: {e}")

    def get_config(self) -> PipelineConfig:
        """Get current configuration, loading if necessary"""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        config = self.get_config()

        # Deep update configuration
        config_dict = config.model_dump()
        self._deep_update(config_dict, updates)

        self._config = PipelineConfig(**config_dict)

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get_environment_config(self, env: str) -> PipelineConfig:
        """Load environment-specific configuration"""
        env_config_path = self.config_path.parent / f"pipeline.{env}.yaml"

        if env_config_path.exists():
            base_config = self.load_config()
            env_config = self.load_config(env_config_path)

            # Merge configurations
            base_dict = base_config.dict()
            env_dict = env_config.dict()
            self._deep_update(base_dict, env_dict)

            return PipelineConfig(**base_dict)

        return self.get_config()
