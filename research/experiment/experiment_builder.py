import logging
from typing import List, Dict

import yaml

from core.config.pipeline_config import PipelineConfig
from research.experiment import ExperimentConfig
from research.experiment.feature_extractor import FeatureType


class ExperimentBuilder:
    """Helper class to build experiment configurations"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_templates(self, templates: str = "research_templates.yaml") -> dict:
        """Load research templates from YAML file"""
        try:
            with open(self.config.paths.configs_dir / templates, "r") as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Templates file not found: {templates}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing templates file: {e}")
            raise

    @classmethod
    def find_template(cls, templates: dict, name: str, experiment_type: str = "baseline") -> dict:
        """Find experiment configuration by name and type"""

        # Map type to section in templates
        type_mapping = {
            "baseline": "baseline_experiments",
            "advanced": "advanced_experiments",
            "feature_study": "feature_studies",
            "tuning": "hyperparameter_tuning",
        }

        section_name = type_mapping.get(experiment_type)
        if not section_name:
            available_types = list(type_mapping.keys())
            raise ValueError(
                f"Unknown experiment type '{experiment_type}'. Available types: {available_types}"
            )

        if section_name not in templates:
            raise ValueError(f"Section '{section_name}' not found in templates")

        experiments = templates[section_name]

        # Search for experiment by model name
        for experiment in experiments:
            # Check if this is the experiment we're looking for
            # Look for experiments that match the model type or contain the name
            if (
                experiment.get("model_type") == name
                or name.lower() in experiment.get("name", "").lower()
                or experiment.get("name") == name
                or f"baseline_{name}" == experiment.get("name")
                or f"advanced_{name}" == experiment.get("name")
            ):
                return experiment

        # If not found, list available experiments
        available_experiments = [
            exp.get("name", exp.get("model_type", "unknown")) for exp in experiments
        ]
        raise ValueError(
            f"Experiment '{name}' not found in '{experiment_type}' section. "
            f"Available experiments: {available_experiments}"
        )

    def get_templates(
        self, templates_path: str = "research_templates.yaml"
    ) -> Dict[str, List[Dict]]:
        """Get all available experiments from templates organized by type"""
        templates = self.load_templates(templates_path)

        return {
            "baseline": templates.get("baseline_experiments", []),
            "advanced": templates.get("advanced_experiments", []),
            "feature_study": templates.get("feature_studies", []),
            "tuning": templates.get("hyperparameter_tuning", []),
        }

    @classmethod
    def from_template(cls, template_config: dict) -> ExperimentConfig:
        """Create an ExperimentConfig from a template configuration"""
        # Convert feature strings to FeatureType objects
        features = []
        for feature_str in template_config.get("features", []):
            try:
                features.append(FeatureType(feature_str))
            except ValueError:
                logging.warning(f"Unknown feature type: {feature_str}")
                continue

        return ExperimentConfig(
            name=template_config.get("name"),
            description=template_config.get("description"),
            model_type=template_config.get("model_type"),
            features=features,
            model_params=template_config.get("model_params", {}),
            tags=template_config.get("tags", []),
            test_size=template_config.get("test_size", 0.2),
            cross_validation_folds=template_config.get("cross_validation_folds", 5),
            train_data_filter=template_config.get("train_data_filter"),
        )
