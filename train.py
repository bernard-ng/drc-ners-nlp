#!.venv/bin/python3
import argparse
import logging
import sys
import traceback

import yaml

from core.config import setup_config
from research.model_trainer import ModelTrainer


def load_research_templates(templates_path: str = "config/research_templates.yaml") -> dict:
    """Load research templates from YAML file"""
    try:
        with open(templates_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Templates file not found: {templates_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing templates file: {e}")
        raise


def find_experiment_config(templates: dict, name: str, experiment_type: str) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description="Train DRC Names Models using Research Templates")
    parser.add_argument("--name", type=str, required=True, help="Model name to train")
    parser.add_argument("--type", type=str, required=True, help="Experiment type")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="development", help="Environment name")
    parser.add_argument("--templates", type=str, default="config/research_templates.yaml")
    args = parser.parse_args()

    try:
        # Setup pipeline configuration
        config = setup_config(config_path=args.config, env=args.env)

        # Load research templates
        logging.info(f"Loading research templates from: {args.templates}")
        templates = load_research_templates(args.templates)

        # Find the specific experiment configuration
        logging.info(f"Looking for experiment: name='{args.name}', type='{args.type}'")
        experiment_config = find_experiment_config(templates, args.name, args.type)

        logging.info(f"Found experiment: {experiment_config.get('name')}")
        logging.info(f"Description: {experiment_config.get('description')}")
        logging.info(f"Features: {experiment_config.get('features')}")

        # Train the model using template configuration
        trainer = ModelTrainer(config)
        trainer.train_single_model(
            model_name=experiment_config.get("name"),
            model_type=experiment_config.get("model_type"),
            features=experiment_config.get("features"),
            model_params=experiment_config.get("model_params", {}),
            tags=experiment_config.get("tags", []),
        )

        logging.info("Training completed successfully!")
        return 0

    except Exception as e:
        logging.error(f"Training failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
