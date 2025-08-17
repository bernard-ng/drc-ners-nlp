#!.venv/bin/python3
import argparse
import logging
import sys
import traceback

from core.config import setup_config
from research.experiment.experiment_builder import ExperimentBuilder
from research.model_trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DRC Names Models using Research Templates")
    parser.add_argument("--name", type=str, required=True, help="Model name to train")
    parser.add_argument("--type", type=str, required=True, help="Experiment type")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="development", help="Environment name")
    parser.add_argument("--templates", type=str, default="research_templates.yaml")
    args = parser.parse_args()

    try:
        # Setup pipeline configuration
        config = setup_config(config_path=args.config, env=args.env)
        experiment_builder = ExperimentBuilder(config)

        # Load research templates
        logging.info(f"Loading research templates from: {args.templates}")
        templates = experiment_builder.load_templates(args.templates)

        # Find the specific experiment configuration
        logging.info(f"Looking for experiment: name='{args.name}', type='{args.type}'")
        experiment_config = experiment_builder.find_template(templates, args.name, args.type)

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
    sys.exit(main())
