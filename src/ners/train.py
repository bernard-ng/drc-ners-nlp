#!.venv/bin/python3
import logging
import traceback

from ners.core.config import setup_config
from ners.research.experiment.experiment_builder import ExperimentBuilder
from ners.research.model_trainer import ModelTrainer


def train_from_template(
    name: str,
    type: str,
    *,
    templates: str = "research_templates.yaml",
    config: str | None = None,
    env: str = "development",
) -> int:
    try:
        cfg = setup_config(config_path=config, env=env)
        experiment_builder = ExperimentBuilder(cfg)

        logging.info(f"Loading research templates from: {templates}")
        tmpl = experiment_builder.load_templates(templates)

        logging.info(f"Looking for experiment: name='{name}', type='{type}'")
        experiment_config = experiment_builder.find_template(tmpl, name, type)

        logging.info(f"Found experiment: {experiment_config.get('name')}")
        logging.info(f"Description: {experiment_config.get('description')}")
        logging.info(f"Features: {experiment_config.get('features')}")

        trainer = ModelTrainer(cfg)
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
