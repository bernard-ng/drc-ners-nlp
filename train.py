#!.venv/bin/python3
import argparse
import logging
import sys
import traceback

from core.config import setup_config
from research.model_trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DRC Names Models")
    parser.add_argument("--type", type=str, help="Specific model type to train")
    parser.add_argument("--name", type=str, help="Model name")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="development", help="Environment name")
    args = parser.parse_args()

    try:
        config = setup_config(config_path=args.config, env=args.env)
        trainer = ModelTrainer(config)

        # Train specific model
        trainer.train_single_model(
            model_name=args.name,
            model_type=args.type,
            features=["full_name"]
        )
        return 0

    except Exception as e:
        logging.error(f"Training failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
