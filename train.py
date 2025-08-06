#!.venv/bin/python3
import argparse
import sys

from core.config import setup_config_and_logging
from research.model_trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train DRC Names Models")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--env", type=str, default="development",
        help="Environment name (default: development)"
    )
    parser.add_argument("--type", type=str, help="Specific model type to train")
    parser.add_argument("--name", type=str, help="Model name")

    args = parser.parse_args()

    try:
        # Load configuration and setup logging
        config = setup_config_and_logging(config_path=args.config, env=args.env)

        trainer = ModelTrainer()

        # Train specific model
        trainer.train_single_model(
            model_name=args.name,
            model_type=args.type,
            features=["full_name"]
        )

        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
