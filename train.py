#!.venv/bin/python3
import argparse

from core.config import setup_logging, get_config
from research.model_trainer import ModelTrainer


def main():
    setup_logging(get_config())
    parser = argparse.ArgumentParser(description="Train DRC Names Models")
    parser.add_argument("--type", type=str, help="Specific model type to train")
    parser.add_argument("--name", type=str, help="Model name")

    args = parser.parse_args()
    trainer = ModelTrainer()

    # Train specific model
    trainer.train_single_model(
        model_name=args.name,
        model_type=args.type,
        features=["full_name"]
    )


if __name__ == "__main__":
    main()
