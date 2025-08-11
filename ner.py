#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from core.config import setup_config
from processing.ner.ner_data_builder import NERDataBuilder
from processing.ner.ner_name_model import NERNameModel


def train(config_path=None, env="development"):
    """Train the NER model."""
    try:
        config = setup_config(config_path=config_path, env=env)
        trainer = NERNameModel(config)
        builder = NERDataBuilder(config)

        data_path = Path(config.paths.data_dir) / config.data.output_files["ner_data"]
        if not data_path.exists():
            builder.build()

        trainer.create_blank_model("fr")
        data = trainer.load_data(str(data_path))

        split_idx = int(len(data) * 0.8)
        train_data, eval_data = data[:split_idx], data[split_idx:]

        logging.info(f"Training with {len(train_data)} examples, evaluating on {len(eval_data)}")
        trainer.train(train_data, epochs=1, batch_size=config.processing.batch_size, dropout_rate=0.3)
        trainer.evaluate(eval_data)

        model_path = trainer.save()
        logging.info(f"Model saved to: {model_path}")
        return 0

    except Exception as e:
        logging.error(f"NER Training failed: {e}", exc_info=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Train NER model for DRC names")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="development", help="Environment name")
    args = parser.parse_args()

    sys.exit(train(config_path=args.config, env=args.env))


if __name__ == "__main__":
    main()
