#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

from core.config import setup_config, PipelineConfig
from processing.ner.name_builder import NameBuilder
from processing.ner.name_engineering import NameEngineering
from processing.ner.name_model import NameModel


def feature(config: PipelineConfig):
    """Apply feature engineering to create position-independent NER dataset."""
    NameEngineering(config).compute()


def build(config: PipelineConfig):
    """Build NER dataset using NERDataBuilder."""
    NameBuilder(config).build()


def train(config: PipelineConfig):
    """Train the NER model."""
    name_model = NameModel(config)

    data_path = Path(config.paths.data_dir) / config.data.output_files["ner_data"]
    if not data_path.exists():
        logging.info("NER data not found. Building dataset first...")
        build(config)

    name_model.create_blank_model("fr")
    data = name_model.load_data(str(data_path))

    split_idx = int(len(data) * 0.9)
    train_data, eval_data = data[:split_idx], data[split_idx:]

    logging.info(f"Training with {len(train_data)} examples, evaluating on {len(eval_data)}")
    name_model.train(
        data=train_data,
        epochs=config.processing.epochs,
        batch_size=config.processing.batch_size,
        dropout_rate=0.3,
    )
    evaluation_results = name_model.evaluate(eval_data)

    model_path = name_model.save()
    logging.info(f"Model saved to: {model_path}")
    print(f"Evaluation results: {evaluation_results}")


def run_pipeline(config: PipelineConfig, reset: bool = False):
    if not reset and os.path.exists(
        config.paths.get_data_path(config.data.output_files["engineered"])
    ):
        logging.info("Step 1: Feature engineering already done.")
    else:
        logging.info("Step 1: Running feature engineering")
        feature(config)

    if not reset and os.path.exists(
        config.paths.get_data_path(config.data.output_files["ner_data"])
    ):
        logging.info("Step 2: NER dataset already built.")
    else:
        logging.info("Step 2: Building NER dataset")
        build(config)

    logging.info("Step 3: Training NER Model")
    train(config)

    return 0


def main():
    parser = argparse.ArgumentParser(description="NER model management for DRC names")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="development", help="Environment name")
    parser.add_argument("--reset", action="store_true", help="Reset all steps")
    args = parser.parse_args()

    try:
        config = setup_config(config_path=args.config, env=args.env)
        return run_pipeline(config, args.reset)

    except Exception as e:
        print(f"Pipeline failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
