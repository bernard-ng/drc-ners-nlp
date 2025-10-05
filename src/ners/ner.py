#!/usr/bin/env python3
import logging
import os
import traceback
from pathlib import Path

from ners.core.config import PipelineConfig
from ners.processing.ner.name_builder import NameBuilder
from ners.processing.ner.name_engineering import NameEngineering
from ners.processing.ner.name_model import NameModel


def feature(config: PipelineConfig):
    NameEngineering(config).compute()


def build(config: PipelineConfig):
    NameBuilder(config).build()


def train(config: PipelineConfig):
    name_model = NameModel(config)

    data_path = Path(config.paths.data_dir) / config.data.output_files["ner_data"]
    if not data_path.exists():
        logging.info("NER data not found. Building dataset first...")
        build(config)

    name_model.create_blank_model("fr")
    data = name_model.load_data(str(data_path))

    split_idx = int(len(data) * 0.9)
    train_data, eval_data = data[:split_idx], data[split_idx:]

    logging.info(
        f"Training with {len(train_data)} examples, evaluating on {len(eval_data)}"
    )
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
    try:
        logging.error("This module is no longer a CLI. Use 'ners ner ...' instead.")
        return 1
    except Exception:
        traceback.print_exc()
        return 1
