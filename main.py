#!.venv/bin/python3
import argparse
import logging
import sys
import traceback

from core.config import setup_config
from core.utils import get_data_file_path
from core.utils.data_loader import DataLoader
from processing.batch.batch_config import BatchConfig
from processing.pipeline import Pipeline
from processing.steps.data_cleaning_step import DataCleaningStep
from processing.steps.data_splitting_step import DataSplittingStep
from processing.steps.feature_extraction_step import FeatureExtractionStep
from processing.steps.llm_annotation_step import LLMAnnotationStep


def create_pipeline(config) -> Pipeline:
    """Create pipeline from configuration"""
    batch_config = BatchConfig(
        batch_size=config.processing.batch_size,
        max_workers=config.processing.max_workers,
        checkpoint_interval=config.processing.checkpoint_interval,
        use_multiprocessing=config.processing.use_multiprocessing,
    )

    # Add steps based on configuration
    pipeline = Pipeline(batch_config)
    steps = [
        DataCleaningStep(config),
        FeatureExtractionStep(config),
        # NERAnnotationStep(config),
        LLMAnnotationStep(config),
    ]

    for stage in config.stages:
        for step in steps:
            if step.name == stage:
                pipeline.add_step(step)

    return pipeline


def run_pipeline(config) -> int:
    """Run the complete pipeline"""
    try:
        logging.info(f"Starting pipeline: {config.name} v{config.version}")

        # Load input data
        input_file_path = get_data_file_path(config.data.input_file, config)
        if not input_file_path.exists():
            logging.error(f"Input file not found: {input_file_path}")
            return 1

        data_loader = DataLoader(config)
        data_splitter = DataSplittingStep(config)
        logging.info(f"Loading data from {input_file_path}")
        df = data_loader.load_csv_complete(input_file_path)
        logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Create and run pipeline
        pipeline = create_pipeline(config)

        logging.info("Starting pipeline execution")
        data_splitter.split(pipeline.run(df))

        # Show completion statistics
        progress = pipeline.get_progress()
        logging.info("=== Pipeline Completion Summary ===")
        for step_name, stats in progress.items():
            logging.info(
                f"{step_name}: {stats['completion_percentage']:.1f}% "
                f"({stats['processed_batches']}/{stats['total_batches']} batches)"
            )
            if stats["failed_batches"] > 0:
                logging.warning(f"  {stats['failed_batches']} failed batches")

        logging.info("Pipeline completed successfully")
        return 0

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def main():
    """Main entry point with unified configuration loading"""
    parser = argparse.ArgumentParser(
        description="DRC NERS Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--env", type=str, default="development", help="Environment name")
    args = parser.parse_args()

    try:
        config = setup_config(config_path=args.config, env=args.env)
        return run_pipeline(config)

    except Exception as e:
        print(f"Pipeline failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
