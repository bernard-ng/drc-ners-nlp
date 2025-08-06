#!.venv/bin/python3
import sys
import argparse
import logging
from pathlib import Path

from core.utils.data_loader import DataLoader
from core.config import setup_config_and_logging
from core.utils import get_data_file_path

from processing.pipeline import Pipeline
from processing.batch.batch_config import BatchConfig
from processing.steps.data_splitting_step import DataSplittingStep
from processing.steps.llm_annotation_step import LLMAnnotationStep
from processing.steps.feature_extraction_step import FeatureExtractionStep
from processing.steps.data_cleaning_step import DataCleaningStep


def create_pipeline_from_config(config) -> Pipeline:
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
        LLMAnnotationStep(config),
        DataSplittingStep(config),
    ]

    for stage in config.stages:
        for step in steps:
            if step.name == stage:
                pipeline.add_step(step)

    return pipeline


def run_pipeline(config, resume: bool = False) -> int:
    """Run the complete pipeline"""
    try:
        logging.info(f"Starting pipeline: {config.name} v{config.version}")

        # Load input data
        input_file_path = get_data_file_path(config.data.input_file, config)

        if not input_file_path.exists():
            logging.error(f"Input file not found: {input_file_path}")
            return 1

        data_loader = DataLoader(config)
        logging.info(f"Loading data from {input_file_path}")
        df = data_loader.load_csv_complete(input_file_path)
        logging.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Create and run pipeline
        pipeline = create_pipeline_from_config(config)

        logging.info("Starting pipeline execution")
        result_df = pipeline.run(df)

        # Save results using the splitting step
        splitting_step = pipeline.steps[-1]
        if isinstance(splitting_step, DataSplittingStep):
            splitting_step.save_splits(result_df)

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
        description="DRC Names Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration File Examples:
  config/pipeline.yaml              - Main configuration
  config/pipeline.development.yaml  - Development environment (default)
  config/pipeline.production.yaml   - Production environment

Usage Examples:
  python main.py                                   # Use development config (default)
  python main.py --config config/pipeline.yaml    # Use specific config
  python main.py --env production                  # Use production environment
  python main.py --resume                         # Resume from checkpoints
        """,
    )

    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument(
        "--env", type=str, default="development",
        help="Environment name (default: development)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume pipeline from existing checkpoints"
    )
    parser.add_argument(
        "--validate-config", action="store_true", help="Validate configuration file and exit"
    )
    args = parser.parse_args()

    try:
        # Load configuration and setup logging
        config = setup_config_and_logging(config_path=args.config, env=args.env)

        if args.validate_config:
            print(f"Configuration is valid: {config.name} v{config.version}")
            return 0

        # Run pipeline
        return run_pipeline(config, args.resume)

    except Exception as e:
        print(f"Configuration or pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
