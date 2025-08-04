#!.venv/bin/python3
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from core.utils.data_loader import DataLoader
from core.config import ConfigManager, setup_logging
from core.utils import ensure_directories, get_data_file_path

from processing.pipeline import Pipeline
from processing.batch.batch_config import BatchConfig
from processing.steps.data_splitting_step import DataSplittingStep
from processing.steps.llm_annotation_step import LLMAnnotationStep
from processing.steps.feature_extraction_step import FeatureExtractionStep
from processing.steps.data_cleaning_step import DataCleaningStep


def create_pipeline_from_config(config_path: Optional[Path] = None) -> Pipeline:
    """Create pipeline from configuration file"""
    config = ConfigManager(config_path).load_config()

    # Setup logging
    setup_logging(config)
    ensure_directories(config)
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


def run_pipeline(config_path: Optional[Path] = None, resume: bool = False) -> int:
    """Run the complete pipeline"""
    try:
        config = ConfigManager(config_path).load_config()

        logging.info(f"Starting pipeline: {config.name} v{config.version}")
        logging.info(f"Environment: {config.environment}")

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
        pipeline = create_pipeline_from_config(config_path)

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
    """Main entry point with minimal command-line interface"""
    parser = argparse.ArgumentParser(
        description="DRC Names Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration File Examples:
  config/pipeline.yaml              - Main configuration
  config/pipeline.development.yaml  - Development environment
  config/pipeline.production.yaml   - Production environment

Usage Examples:
  python processing/main.py                                   # Use default config
  python processing/main.py --config config/pipeline.yaml     # Use specific config
  python processing/main.py --env development                 # Use environment config
  python processing/main.py --resume                          # Resume from checkpoints
        """,
    )

    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument(
        "--env", type=str, help="Environment name (loads config/pipeline.{env}.yaml)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume pipeline from existing checkpoints"
    )
    parser.add_argument(
        "--validate-config", action="store_true", help="Validate configuration file and exit"
    )
    args = parser.parse_args()

    # Determine config path
    config_path = None
    if args.config:
        config_path = args.config
    elif args.env:
        config_path = Path("config") / f"pipeline.{args.env}.yaml"

    if args.validate_config:
        try:
            config = ConfigManager(config_path).load_config()
            print(f"Configuration is valid: {config.name} v{config.version}")
            return 0
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return 1

    # Run pipeline
    return run_pipeline(config_path, args.resume)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
