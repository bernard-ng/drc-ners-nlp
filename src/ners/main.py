#!.venv/bin/python3
import logging
from ners.core.utils.data_loader import DataLoader
from ners.processing.batch.batch_config import BatchConfig
from ners.processing.pipeline import Pipeline
from ners.processing.steps.data_cleaning_step import DataCleaningStep
from ners.processing.steps.data_selection_step import DataSelectionStep
from ners.processing.steps.data_splitting_step import DataSplittingStep
from ners.processing.steps.llm_annotation_step import LLMAnnotationStep
from ners.processing.steps.ner_annotation_step import NERAnnotationStep
from ners.processing.steps.feature_extraction_step import FeatureExtractionStep


def create_pipeline(config) -> Pipeline:
    batch_config = BatchConfig(
        batch_size=config.processing.batch_size,
        max_workers=config.processing.max_workers,
        checkpoint_interval=config.processing.checkpoint_interval,
        use_multiprocessing=config.processing.use_multiprocessing,
    )

    pipeline = Pipeline(batch_config)
    steps = [
        DataCleaningStep(config),
        FeatureExtractionStep(config),
        DataSelectionStep(config),
        NERAnnotationStep(config),
        LLMAnnotationStep(config),
    ]

    for stage in config.stages:
        for step in steps:
            if step.name == stage:
                pipeline.add_step(step)

    return pipeline


def run_pipeline(config) -> int:
    try:
        logging.info(f"Starting pipeline: {config.name} v{config.version}")

        # Load input data
        input_file_path = config.paths.get_data_path(config.data.input_file)
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
