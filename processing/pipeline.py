import logging

import pandas as pd
from typing import Dict, Any
import time

from processing.batch.batch_config import BatchConfig
from processing.batch.batch_processor import BatchProcessor
from processing.steps import PipelineStep


class Pipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.processor = BatchProcessor(config)
        self.steps = []

    def add_step(self, step: PipelineStep):
        """Add a processing step to the pipeline"""
        self.steps.append(step)

    def run(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Run the complete pipeline"""
        current_data = input_data.copy()

        for step in self.steps:
            logging.info(f"Running pipeline step: {step.name}")
            start_time = time.time()

            current_data = self.processor.process(step, current_data)

            elapsed_time = time.time() - start_time
            logging.info(f"Completed {step.name} in {elapsed_time:.2f} seconds")

            if step.state.failed_batches:
                logging.warning(
                    f"Step {step.name} had {len(step.state.failed_batches)} failed batches"
                )

        return current_data

    def get_progress(self) -> Dict[str, Any]:
        """Get progress information for all steps"""
        progress = {}
        for step in self.steps:
            progress[step.name] = {
                "processed_batches": step.state.processed_batches,
                "total_batches": step.state.total_batches,
                "failed_batches": len(step.state.failed_batches),
                "completion_percentage": (
                    step.state.processed_batches / max(1, step.state.total_batches)
                )
                * 100,
            }
        return progress
