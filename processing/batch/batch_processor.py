import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Iterator

import pandas as pd

from processing.batch.batch_config import BatchConfig
from processing.steps import PipelineStep


class BatchProcessor:
    """Handles batch processing with concurrency and checkpointing"""

    def __init__(self, config: BatchConfig):
        self.config = config

    def create_batches(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, int]]:
        """Create batches from DataFrame"""
        total_rows = len(df)
        batch_size = self.config.batch_size

        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i : i + batch_size].copy()
            batch_id = i // batch_size
            yield batch, batch_id

    def process_sequential(self, step: PipelineStep, df: pd.DataFrame) -> pd.DataFrame:
        """Process batches sequentially"""
        results = []

        for batch, batch_id in self.create_batches(df):
            if step.batch_exists(batch_id):
                logging.info(f"Batch {batch_id} already processed, loading from checkpoint")
                processed_batch = step.load_batch(batch_id)
            else:
                try:
                    processed_batch = step.process_batch(batch, batch_id)
                    step.save_batch(processed_batch, batch_id)
                    step.state.processed_batches += 1
                except Exception as e:
                    logging.error(f"Failed to process batch {batch_id}: {e}")
                    step.state.failed_batches.append(batch_id)
                    continue

            results.append(processed_batch)

            # Save state periodically
            if batch_id % self.config.checkpoint_interval == 0:
                step.save_state()

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def process_concurrent(self, step: PipelineStep, df: pd.DataFrame) -> pd.DataFrame:
        """Process batches concurrently"""
        executor_class = (
            ProcessPoolExecutor if self.config.use_multiprocessing else ThreadPoolExecutor
        )
        results = {}

        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {}
            for batch, batch_id in self.create_batches(df):
                if step.batch_exists(batch_id):
                    logging.info(f"Batch {batch_id} already processed, loading from checkpoint")
                    results[batch_id] = step.load_batch(batch_id)
                else:
                    future = executor.submit(step.process_batch, batch, batch_id)
                    future_to_batch[future] = (batch_id, batch)

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_id, batch = future_to_batch[future]
                try:
                    processed_batch = future.result()
                    step.save_batch(processed_batch, batch_id)
                    results[batch_id] = processed_batch
                    step.state.processed_batches += 1
                    logging.info(f"Completed batch {batch_id}")
                except Exception as e:
                    logging.error(f"Failed to process batch {batch_id}: {e}")
                    step.state.failed_batches.append(batch_id)

        # Reassemble results in order
        ordered_results = []
        for batch_id in sorted(results.keys()):
            ordered_results.append(results[batch_id])

        step.save_state()
        return pd.concat(ordered_results, ignore_index=True) if ordered_results else pd.DataFrame()

    def process(self, step: PipelineStep, df: pd.DataFrame) -> pd.DataFrame:
        """Process data using the configured strategy"""
        step.state.total_batches = (len(df) + self.config.batch_size - 1) // self.config.batch_size
        step.load_state()

        logging.info(f"Starting {step.name} with {step.state.total_batches} batches")

        if self.config.max_workers == 1:
            return self.process_sequential(step, df)
        else:
            return self.process_concurrent(step, df)
