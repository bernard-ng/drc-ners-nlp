import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Iterator

import pandas as pd

from ners.processing.batch.batch_config import BatchConfig
from ners.processing.batch.memory_monitor import MemoryMonitor
from ners.processing.steps import PipelineStep


class BatchProcessor:
    """Handles batch processing with concurrency and checkpointing"""

    def __init__(self, config: BatchConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()

    def create_batches(self, df: pd.DataFrame) -> Iterator[tuple[pd.DataFrame, int]]:
        """Create batches from DataFrame without unnecessary copies"""
        total_rows = len(df)
        batch_size = self.config.batch_size

        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i : i + batch_size]
            batch_id = i // batch_size
            yield batch, batch_id

    def process_sequential(self, step: PipelineStep, df: pd.DataFrame) -> pd.DataFrame:
        """Memory-optimized sequential processing"""
        results: list[pd.DataFrame] = []
        memory_threshold_mb = 1000

        for batch_num, (batch, batch_id) in enumerate(self.create_batches(df)):
            processed_batch: pd.DataFrame | None

            if step.batch_exists(batch_id):
                logging.info(
                    f"Batch {batch_id} already processed, loading from checkpoint"
                )
                processed_batch = step.load_batch(batch_id)
            else:
                try:
                    if step.requires_batch_mutation:
                        batch = batch.copy()

                    processed_batch = step.process_batch(batch, batch_id)

                    if processed_batch is not None:
                        step.save_batch(processed_batch, batch_id)
                        step.state.processed_batches += 1
                except Exception as e:
                    logging.error(f"Failed to process batch {batch_id}: {e}")
                    processed_batch = None

            if processed_batch is None:
                if step.state.failed_batches is None:
                    step.state.failed_batches = []
                step.state.failed_batches.append(batch_id)
                continue

            results.append(processed_batch)

            if batch_num % self.config.checkpoint_interval == 0:
                current_memory = self.memory_monitor.get_memory_usage_mb()
                if current_memory > memory_threshold_mb:
                    self.memory_monitor.cleanup_memory()

            if batch_id % self.config.checkpoint_interval == 0:
                step.save_state()

        self.memory_monitor.cleanup_memory()
        result = self._safe_concat(results) if results else pd.DataFrame()
        return result

    def process_concurrent(self, step: PipelineStep, df: pd.DataFrame) -> pd.DataFrame:
        """Memory-optimized concurrent processing"""
        executor_class = (
            ProcessPoolExecutor
            if self.config.use_multiprocessing
            else ThreadPoolExecutor
        )

        results: dict[int, pd.DataFrame] = {}

        with executor_class(max_workers=self.config.max_workers) as executor:
            future_to_batch: dict = {}

            for batch, batch_id in self.create_batches(df):
                if step.batch_exists(batch_id):
                    loaded = step.load_batch(batch_id)
                    if loaded is not None:
                        results[batch_id] = loaded
                else:
                    if step.requires_batch_mutation:
                        batch = batch.copy()

                    future = executor.submit(step.process_batch, batch, batch_id)
                    future_to_batch[future] = batch_id

            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    processed_batch = future.result()
                except Exception as e:
                    logging.error(f"Failed to process batch {batch_id}: {e}")
                    processed_batch = None

                if processed_batch is None:
                    if step.state.failed_batches is None:
                        step.state.failed_batches = []
                    step.state.failed_batches.append(batch_id)
                    continue

                step.save_batch(processed_batch, batch_id)
                results[batch_id] = processed_batch
                step.state.processed_batches += 1

        ordered_results = [results[k] for k in sorted(results)]
        step.save_state()
        return self._safe_concat(ordered_results) if ordered_results else pd.DataFrame()

    def process(self, step: PipelineStep, df: pd.DataFrame) -> pd.DataFrame:
        step.state.total_batches = (
            len(df) + self.config.batch_size - 1
        ) // self.config.batch_size
        step.load_state()

        if self.config.max_workers == 1:
            return self.process_sequential(step, df)

        return self.process_concurrent(step, df)

    def _safe_concat(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True, copy=False)
