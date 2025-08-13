import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Iterator

import pandas as pd

from processing.batch.batch_config import BatchConfig
from processing.batch.memory_monitor import MemoryMonitor
from processing.steps import PipelineStep


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
        results = []
        memory_threshold_mb = 1000  # Clean memory when usage exceeds 1 GB

        for batch_num, (batch, batch_id) in enumerate(self.create_batches(df)):
            if step.batch_exists(batch_id):
                logging.info(f"Batch {batch_id} already processed, loading from checkpoint")
                processed_batch = step.load_batch(batch_id)
            else:
                try:
                    # Only copy if the processing step requires mutation
                    if step.requires_batch_mutation:
                        batch_copy = batch.copy()
                        processed_batch = step.process_batch(batch_copy, batch_id)
                    else:
                        processed_batch = step.process_batch(batch, batch_id)

                    step.save_batch(processed_batch, batch_id)
                    step.state.processed_batches += 1
                except Exception as e:
                    logging.error(f"Failed to process batch {batch_id}: {e}")
                    step.state.failed_batches.append(batch_id)
                    continue

            results.append(processed_batch)

            # Memory management
            if batch_num % self.config.checkpoint_interval == 0:
                current_memory = self.memory_monitor.get_memory_usage_mb()
                if current_memory > memory_threshold_mb:
                    logging.info(f"Memory cleanup triggered at {current_memory:.1f} MB")
                    self.memory_monitor.cleanup_memory()

            # Save state periodically
            if batch_id % self.config.checkpoint_interval == 0:
                step.save_state()

        # Final memory cleanup before concatenation
        self.memory_monitor.cleanup_memory()
        self.memory_monitor.log_memory_usage("before_concat")

        result = self._safe_concat(results) if results else pd.DataFrame()

        # Final cleanup
        del results
        self.memory_monitor.cleanup_memory()
        self.memory_monitor.log_memory_usage("sequential_complete")

        return result

    def process_concurrent(self, step: PipelineStep, df: pd.DataFrame) -> pd.DataFrame:
        """Memory-optimized concurrent processing"""
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
                    # Only copy if necessary for concurrent processing
                    batch_copy = batch.copy() if step.requires_batch_mutation else batch
                    future = executor.submit(step.process_batch, batch_copy, batch_id)
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

        # Memory-efficient reassembly
        ordered_results = []
        for batch_id in sorted(results.keys()):
            ordered_results.append(results[batch_id])

        step.save_state()

        # Cleanup before concat
        del results
        self.memory_monitor.cleanup_memory()

        result = self._safe_concat(ordered_results) if ordered_results else pd.DataFrame()

        # Final cleanup
        del ordered_results
        self.memory_monitor.cleanup_memory()

        return result

    def process(self, step: PipelineStep, df: pd.DataFrame) -> pd.DataFrame:
        """Process data using the configured strategy"""
        step.state.total_batches = (len(df) + self.config.batch_size - 1) // self.config.batch_size
        step.load_state()

        logging.info(f"Starting {step.name} with {step.state.total_batches} batches")
        self.memory_monitor.log_memory_usage("process_start")

        if self.config.max_workers == 1:
            result = self.process_sequential(step, df)
        else:
            result = self.process_concurrent(step, df)

        self.memory_monitor.log_memory_usage("process_complete")
        return result

    def _safe_concat(self, dfs: list) -> pd.DataFrame:
        """Memory-safe concatenation with monitoring"""
        if not dfs:
            return pd.DataFrame()

        memory = self.memory_monitor.get_memory_usage_mb()
        logging.info(f"Starting concat of {len(dfs)} DataFrames at {memory:.1f} MB")

        # Use copy=False to avoid unnecessary copying during concat
        result = pd.concat(dfs, ignore_index=True, copy=False)

        # Monitor memory after concat
        memory = self.memory_monitor.get_memory_usage_mb()
        logging.info(f"Concat complete. Memory: {memory:.1f} MB")

        return result
