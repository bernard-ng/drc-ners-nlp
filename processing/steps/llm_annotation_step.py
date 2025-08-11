import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import ollama
import pandas as pd
from pydantic import ValidationError

from core.config.pipeline_config import PipelineConfig
from core.utils.prompt_manager import PromptManager
from core.utils.rate_limiter import RateLimitConfig
from core.utils.rate_limiter import RateLimiter
from processing.batch.batch_config import BatchConfig
from processing.steps import PipelineStep, NameAnnotation


class LLMAnnotationStep(PipelineStep):
    """Configuration-driven LLM annotation step"""

    def __init__(self, pipeline_config: PipelineConfig):
        # Create custom batch config for LLM processing
        self.llm_config = pipeline_config.annotation.llm
        batch_config = BatchConfig(
            batch_size=pipeline_config.processing.batch_size,
            max_workers=min(
                self.llm_config.max_concurrent_requests,
                pipeline_config.processing.max_workers
            ),
            checkpoint_interval=pipeline_config.processing.checkpoint_interval,
            use_multiprocessing=pipeline_config.processing.use_multiprocessing,
        )
        super().__init__("llm_annotation", pipeline_config, batch_config)

        self.prompt = PromptManager(pipeline_config).load_prompt()
        self.rate_limiter = (
            self._create_rate_limiter() if self.llm_config.enable_rate_limiting else None
        )

        # Statistics
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_retry_attempts = 0

        # Setup logging
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def _create_rate_limiter(self):
        """Create rate limiter based on configuration"""
        rate_config = RateLimitConfig(
            requests_per_minute=self.llm_config.requests_per_minute,
            requests_per_second=self.llm_config.requests_per_second,
        )
        return RateLimiter(rate_config)

    def analyze_name(self, client: ollama.Client, name: str) -> Dict:
        """Analyze a name with retry logic and rate limiting"""
        for attempt in range(self.llm_config.retry_attempts):
            try:
                # Apply rate limiting if enabled
                if self.rate_limiter:
                    self.rate_limiter.wait_if_needed()

                start_time = time.time()
                response = client.chat(
                    model=self.llm_config.model_name,
                    messages=[
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": name},
                    ],
                    format=NameAnnotation.model_json_schema(),
                )
                elapsed_time = time.time() - start_time

                if elapsed_time > self.llm_config.timeout_seconds:
                    raise TimeoutError(
                        f"Request took {elapsed_time:.2f}s, exceeding {self.llm_config.timeout_seconds}s timeout"
                    )

                annotation = NameAnnotation.model_validate_json(response.message.content)
                result = {
                    **annotation.model_dump(),
                    "annotated": 1,
                    "processing_time": elapsed_time,
                    "attempts": attempt + 1,
                }

                self.successful_requests += 1
                if attempt > 0:
                    self.total_retry_attempts += attempt

                return result

            except (ValidationError, TimeoutError, Exception) as e:
                logging.warning(
                    f"Error analyzing '{name}' (attempt {attempt + 1}/{self.llm_config.retry_attempts}): {e}"
                )

                # Exponential backoff with jitter
                if attempt < self.llm_config.retry_attempts - 1:
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    time.sleep(min(wait_time, 10))

        self.failed_requests += 1
        return {
            "identified_name": None,
            "identified_surname": None,
            "annotated": 0,
            "processing_time": 0,
            "attempts": self.llm_config.retry_attempts,
            "failed": True,
        }

    def process_batch(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Process batch with LLM annotation"""
        unannotated_mask = batch.get("annotated", 0) == 0
        unannotated_entries = batch[unannotated_mask]

        if unannotated_entries.empty:
            logging.info(f"Batch {batch_id}: No entries to annotate")
            return batch

        logging.info(f"Batch {batch_id}: Annotating {len(unannotated_entries)} entries with LLM")

        batch = batch.copy()
        client = ollama.Client()

        # Process with controlled concurrency
        max_workers = self.llm_config.max_concurrent_requests

        if len(unannotated_entries) == 1 or max_workers == 1:
            # Sequential processing
            for idx, row in unannotated_entries.iterrows():
                result = self.analyze_name(client, row["name"])
                for field, value in result.items():
                    if field not in ["failed"]:
                        batch.loc[idx, field] = value
        else:
            # Concurrent processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}

                for idx, row in unannotated_entries.iterrows():
                    future = executor.submit(self.analyze_name, client, row["name"])
                    future_to_idx[future] = idx

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        for field, value in result.items():
                            if field not in ["failed"]:
                                batch.loc[idx, field] = value
                    except Exception as e:
                        logging.error(f"Failed to process row {idx}: {e}")
                        batch.loc[idx, "annotated"] = 0

        # Ensure proper data types
        batch["annotated"] = pd.to_numeric(batch["annotated"], errors="coerce").fillna(0).astype("Int8")

        return batch
