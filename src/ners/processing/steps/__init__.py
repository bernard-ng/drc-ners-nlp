import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

from ners.core.config.pipeline_config import PipelineConfig
from ners.core.utils.data_loader import DataLoader
from ners.processing.batch.batch_config import BatchConfig


@dataclass
class PipelineState:
    """Tracks the state of pipeline execution"""

    processed_batches: int = 0
    total_batches: int = 0
    failed_batches: Optional[List[int]] = None
    last_checkpoint: Optional[str] = None

    def __post_init__(self):
        if self.failed_batches is None:
            self.failed_batches = []


class NameAnnotation(BaseModel):
    """Model for name annotation results"""

    identified_name: Optional[str]
    identified_surname: Optional[str]


class PipelineStep(ABC):
    """Abstract base class for pipeline steps"""

    def __init__(
        self,
        name: str,
        pipeline_config: PipelineConfig,
        batch_config: Optional[BatchConfig] = None,
    ):
        self.name = name
        self.pipeline_config = pipeline_config
        self.data_loader = DataLoader(pipeline_config)

        # Use provided batch_config or create default from pipeline config
        if batch_config is None:
            batch_config = BatchConfig(
                batch_size=pipeline_config.processing.batch_size,
                max_workers=pipeline_config.processing.max_workers,
                checkpoint_interval=pipeline_config.processing.checkpoint_interval,
                use_multiprocessing=pipeline_config.processing.use_multiprocessing,
            )
        self.batch_config = batch_config
        self.state = PipelineState()

    @property
    def requires_batch_mutation(self) -> bool:
        """Indicates if this step modifies the batch data"""
        return False

    @abstractmethod
    def process_batch(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Process a single batch of data"""
        pass

    def get_checkpoint_path(self, batch_id: int) -> str:
        """Get the checkpoint file path for a batch"""
        checkpoint_dir = self.pipeline_config.paths.checkpoints_dir / self.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return str(checkpoint_dir / f"batch_{batch_id:06d}.csv")

    def get_state_path(self) -> str:
        """Get the state file path"""
        state_dir = self.pipeline_config.paths.checkpoints_dir / self.name
        state_dir.mkdir(parents=True, exist_ok=True)
        return str(state_dir / "pipeline_state.json")

    def save_state(self):
        """Save pipeline state to disk"""
        state_file = self.get_state_path()
        with open(state_file, "w") as f:
            json.dump(
                {
                    "processed_batches": self.state.processed_batches,
                    "total_batches": self.state.total_batches,
                    "failed_batches": self.state.failed_batches,
                    "last_checkpoint": self.state.last_checkpoint,
                },
                f,
            )

    def load_state(self) -> bool:
        """Load pipeline state from disk. Returns True if state was loaded."""
        state_file = self.get_state_path()
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                self.state.processed_batches = state_data.get("processed_batches", 0)
                self.state.total_batches = state_data.get("total_batches", 0)
                self.state.failed_batches = state_data.get("failed_batches", [])
                self.state.last_checkpoint = state_data.get("last_checkpoint")
                return True
            except Exception as e:
                logging.warning(f"Failed to load state: {e}")
        return False

    def batch_exists(self, batch_id: int) -> bool:
        """Check if a batch has already been processed (idempotency)"""
        checkpoint_path = self.get_checkpoint_path(batch_id)
        return os.path.exists(checkpoint_path)

    def save_batch(self, batch: pd.DataFrame, batch_id: int):
        """Save processed batch to checkpoint"""
        checkpoint_path = self.get_checkpoint_path(batch_id)
        self.data_loader.save_csv(batch, checkpoint_path)
        logging.info(f"Saved batch {batch_id} to {checkpoint_path}")

    def load_batch(self, batch_id: int) -> Optional[pd.DataFrame]:
        """Load processed batch from checkpoint"""
        checkpoint_path = self.get_checkpoint_path(batch_id)
        if os.path.exists(checkpoint_path):
            return self.data_loader.load_csv_complete(checkpoint_path)
        return None
