import logging

import pandas as pd

from core.config.pipeline_config import PipelineConfig
from core.utils.text_cleaner import TextCleaner
from processing.steps import PipelineStep


class DataCleaningStep(PipelineStep):
    """Configuration-driven data cleaning step"""

    def __init__(self, pipeline_config: PipelineConfig):
        super().__init__("data_cleaning", pipeline_config)
        self.text_cleaner = TextCleaner()
        self.required_columns = ["name", "sex", "region"]

    def process_batch(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Process a single batch for data cleaning"""
        logging.info(f"Cleaning batch {batch_id} with {len(batch)} rows")

        # Drop rows with essential missing values
        batch = batch.dropna(subset=self.required_columns)

        # Apply text cleaning
        batch = self.text_cleaner.clean_dataframe_text_columns(batch)

        return batch
