import logging

import pandas as pd

from core.config.pipeline_config import PipelineConfig
from processing.steps import PipelineStep


class DataSelectionStep(PipelineStep):
    """Configuration-driven data selection step to keep only specified columns"""

    def __init__(self, pipeline_config: PipelineConfig):
        super().__init__("data_selection", pipeline_config)
        self.selected_columns = pipeline_config.data.selected_columns

    def process_batch(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Process a single batch for data selection"""
        logging.info(f"Selecting columns for batch {batch_id} with {len(batch)} rows")

        # Remove rows where region == "global" only for specific years
        if "region" in batch.columns and "year" in batch.columns:
            target_years = {2015, 2021, 2022}
            mask_remove = batch["region"].str.lower().eq("global") & batch["year"].isin(
                target_years
            )
            removed = int(mask_remove.sum())
            if removed:
                batch = batch[~mask_remove]
                logging.info(
                    f"Removed {removed} rows with region == 'global' for years {sorted(target_years)} in batch {batch_id}"
                )

        # Check which columns exist in the batch
        available_columns = [col for col in self.selected_columns if col in batch.columns]
        missing_columns = [col for col in self.selected_columns if col not in batch.columns]

        if missing_columns:
            logging.warning(f"Missing columns in batch {batch_id}: {missing_columns}")

        if not available_columns:
            logging.error(f"No required columns found in batch {batch_id}")
            return pd.DataFrame()  # Return empty DataFrame if no required columns exist

        # Select only the available required columns
        selected_batch = batch[available_columns].copy()

        logging.info(
            f"Selected {len(available_columns)} columns for batch {batch_id}: {available_columns}"
        )

        return selected_batch

    @property
    def requires_batch_mutation(self) -> bool:
        """This step modifies the batch data by selecting columns"""
        return True
