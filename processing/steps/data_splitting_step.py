import numpy as np
import pandas as pd

from core.config.pipeline_config import PipelineConfig
from core.utils.region_mapper import RegionMapper
from processing.batch.batch_config import BatchConfig
from processing.steps import PipelineStep
from processing.steps.feature_extraction_step import Gender


class DataSplittingStep(PipelineStep):
    """Configuration-driven data splitting step"""

    def __init__(self, pipeline_config: PipelineConfig):
        batch_config = BatchConfig(
            batch_size=pipeline_config.processing.batch_size,
            max_workers=1,  # No need for parallelism in splitting
            checkpoint_interval=pipeline_config.processing.checkpoint_interval,
            use_multiprocessing=False,
        )
        super().__init__("data_splitting", pipeline_config, batch_config)
        self.eval_indices = None

    def determine_eval_indices(self, total_size: int) -> set:
        """Determine evaluation indices consistently across batches"""
        if self.eval_indices is None:
            np.random.seed(self.pipeline_config.data.random_seed)
            eval_size = int(total_size * self.pipeline_config.data.evaluation_fraction)
            self.eval_indices = set(np.random.choice(total_size, size=eval_size, replace=False))
        return self.eval_indices

    def process_batch(self, batch: pd.DataFrame, batch_id: int) -> pd.DataFrame:
        """Process batch for data splitting - no modification needed"""
        return batch

    def split(self, df: pd.DataFrame) -> None:
        """Save the split datasets based on configuration"""
        output_files = self.pipeline_config.data.output_files
        data_dir = self.pipeline_config.paths.data_dir

        if self.pipeline_config.data.split_evaluation:
            eval_indices = self.determine_eval_indices(len(df))
            eval_mask = df.index.isin(eval_indices)

            df_evaluation = df[eval_mask]
            df_featured = df[~eval_mask]

            self.data_loader.save_csv(df_evaluation, data_dir / output_files["evaluation"])
            self.data_loader.save_csv(df_featured, data_dir / output_files["featured"])
        else:
            self.data_loader.save_csv(df, data_dir / output_files["featured"])

        if self.pipeline_config.data.split_by_province:
            for province in RegionMapper.get_provinces():
                df_region = df[df.province == province]
                self.data_loader.save_csv(df_region, data_dir / "provinces" / f"{province}.csv")

        if self.pipeline_config.data.split_by_gender:
            df_males = df[df.sex == Gender.MALE.value]
            df_females = df[df.sex == Gender.FEMALE.value]

            self.data_loader.save_csv(df_males, data_dir / output_files["males"])
            self.data_loader.save_csv(df_females, data_dir / output_files["females"])
