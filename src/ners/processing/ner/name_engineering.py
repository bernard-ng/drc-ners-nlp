import gc
import random
import logging
from typing import List, Dict, Any, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from ners.core.config import PipelineConfig
from ners.core.utils.data_loader import DataLoader
from ners.processing.ner.formats.connectors_format import ConnectorFormatter
from ners.processing.ner.formats.extended_surname_format import ExtendedSurnameFormatter
from ners.processing.ner.formats.native_only_format import NativeOnlyFormatter
from ners.processing.ner.formats.original_format import OriginalFormatter
from ners.processing.ner.formats.position_flipped_format import PositionFlippedFormatter
from ners.processing.ner.formats.reduced_native_format import ReducedNativeFormatter


class NameEngineering:
    """
    Feature engineering for NER dataset to prevent position-based learning
    and encourage sequence characteristic learning.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_loader = DataLoader(config)
        self.connectors = ["wa", "ya", "ka", "ba", "la"]
        self.additional_surnames = [
            "jean",
            "paul",
            "marie",
            "joseph",
            "pierre",
            "claude",
            "andre",
            "michel",
            "robert",
        ]

        random.seed(self.config.data.random_seed)
        np.random.seed(self.config.data.random_seed)

        # Initialize format classes
        self.formatters = {
            "original": OriginalFormatter(self.connectors, self.additional_surnames),
            "native_only": NativeOnlyFormatter(
                self.connectors, self.additional_surnames
            ),
            "position_flipped": PositionFlippedFormatter(
                self.connectors, self.additional_surnames
            ),
            "reduced_native": ReducedNativeFormatter(
                self.connectors, self.additional_surnames
            ),
            "connector_added": ConnectorFormatter(
                self.connectors, self.additional_surnames
            ),
            "extended_surname": ExtendedSurnameFormatter(
                self.connectors, self.additional_surnames
            ),
        }

    def load_data(self) -> pd.DataFrame:
        """Load and filter NER-tagged data from CSV file"""

        filepath = self.config.paths.get_data_path(
            self.config.data.output_files["featured"]
        )
        df = self.data_loader.load_csv_complete(filepath)

        mask = df["ner_tagged"] == 1
        ner_data = df.loc[mask].copy()
        
        if not isinstance(ner_data, pd.DataFrame):
            ner_data = ner_data.to_frame().T

        logging.info(
            f"Loaded {len(ner_data)} NER-tagged records from {len(df)} total records"
        )

        return cast(pd.DataFrame, ner_data)

    def compute(self) -> None:
        """Apply transformations and save engineered dataset"""
        logging.info("Applying feature engineering transformations...")
        input_filepath = self.config.paths.get_data_path(
            self.config.data.output_files["featured"]
        )
        output_filepath = self.config.paths.get_data_path(
            self.config.data.output_files["engineered"]
        )

        df = self.data_loader.load_csv_complete(input_filepath)
        
        # Consistent filtering with type safety
        mask = df["ner_tagged"] == 1
        ner_df = df.loc[mask].copy()
        
        if not isinstance(ner_df, pd.DataFrame):
            ner_df = ner_df.to_frame().T

        logging.info(
            f"Loaded {len(ner_df)} NER-tagged records from {len(df)} total records"
        )

        del df  # No need to keep in memory
        gc.collect()

        # Shuffle dataset
        ner_df = ner_df.sample(
            frac=1, random_state=self.config.data.random_seed
        ).reset_index(drop=True)
        total_rows = len(ner_df)

        # Calculate split points
        split_25_1 = int(total_rows * 0.25)
        split_25_2 = int(total_rows * 0.50)
        split_25_3 = int(total_rows * 0.75)
        split_10_1 = int(total_rows * 0.85)
        split_10_2 = int(total_rows * 0.95)

        # Define transformation groups
        groups = [
            (0, split_25_1, "original"),  
            (split_25_1, split_25_2, "native_only"),  
            (split_25_2, split_25_3, "position_flipped"),  
            (split_25_3, split_10_1, "reduced_native"),  
            (split_10_1, split_10_2, "connector_added"),  
            (split_10_2, total_rows, "extended_surname"),  
        ]

        for start, end, trans_type in groups:
            logging.info(f"Group {trans_type}: {start} to {end} ({end - start} rows)")

        # Process each group
        rows: List[Dict[str, Any]] = []
        for start, end, formatter_key in groups:
            formatter = self.formatters[formatter_key]

            for idx in tqdm(range(start, end), desc=f"Processing {formatter_key}"):
                row = ner_df.iloc[idx]
                transformed = formatter.transform(row)

                # Keep original columns and add transformed ones
                new_row = row.to_dict()
                new_row.update(transformed)
                rows.append(new_row)

        # Save results
        self.data_loader.save_csv(pd.DataFrame(rows), output_filepath)
        logging.info(f"Engineered dataset saved to {output_filepath}")