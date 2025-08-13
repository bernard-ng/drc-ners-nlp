import random
from typing import List
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from core.config import PipelineConfig
from core.utils import get_data_file_path
from core.utils.data_loader import OPTIMIZED_DTYPES, DataLoader
from processing.ner.formats.connectors_format import ConnectorFormatter
from processing.ner.formats.extended_surname_format import ExtendedSurnameFormatter
from processing.ner.formats.native_only_format import NativeOnlyFormatter
from processing.ner.formats.original_format import OriginalFormatter
from processing.ner.formats.position_flipped_format import PositionFlippedFormatter
from processing.ner.formats.reduced_native_format import ReducedNativeFormatter


class NEREngineering:
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
            "native_only": NativeOnlyFormatter(self.connectors, self.additional_surnames),
            "position_flipped": PositionFlippedFormatter(self.connectors, self.additional_surnames),
            "reduced_native": ReducedNativeFormatter(self.connectors, self.additional_surnames),
            "connector_added": ConnectorFormatter(self.connectors, self.additional_surnames),
            "extended_surname": ExtendedSurnameFormatter(self.connectors, self.additional_surnames),
        }

    def load_data(self) -> pd.DataFrame:
        """Load and filter NER-tagged data from CSV file"""

        filepath = get_data_file_path(self.config.data.output_files["featured"], self.config)
        df = self.data_loader.load_csv_complete(filepath)

        # Filter only NER-tagged rows
        ner_data = df[df["ner_tagged"] == 1].copy()
        logging.info(f"Loaded {len(ner_data)} NER-tagged records from {len(df)} total records")

        return ner_data

    def compute(self) -> None:
        logging.info("Applying feature engineering transformations...")
        input_filepath = get_data_file_path(self.config.data.output_files["featured"], self.config)
        output_filepath = get_data_file_path(
            self.config.data.output_files["engineered"], self.config
        )

        df = self.data_loader.load_csv_complete(input_filepath)
        ner_df = df[df["ner_tagged"] == 1].copy()
        logging.info(f"Loaded {len(ner_df)} NER-tagged records from {len(df)} total records")

        del df  # No need to keep in memory

        ner_df = ner_df.sample(frac=1, random_state=self.config.data.random_seed).reset_index(
            drop=True
        )
        total_rows = len(ner_df)

        # Calculate split points
        split_25_1 = int(total_rows * 0.25)
        split_25_2 = int(total_rows * 0.50)
        split_25_3 = int(total_rows * 0.75)
        split_10_1 = int(total_rows * 0.85)
        split_10_2 = int(total_rows * 0.95)

        # Define transformation groups
        groups = [
            (0, split_25_1, "original"),  # First 25%: original format
            (split_25_1, split_25_2, "native_only"),  # Second 25%: remove surname
            (split_25_2, split_25_3, "position_flipped"),  # Third 25%: flip positions
            (split_25_3, split_10_1, "reduced_native"),  # Fourth 10%: reduce native components
            (split_10_1, split_10_2, "connector_added"),  # Fifth 10%: add connectors
            (split_10_2, total_rows, "extended_surname"),  # Last 5%: extend surnames
        ]

        for start, end, trans_type in groups:
            logging.info(f"Group {trans_type}: {start} to {end} ({end - start} rows)")

        # Process each group
        rows = []
        for start, end, formatter_key in groups:
            formatter = self.formatters[formatter_key]

            for idx in tqdm(range(start, end), desc=f"Processing {formatter_key}"):
                row = ner_df.iloc[idx]
                transformed = formatter.transform(row)

                # Keep original columns and add transformed ones
                new_row = row.to_dict()
                new_row.update(transformed)
                rows.append(new_row)

        self.data_loader.save_csv(pd.DataFrame(rows), output_filepath)
        logging.info(f"Engineered dataset saved to {output_filepath}")
