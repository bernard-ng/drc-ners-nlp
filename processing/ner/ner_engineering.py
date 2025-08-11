import random
from typing import List

import numpy as np
import pandas as pd

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

    def __init__(self, connectors: List[str] = None, additional_surnames: List[str] = None):
        self.connectors = connectors or ['wa', 'ya', 'ka', 'ba', 'la']
        self.additional_surnames = additional_surnames or [
            'jean', 'paul', 'marie', 'joseph', 'pierre', 'claude',
            'andre', 'michel', 'robert'
        ]

        # Initialize format classes
        self.formatters = {
            'original': OriginalFormatter(self.connectors, self.additional_surnames),
            'native_only': NativeOnlyFormatter(self.connectors, self.additional_surnames),
            'position_flipped': PositionFlippedFormatter(self.connectors, self.additional_surnames),
            'reduced_native': ReducedNativeFormatter(self.connectors, self.additional_surnames),
            'connector_added': ConnectorFormatter(self.connectors, self.additional_surnames),
            'extended_surname': ExtendedSurnameFormatter(self.connectors, self.additional_surnames)
        }

    @classmethod
    def load_ner_data(cls, filepath: str) -> pd.DataFrame:
        """Load and filter NER-tagged data from CSV file"""
        df = pd.read_csv(filepath)

        # Filter only NER-tagged rows
        ner_data = df[df['ner_tagged'] == 1].copy()
        print(f"Loaded {len(ner_data)} NER-tagged records from {len(df)} total records")

        return ner_data

    def engineer_dataset(self, df: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
        """
        Apply feature engineering transformations according to the specified rules:
        - First 25%: original format
        - Second 25%: remove surname
        - Third 25%: flip positions
        - Fourth 10%: reduce native components
        - Fifth 10%: add connectors
        - Last 5%: extend surnames
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Shuffle the dataset
        df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        total_rows = len(df_shuffled)

        # Calculate split points
        split_25_1 = int(total_rows * 0.25)
        split_25_2 = int(total_rows * 0.50)
        split_25_3 = int(total_rows * 0.75)
        split_10_1 = int(total_rows * 0.85)
        split_10_2 = int(total_rows * 0.95)

        # Define transformation groups
        transformation_groups = [
            (0, split_25_1, 'original'),
            (split_25_1, split_25_2, 'native_only'),
            (split_25_2, split_25_3, 'position_flipped'),
            (split_25_3, split_10_1, 'reduced_native'),
            (split_10_1, split_10_2, 'connector_added'),
            (split_10_2, total_rows, 'extended_surname')
        ]

        print("Dataset splits:")
        for start, end, trans_type in transformation_groups:
            print(f"Group {trans_type}: {start} to {end} ({end - start} rows)")

        # Process each group
        engineered_rows = []
        for start, end, formatter_key in transformation_groups:
            formatter = self.formatters[formatter_key]

            for idx in range(start, end):
                row = df_shuffled.iloc[idx]
                transformed = formatter.transform(row)

                # Keep original columns and add transformed ones
                new_row = row.to_dict()
                new_row.update(transformed)
                engineered_rows.append(new_row)

        return pd.DataFrame(engineered_rows)

    @classmethod
    def save_engineered_dataset(cls, df: pd.DataFrame, output_path: str):
        """Save the engineered dataset to CSV file"""
        df.to_csv(output_path, index=False)
        print(f"Engineered dataset saved to {output_path}")
