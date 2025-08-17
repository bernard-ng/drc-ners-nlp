import gc
import logging
from pathlib import Path
from typing import Optional, Union, Iterator, Dict

import pandas as pd

from core.config.pipeline_config import PipelineConfig

OPTIMIZED_DTYPES = {
    # Numeric columns with appropriate bit-width
    "year": "Int16",  # Years fit in 16-bit integer
    "words": "Int8",  # Word counts typically < 128
    "length": "Int16",  # Name lengths fit in 16-bit
    "annotated": "Int8",  # Binary flag (0/1)
    "ner_tagged": "Int8",  # Binary flag (0/1)
    # Categorical columns (memory efficient for repeated values)
    "sex": "category",
    "province": "category",
    "region": "category",
    "identified_category": "category",
    "transformation_type": "category",
    # String columns with proper string dtype
    "name": "string",
    "probable_native": "string",
    "probable_surname": "string",
    "identified_name": "string",
    "identified_surname": "string",
    "ner_entities": "string",
}


class DataLoader:
    """Reusable data loading utilities"""

    def __init__(self, config: PipelineConfig, custom_dtypes: Optional[Dict] = None):
        self.config = config
        self.dtypes = {**OPTIMIZED_DTYPES, **(custom_dtypes or {})}

    def load_csv_chunked(
        self, filepath: Union[str, Path], chunk_size: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Load CSV file in chunks for memory efficiency"""
        chunk_size = chunk_size or self.config.processing.chunk_size
        encodings = self.config.processing.encoding_options
        filepath = Path(filepath)

        for encoding in encodings:
            try:
                logging.info(f"Reading {filepath} with encoding: {encoding}")

                # Read with optimal dtypes
                chunk_iter = pd.read_csv(
                    filepath,
                    encoding=encoding,
                    chunksize=chunk_size,
                    on_bad_lines="skip",
                    dtype=self.dtypes,
                )

                for i, chunk in enumerate(chunk_iter):
                    logging.debug(f"Processing optimized chunk {i + 1}")
                    yield chunk

                logging.info(f"Successfully read {filepath} with encoding: {encoding}")
                return

            except Exception as e:
                logging.warning(f"Failed with encoding {encoding}: {e}")
                continue

        raise ValueError(f"Unable to decode {filepath} with any encoding: {encodings}")

    def load_csv_complete(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load complete CSV with memory optimization"""
        chunks = []
        for chunk in self.load_csv_chunked(filepath):
            chunks.append(chunk)

        if not chunks:
            return pd.DataFrame()

        logging.info(f"Concatenating {len(chunks)} optimized chunks")
        df = pd.concat(chunks, ignore_index=True, copy=False)

        # Cleanup chunks from memory
        del chunks
        gc.collect()

        # Apply dataset size limiting if configured
        if self.config.data.max_dataset_size is not None:
            df = self._limit_dataset_size(df)

        return df

    def _limit_dataset_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limit dataset size with optional sex balancing"""
        max_size = self.config.data.max_dataset_size

        if max_size is None or len(df) <= max_size:
            return df

        if self.config.data.balance_by_sex and "sex" in df.columns:
            return self._balanced_sample(df, max_size)
        else:
            # Simple random sampling
            return df.sample(n=max_size, random_state=self.config.data.random_seed)

    def _balanced_sample(self, df: pd.DataFrame, max_size: int) -> pd.DataFrame:
        """Sample data with balanced sex distribution"""

        # Get unique sex values
        sex_values = df["sex"].dropna().unique()

        if len(sex_values) == 0:
            logging.warning(f"No valid values found in sex column 'sex', using random sampling")
            return df.sample(n=max_size, random_state=self.config.data.random_seed)

        # Calculate samples per sex category
        samples_per_sex = max_size // len(sex_values)
        remaining_samples = max_size % len(sex_values)

        balanced_samples = []

        for i, sex in enumerate(sex_values):
            # Use boolean indexing instead of creating temporary DataFrames
            sex_mask = df["sex"] == sex
            sex_indices = df[sex_mask].index

            # Distribute remaining samples to first categories
            current_samples = samples_per_sex + (1 if i < remaining_samples else 0)
            current_samples = min(current_samples, len(sex_indices))

            if current_samples > 0:
                # Sample indices instead of DataFrame
                sampled_indices = pd.Series(sex_indices).sample(
                    n=current_samples, random_state=self.config.data.random_seed + i
                )
                balanced_samples.extend(sampled_indices.tolist())
                logging.info(f"Sampled {current_samples} records for sex '{sex}'")

        if not balanced_samples:
            logging.warning("No balanced samples could be created, using random sampling")
            return df.sample(n=max_size, random_state=self.config.data.random_seed)

        # Create result using iloc with indices (no copying until final step)
        result = df.iloc[balanced_samples].copy()

        # Shuffle the final result
        result = result.sample(frac=1, random_state=self.config.data.random_seed).reset_index(
            drop=True
        )

        logging.info(f"Created balanced dataset with {len(result)} records from {len(df)} total")
        return result

    @classmethod
    def save_csv(
        cls, df: pd.DataFrame, filepath: Union[str, Path], create_dirs: bool = True
    ) -> None:
        """Save DataFrame to CSV with proper handling"""
        filepath = Path(filepath)

        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False, encoding="utf-8", sep=",", quoting=1)
        logging.info(f"Saved {len(df)} rows to {filepath}")
