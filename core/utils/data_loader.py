import logging
from pathlib import Path
from typing import Optional, Union, Iterator

import pandas as pd

from core.config.pipeline_config import PipelineConfig


class DataLoader:
    """Reusable data loading utilities"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_csv_chunked(
        self, filepath: Union[str, Path], chunk_size: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        """Load CSV file in chunks for memory efficiency"""
        chunk_size = chunk_size or self.config.processing.chunk_size
        encodings = self.config.processing.encoding_options

        filepath = Path(filepath)

        for encoding in encodings:
            try:
                logging.info(f"Attempting to read {filepath} with encoding: {encoding}")

                chunk_iter = pd.read_csv(
                    filepath, encoding=encoding, chunksize=chunk_size, on_bad_lines="skip"
                )

                for i, chunk in enumerate(chunk_iter):
                    logging.debug(f"Processing chunk {i+1}")
                    yield chunk

                logging.info(f"Successfully read {filepath} with encoding: {encoding}")
                return

            except Exception as e:
                logging.warning(f"Failed with encoding {encoding}: {e}")
                continue

        raise ValueError(f"Unable to decode {filepath} with any encoding: {encodings}")

    def load_csv_complete(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load complete CSV file into memory"""
        chunks = list(self.load_csv_chunked(filepath))
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    @classmethod
    def save_csv(
        cls, df: pd.DataFrame, filepath: Union[str, Path], create_dirs: bool = True
    ) -> None:
        """Save DataFrame to CSV with proper handling"""
        filepath = Path(filepath)

        if create_dirs:
            filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False, encoding="utf-8")
        logging.info(f"Saved {len(df)} rows to {filepath}")
