from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from ners.utils import (
    NameView,
    coerce_name_view,
    name_view_series,
    normalize_name_expression,
    stable_text_buckets,
)


LABELS = ("f", "m")
_BUCKET_COUNT = 10_000
_SPLIT_NAMESPACE = b"drcners-split1"
_SAMPLE_NAMESPACE = b"drcners-sample1"


class DatasetSchemaError(ValueError):
    """Raised when names.csv does not match the published schema."""


@dataclass(frozen=True, slots=True)
class DatasetBatch:
    names: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class DatasetPartition:
    """Train and test rows selected from one streamed CSV batch."""

    train: DatasetBatch | None
    test: DatasetBatch | None


class NameDataset:
    """Stream the published names.csv directly through Polars."""

    def __init__(
        self,
        path: str | Path,
        *,
        chunk_size: int,
        test_fraction: float,
        sample_fraction: float = 1.0,
        name_column: str = "name",
        target_column: str = "sex",
        split_group_view: str | NameView = NameView.FULL,
        required_token_count: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.test_fraction = test_fraction
        self.sample_fraction = sample_fraction
        self.name_column = name_column
        self.target_column = target_column
        self.split_group_view = coerce_name_view(split_group_view)
        self.required_token_count = required_token_count

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Dataset not found at {self.path}. Place names.csv in data/dataset/."
            )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be between zero and one")
        if self.required_token_count is not None and self.required_token_count <= 0:
            raise ValueError("required_token_count must be positive when configured")

        columns = set(pl.scan_csv(self.path).collect_schema().names())
        missing = {self.name_column, self.target_column} - columns
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise DatasetSchemaError(f"names.csv is missing columns: {missing_text}")

    def iter_partitions(self) -> Iterator[DatasetPartition]:
        """Yield train and test rows together while scanning the CSV only once."""

        for chunk_number, frame in enumerate(self._iter_frames(), start=1):
            frame = self._validate_frame(frame, chunk_number)
            if frame.is_empty():
                continue

            names = frame.get_column(self.name_column)
            group_names = name_view_series(names, self.split_group_view)
            if self.sample_fraction < 1.0:
                sample_threshold = round(self.sample_fraction * _BUCKET_COUNT)
                sample_buckets = stable_text_buckets(
                    group_names,
                    namespace=_SAMPLE_NAMESPACE,
                    bucket_count=_BUCKET_COUNT,
                )
                frame = frame.filter(pl.Series(sample_buckets < sample_threshold))
                if frame.is_empty():
                    continue
                names = frame.get_column(self.name_column)
                group_names = name_view_series(names, self.split_group_view)

            test_threshold = round(self.test_fraction * _BUCKET_COUNT)
            split_buckets = stable_text_buckets(
                group_names,
                namespace=_SPLIT_NAMESPACE,
                bucket_count=_BUCKET_COUNT,
            )
            is_test = split_buckets < test_threshold
            yield DatasetPartition(
                train=self._to_batch(frame.filter(pl.Series(~is_test))),
                test=self._to_batch(frame.filter(pl.Series(is_test))),
            )

    def _iter_frames(self) -> Iterator[pl.DataFrame]:
        lazy_frame = pl.scan_csv(
            self.path,
            schema_overrides={self.name_column: pl.String, self.target_column: pl.String},
            low_memory=True,
        ).select(self.name_column, self.target_column)
        yield from lazy_frame.collect_batches(
            chunk_size=self.chunk_size,
            maintain_order=True,
            engine="streaming",
        )

    def _validate_frame(self, frame: pl.DataFrame, chunk_number: int) -> pl.DataFrame:
        invalid_labels = frame.select(
            (
                pl.col(self.target_column).is_null() | ~pl.col(self.target_column).is_in(LABELS)
            ).sum()
        ).item()
        if invalid_labels:
            raise DatasetSchemaError(
                "Invalid published dataset values in chunk "
                f"{chunk_number}: {invalid_labels} labels outside {LABELS}."
            )

        valid_names = pl.col(self.name_column).is_not_null() & pl.col(
            self.name_column
        ).str.strip_chars().ne("")
        valid_count = frame.select(valid_names.sum()).item()
        invalid_names = frame.height - int(valid_count)
        if invalid_names:
            logging.warning(
                "Skipping %d rows with empty names in dataset chunk %d",
                invalid_names,
                chunk_number,
            )
            frame = frame.filter(valid_names)

        frame = frame.with_columns(
            normalize_name_expression(self.name_column).alias(self.name_column)
        )
        if self.required_token_count is not None:
            frame = frame.filter(
                pl.col(self.name_column).str.split(" ").list.len() == self.required_token_count
            )
        return frame

    def _to_batch(self, frame: pl.DataFrame) -> DatasetBatch | None:
        if frame.is_empty():
            return None
        return DatasetBatch(
            names=frame.get_column(self.name_column).to_numpy(),
            labels=frame.get_column(self.target_column).to_numpy(),
        )
