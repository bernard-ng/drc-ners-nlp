"""Read-only summary of the published dataset for the experiment interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from drc_names_classifier.dataset import DatasetSchemaError, LABELS


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """Bounded preview and aggregate facts from one published CSV."""

    path: Path
    file_size_bytes: int
    row_count: int
    empty_name_count: int
    invalid_label_count: int
    schema: tuple[tuple[str, str], ...]
    label_distribution: pl.DataFrame
    preview: pl.DataFrame

    @property
    def valid(self) -> bool:
        return self.empty_name_count == 0 and self.invalid_label_count == 0


def inspect_dataset(
    path: str | Path,
    *,
    preview_rows: int = 20,
    name_column: str = "name",
    target_column: str = "sex",
) -> DatasetSnapshot:
    """Scan dataset facts with Polars without changing or exporting any rows."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset not found at {source_path}")
    if preview_rows < 0:
        raise ValueError("preview_rows must not be negative")

    lazy_frame = pl.scan_csv(
        source_path,
        schema_overrides={name_column: pl.String, target_column: pl.String},
        low_memory=True,
    )
    collected_schema = lazy_frame.collect_schema()
    missing = {name_column, target_column} - set(collected_schema.names())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise DatasetSchemaError(f"names.csv is missing columns: {missing_text}")

    projected = lazy_frame.select(name_column, target_column)
    name_is_empty = (
        pl.col(name_column).is_null() | pl.col(name_column).str.strip_chars().eq("")
    ).fill_null(True)
    label_is_invalid = (~pl.col(target_column).is_in(LABELS)).fill_null(True)
    summary = projected.select(
        pl.len().alias("row_count"),
        name_is_empty.sum().alias("empty_name_count"),
        label_is_invalid.sum().alias("invalid_label_count"),
        *[
            pl.col(target_column).eq(label).fill_null(False).sum().alias(label)
            for label in LABELS
        ],
    ).collect(engine="streaming")
    values = summary.row(0, named=True)
    row_count = int(values["row_count"])
    distribution = pl.DataFrame(
        {
            "sex": list(LABELS),
            "rows": [int(values[label]) for label in LABELS],
        },
        schema={"sex": pl.String, "rows": pl.Int64},
    ).with_columns(
        pl.when(row_count > 0).then(pl.col("rows") / row_count).otherwise(0.0).alias("share")
    )
    preview = projected.head(preview_rows).collect(engine="streaming")

    return DatasetSnapshot(
        path=source_path,
        file_size_bytes=source_path.stat().st_size,
        row_count=row_count,
        empty_name_count=int(values["empty_name_count"]),
        invalid_label_count=int(values["invalid_label_count"]),
        schema=tuple((name, str(dtype)) for name, dtype in collected_schema.items()),
        label_distribution=distribution,
        preview=preview,
    )
