from __future__ import annotations

from enum import StrEnum

import numpy as np
import polars as pl


FULL_NAME_COLUMN = "name"


class NameView(StrEnum):
    """Supported predictor views for the three-token ablation study."""

    FULL = "full"
    NATIVE_ONLY = "native_only"


def coerce_name_view(value: str | NameView) -> NameView:
    try:
        return NameView(value)
    except ValueError as error:
        available = ", ".join(view.value for view in NameView)
        raise ValueError(f"Unknown name view '{value}'. Available: {available}") from error


def normalize_name_expression(column: str = FULL_NAME_COLUMN) -> pl.Expr:
    """Normalize only representation noise that cannot carry name information."""

    return (
        pl.col(column)
        .cast(pl.String)
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def name_view_expression(
    view: str | NameView,
    *,
    column: str = FULL_NAME_COLUMN,
) -> pl.Expr:
    """Project a normalized name to the configured model-visible sequence."""

    resolved = coerce_name_view(view)
    expression = pl.col(column)
    if resolved is NameView.NATIVE_ONLY:
        return expression.str.split(" ").list.slice(0, 2).list.join(" ")
    return expression


def name_view_series(values: pl.Series, view: str | NameView) -> pl.Series:
    """Project a series without changing row order or its source dataframe."""

    frame = pl.DataFrame({FULL_NAME_COLUMN: values})
    return frame.select(name_view_expression(view).alias(FULL_NAME_COLUMN)).to_series()


def native_group_array(frame: pl.DataFrame) -> np.ndarray:
    """Return the first-two-token grouping key used by leakage-safe validation."""

    return full_name_series(frame).str.split(" ").list.slice(0, 2).list.join(" ").to_numpy()


def full_name_series(frame: pl.DataFrame) -> pl.Series:
    """Return validated, non-null full-name strings from a model input frame."""

    if FULL_NAME_COLUMN not in frame.columns:
        raise ValueError("Model input must contain the published 'name' column")
    return frame.get_column(FULL_NAME_COLUMN).fill_null("").cast(pl.String)


def full_name_array(frame: pl.DataFrame) -> np.ndarray:
    """Return full names in the one-dimensional form expected by text estimators."""

    return full_name_series(frame).to_numpy()
