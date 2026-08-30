from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ners.dataset import DatasetSchemaError, NameDataset


def _write_dataset(path: Path, rows: list[dict[str, str]]) -> None:
    pl.DataFrame(rows).write_csv(path)


def test_duplicate_names_stay_in_one_split(tmp_path: Path) -> None:
    path = tmp_path / "names.csv"
    rows = [
        {"name": f"family {index} given", "sex": "f" if index % 2 else "m"}
        for index in range(100)
        for _ in range(2)
    ]
    _write_dataset(path, rows)
    dataset = NameDataset(path, chunk_size=17, test_fraction=0.2)

    train_names = {str(name) for batch in dataset.iter_split("train") for name in batch.names}
    test_names = {str(name) for batch in dataset.iter_split("test") for name in batch.names}

    assert train_names
    assert test_names
    assert train_names.isdisjoint(test_names)
    assert train_names | test_names == {row["name"] for row in rows}


def test_sample_is_deterministic_across_chunk_sizes(tmp_path: Path) -> None:
    path = tmp_path / "names.csv"
    rows = [
        {"name": f"synthetic complete name {index}", "sex": "f" if index % 2 else "m"}
        for index in range(500)
    ]
    _write_dataset(path, rows)

    def selected(chunk_size: int) -> set[str]:
        dataset = NameDataset(
            path,
            chunk_size=chunk_size,
            test_fraction=0.2,
            sample_fraction=0.25,
        )
        return {
            str(name)
            for split in ("train", "test")
            for batch in dataset.iter_split(split)
            for name in batch.names
        }

    assert selected(31) == selected(127)


def test_missing_or_invalid_columns_fail_fast(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    _write_dataset(missing, [{"name": "synthetic name"}])
    with pytest.raises(DatasetSchemaError, match="missing columns: sex"):
        NameDataset(missing, chunk_size=10, test_fraction=0.2)

    invalid = tmp_path / "invalid.csv"
    _write_dataset(invalid, [{"name": "synthetic name", "sex": "unknown"}])
    dataset = NameDataset(invalid, chunk_size=10, test_fraction=0.2)
    with pytest.raises(DatasetSchemaError, match="labels outside"):
        list(dataset.iter_split("train"))


def test_empty_names_are_skipped_without_creating_a_cleaned_dataset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "names.csv"
    _write_dataset(
        path,
        [
            {"name": "", "sex": "f"},
            {"name": "synthetic valid name", "sex": "m"},
        ],
    )
    dataset = NameDataset(path, chunk_size=10, test_fraction=0.5)

    selected = [
        str(name)
        for split in ("train", "test")
        for batch in dataset.iter_split(split)
        for name in batch.names
    ]

    assert selected == ["synthetic valid name"]
    assert list(tmp_path.iterdir()) == [path]


def test_partition_stream_selects_both_sides_in_one_pass(tmp_path: Path) -> None:
    path = tmp_path / "names.csv"
    rows = [
        {"name": "same full name", "sex": "f"},
        {"name": "different name", "sex": "m"},
        {"name": "same full name", "sex": "f"},
        {"name": "another name", "sex": "m"},
        {"name": "same full name", "sex": "f"},
    ]
    _write_dataset(path, rows)
    dataset = NameDataset(path, chunk_size=2, test_fraction=0.5)

    train_names: list[str] = []
    test_names: list[str] = []
    for partition in dataset.iter_partitions():
        if partition.train is not None:
            train_names.extend(partition.train.names.tolist())
        if partition.test is not None:
            test_names.extend(partition.test.names.tolist())

    assert ("same full name" in train_names) != ("same full name" in test_names)
    assert train_names.count("same full name") + test_names.count("same full name") == 3


def test_three_token_native_view_uses_native_groups_for_sampling_and_split(
    tmp_path: Path,
) -> None:
    path = tmp_path / "names.csv"
    rows = [
        {"name": "  KABONGO   ILUNGA Jean ", "sex": "m"},
        {"name": "kabongo ilunga marie", "sex": "f"},
        {"name": "kavira mapendo esther", "sex": "f"},
        {"name": "two tokens", "sex": "m"},
    ]
    _write_dataset(path, rows)
    full = NameDataset(
        path,
        chunk_size=2,
        test_fraction=0.5,
        input_view="full",
        split_group_view="native_only",
        required_token_count=3,
    )
    native = NameDataset(
        path,
        chunk_size=3,
        test_fraction=0.5,
        input_view="native_only",
        split_group_view="native_only",
        required_token_count=3,
    )

    full_splits = {
        split: [str(name) for batch in full.iter_split(split) for name in batch.names]
        for split in ("train", "test")
    }
    native_splits = {
        split: [str(name) for batch in native.iter_split(split) for name in batch.names]
        for split in ("train", "test")
    }

    assert sum(map(len, full_splits.values())) == 3
    assert sum(map(len, native_splits.values())) == 3
    assert "two tokens" not in full_splits["train"] + full_splits["test"]
    assert all(
        name == name.lower() and "  " not in name for name in sum(full_splits.values(), [])
    )
    assert all(len(name.split()) == 2 for name in sum(native_splits.values(), []))
    assert ("kabongo ilunga jean" in full_splits["train"]) == (
        "kabongo ilunga marie" in full_splits["train"]
    )
