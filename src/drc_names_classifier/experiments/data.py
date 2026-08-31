from __future__ import annotations

from dataclasses import dataclass
import polars as pl

from drc_names_classifier.config import ExperimentSettings
from drc_names_classifier.dataset import NameDataset
from drc_names_classifier.utils import NameView


@dataclass(frozen=True, slots=True)
class ExperimentDataset:
    """Materialized train/test frames shared by comparable architectures."""

    train: pl.DataFrame
    test: pl.DataFrame


class ExperimentDatasetStore:
    """Load and cache deterministic dataset splits for one experiment session."""

    def __init__(self, config: ExperimentSettings) -> None:
        self.config = config
        self._cache: dict[tuple[float, float, int | None, str], ExperimentDataset] = {}

    def load(
        self,
        *,
        sample_fraction: float,
        test_fraction: float,
        required_token_count: int | None = None,
        split_group_view: str = NameView.FULL.value,
    ) -> ExperimentDataset:
        key = (
            sample_fraction,
            test_fraction,
            required_token_count,
            split_group_view,
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        dataset = NameDataset(
            self.config.dataset_path,
            chunk_size=self.config.chunk_size,
            sample_fraction=sample_fraction,
            test_fraction=test_fraction,
            required_token_count=required_token_count,
            split_group_view=split_group_view,
        )
        train_frames: list[pl.DataFrame] = []
        test_frames: list[pl.DataFrame] = []
        for partition in dataset.iter_partitions():
            if partition.train is not None:
                train_frames.append(
                    pl.DataFrame({"name": partition.train.names, "sex": partition.train.labels})
                )
            if partition.test is not None:
                test_frames.append(
                    pl.DataFrame({"name": partition.test.names, "sex": partition.test.labels})
                )

        split = ExperimentDataset(
            train=self._concat(train_frames),
            test=self._concat(test_frames),
        )
        if split.train.is_empty() or split.test.is_empty():
            raise ValueError("Experiment split produced no usable rows")
        self._cache[key] = split
        return split

    @staticmethod
    def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
        if not frames:
            return pl.DataFrame(schema={"name": pl.String, "sex": pl.String})
        return pl.concat(frames, how="vertical", rechunk=True)
