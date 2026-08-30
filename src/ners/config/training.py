from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ners.config.defaults import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Configuration for memory-bounded full-corpus model training."""

    dataset_path: Path = DEFAULT_DATASET_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    metrics_path: Path | None = None
    name_column: str = "name"
    target_column: str = "sex"
    chunk_size: int = 100_000
    test_fraction: float = 0.2
    sample_fraction: float = 1.0
    epochs: int = 1
    n_features: int = 2**20
    ngram_min: int = 2
    ngram_max: int = 5
    alpha: float = 1e-6
    balanced: bool = False
    random_seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(self, "model_path", Path(self.model_path))
        if self.metrics_path is not None:
            object.__setattr__(self, "metrics_path", Path(self.metrics_path))

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be between zero and one")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        if self.n_features <= 0:
            raise ValueError("n_features must be greater than zero")
        if not 1 <= self.ngram_min <= self.ngram_max:
            raise ValueError("ngram_min must be positive and no larger than ngram_max")
        if self.alpha <= 0:
            raise ValueError("alpha must be greater than zero")

    @property
    def resolved_metrics_path(self) -> Path:
        if self.metrics_path is not None:
            return self.metrics_path
        return self.model_path.with_suffix(".metrics.json")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["dataset_path"] = str(self.dataset_path)
        values["model_path"] = str(self.model_path)
        values["metrics_path"] = str(self.resolved_metrics_path)
        return values
