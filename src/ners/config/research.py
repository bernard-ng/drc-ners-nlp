from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ners.config.defaults import (
    DEFAULT_DATASET_PATH,
    DEFAULT_RESEARCH_MODELS_DIR,
    DEFAULT_RESEARCH_OUTPUTS_DIR,
    DEFAULT_RESEARCH_TEMPLATES_PATH,
)


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Shared inputs and artifact locations for comparable model experiments."""

    dataset_path: Path = DEFAULT_DATASET_PATH
    templates_path: Path = DEFAULT_RESEARCH_TEMPLATES_PATH
    models_dir: Path = DEFAULT_RESEARCH_MODELS_DIR
    outputs_dir: Path = DEFAULT_RESEARCH_OUTPUTS_DIR
    chunk_size: int = 100_000
    sample_fraction: float = 0.01
    test_fraction: float = 0.2

    def __post_init__(self) -> None:
        for field_name in ("dataset_path", "templates_path", "models_dir", "outputs_dir"):
            object.__setattr__(self, field_name, Path(getattr(self, field_name)))

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("sample_fraction must be between zero and one")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")

    @property
    def experiments_dir(self) -> Path:
        return self.outputs_dir / "experiments"

    @property
    def experiment_models_dir(self) -> Path:
        return self.models_dir / "experiments"
