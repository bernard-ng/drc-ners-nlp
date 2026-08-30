from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from ners.config import ExperimentConfig


class ExperimentStatus(StrEnum):
    """Lifecycle state persisted for a research run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ExperimentResult:
    """Serializable outcome and artifacts for one experiment."""

    experiment_id: str
    config: ExperimentConfig
    start_time: datetime
    end_time: datetime | None = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    error_message: str | None = None
    model_path: str | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)
    cv_metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    feature_importance: dict[str, float] | None = None
    train_size: int = 0
    test_size: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["config"] = self.config.to_dict()
        result["start_time"] = self.start_time.isoformat()
        result["end_time"] = self.end_time.isoformat() if self.end_time else None
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        values = dict(data)
        values["config"] = ExperimentConfig.from_dict(values["config"])
        values["start_time"] = datetime.fromisoformat(values["start_time"])
        values["end_time"] = (
            datetime.fromisoformat(values["end_time"]) if values.get("end_time") else None
        )
        values["status"] = ExperimentStatus(values["status"])
        return cls(**values)
