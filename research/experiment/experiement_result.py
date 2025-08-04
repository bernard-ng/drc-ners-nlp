from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, List, Any

from research.experiment import ExperimentConfig, ExperimentStatus


@dataclass
class ExperimentResult:
    """Results from an experiment execution"""

    experiment_id: str
    config: ExperimentConfig

    # Execution metadata
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    error_message: Optional[str] = None

    # Model artifacts
    model_path: Optional[str] = None
    feature_extractor_path: Optional[str] = None

    # Metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    cv_metrics: Dict[str, float] = field(default_factory=dict)

    # Additional results
    confusion_matrix: Optional[List[List[int]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    prediction_examples: Optional[List[Dict]] = None

    # Data statistics
    train_size: int = 0
    test_size: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result["config"] = self.config.to_dict()
        result["start_time"] = self.start_time.isoformat()
        result["end_time"] = self.end_time.isoformat() if self.end_time else None
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary"""
        data["config"] = ExperimentConfig.from_dict(data["config"])
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        data["end_time"] = datetime.fromisoformat(data["end_time"]) if data["end_time"] else None
        data["status"] = ExperimentStatus(data["status"])
        return cls(**data)
