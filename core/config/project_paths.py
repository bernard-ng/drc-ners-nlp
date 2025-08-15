from pathlib import Path

from pydantic import BaseModel, field_validator


class ProjectPaths(BaseModel):
    """Project directory structure configuration"""

    root_dir: Path
    data_dir: Path
    models_dir: Path
    outputs_dir: Path
    logs_dir: Path
    configs_dir: Path
    checkpoints_dir: Path

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    @field_validator("*", mode="before")
    def convert_to_path(cls, v):
        return Path(v) if not isinstance(v, Path) else v

    def get_data_path(self, filename: str) -> Path:
        return self.data_dir / filename
