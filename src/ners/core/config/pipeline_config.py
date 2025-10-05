from pydantic import BaseModel

from ners.core.config.annotation_config import AnnotationConfig
from ners.core.config.data_config import DataConfig
from ners.core.config.logging_config import LoggingConfig
from ners.core.config.processing_config import ProcessingConfig
from ners.core.config.project_paths import ProjectPaths


class PipelineConfig(BaseModel):
    """Main pipeline configuration"""

    name: str = "drc_names_pipeline"
    version: str = "1.0.0"
    description: str = "DRC Names NLP Processing Pipeline"

    paths: ProjectPaths
    stages: list[str] = []
    processing: ProcessingConfig = ProcessingConfig()
    annotation: AnnotationConfig = AnnotationConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()

    # Environment-specific settings
    environment: str = "development"
    debug: bool = True

    class Config:
        arbitrary_types_allowed = True
