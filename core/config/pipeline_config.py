from pydantic import BaseModel

from core.config.logging_config import LoggingConfig
from core.config.data_config import DataConfig
from core.config.llm_config import LLMConfig
from core.config.processing_config import ProcessingConfig
from core.config.project_paths import ProjectPaths


class PipelineConfig(BaseModel):
    """Main pipeline configuration"""

    name: str = "drc_names_pipeline"
    version: str = "1.0.0"
    description: str = "DRC Names NLP Processing Pipeline"

    paths: ProjectPaths
    stages: list[str] = []
    processing: ProcessingConfig = ProcessingConfig()
    llm: LLMConfig = LLMConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()

    # Environment-specific settings
    environment: str = "development"
    debug: bool = True

    class Config:
        arbitrary_types_allowed = True
