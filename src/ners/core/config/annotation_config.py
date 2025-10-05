from pydantic import BaseModel


class NERConfig(BaseModel):
    """NER annotation configuration"""

    model_name: str = "drc_names_ner"
    retry_attempts: int = 3


class LLMConfig(BaseModel):
    """LLM annotation configuration"""

    model_name: str = "mistral:7b"
    requests_per_minute: int = 60
    requests_per_second: int = 2
    retry_attempts: int = 3
    timeout_seconds: int = 30
    max_concurrent_requests: int = 2
    enable_rate_limiting: bool = False


class AnnotationConfig(BaseModel):
    """Base class for annotation configurations"""

    llm: LLMConfig = LLMConfig()
    ner: NERConfig = NERConfig()

    class Config:
        arbitrary_types_allowed = True
