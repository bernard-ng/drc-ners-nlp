from pydantic import BaseModel


class LLMConfig(BaseModel):
    """LLM annotation configuration"""

    model_name: str = "mistral:7b"
    requests_per_minute: int = 60
    requests_per_second: int = 2
    retry_attempts: int = 3
    timeout_seconds: int = 30
    max_concurrent_requests: int = 2
    enable_rate_limiting: bool = False
