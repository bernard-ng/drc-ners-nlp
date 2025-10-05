from dataclasses import field

from pydantic import BaseModel


class ProcessingConfig(BaseModel):
    """Data processing pipeline configuration"""

    batch_size: int = 1000
    max_workers: int = 4
    checkpoint_interval: int = 5
    use_multiprocessing: bool = False
    encoding_options: list = field(
        default_factory=lambda: ["utf-8", "utf-16", "latin1"]
    )
    chunk_size: int = 100_000
    epochs: int = 2
