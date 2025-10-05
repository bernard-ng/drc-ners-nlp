from dataclasses import dataclass


@dataclass
class BatchConfig:
    """Configuration for batch processing"""

    batch_size: int = 1000
    max_workers: int = 4
    checkpoint_interval: int = 5  # Save checkpoint every N batches
    use_multiprocessing: bool = (
        False  # Use ProcessPoolExecutor instead of ThreadPoolExecutor
    )
