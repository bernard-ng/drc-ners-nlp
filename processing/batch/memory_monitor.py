import gc
import logging

import psutil


class MemoryMonitor:
    """Monitor and manage memory usage during batch processing"""

    @staticmethod
    def get_memory_usage_mb() -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def cleanup_memory():
        """Force garbage collection"""
        gc.collect()

    @staticmethod
    def log_memory_usage(step_name: str):
        """Log current memory usage"""
        memory_mb = MemoryMonitor.get_memory_usage_mb()
        logging.info(f"Memory usage after {step_name}: {memory_mb:.1f} MB")
