import threading
import time
from dataclasses import dataclass
from queue import Queue


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting LLM requests"""

    requests_per_minute: int = 60
    requests_per_second: int = 2
    burst_limit: int = 5


class RateLimiter:
    """Thread-safe rate limiter for LLM requests"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times = Queue()
        self.lock = threading.Lock()
        self.last_request_time = 0

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            current_time = time.time()

            # Check requests per second limit
            time_since_last = current_time - self.last_request_time
            min_interval = 1.0 / self.config.requests_per_second

            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
                current_time = time.time()

            # Clean old request times (older than 1 minute)
            while not self.request_times.empty():
                if current_time - self.request_times.queue[0] > 60:
                    self.request_times.get()
                else:
                    break

            # Check requests per minute limit
            if self.request_times.qsize() >= self.config.requests_per_minute:
                oldest_request = self.request_times.queue[0]
                wait_time = 60 - (current_time - oldest_request)
                if wait_time > 0:
                    time.sleep(wait_time)
                    current_time = time.time()

            # Record this request
            self.request_times.put(current_time)
            self.last_request_time = current_time
