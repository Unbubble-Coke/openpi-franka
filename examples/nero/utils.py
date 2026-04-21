import time
import logging

class FpsCounter:
    """Utility class to measure the frequency of a loop"""

    def __init__(self, name: str, log_interval: float = 1.0):
        self.name = name
        self.last_time = time.perf_counter()

    def update(self):
        """compute and log the frequency"""
        now = time.perf_counter()
        fps = 1.0 / (now - self.last_time)
        self.last_time = now
        logging.info(f"[DEBUG] {self.name} loop actual frequency: {fps:.2f} Hz")