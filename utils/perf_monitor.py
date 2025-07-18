from contextlib import contextmanager
import time

class Timer:
    def __init__(self, name):
        self.name = name
        self.collects = []
        
    def reset(self):
        """Reset the timer."""
        self.collects = []
        
    def average(self):
        """Calculate the average time from collected times."""
        if len(self.collects) == 0:
            return 0
        return sum(self.collects) / len(self.collects)
        
    def __enter__(self):
        self.start_time = time.monotonic()
        
    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.monotonic()
        elapsed_time = end_time - self.start_time
        self.collects.append(elapsed_time)