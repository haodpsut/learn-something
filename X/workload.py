# src/satgym/workload.py
from dataclasses import dataclass
import numpy as np

@dataclass
class Task:
    """Represents a single computation task."""
    data_size_mb: float  # Data to be uploaded (in Megabytes)
    compute_load_gflops: float # Required computation (in Giga-FLOPs)

class TaskGenerator:
    """Generates computation tasks."""
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def generate_task(self) -> Task:
        """Generates a single random task."""
        data_size = self.rng.uniform(1.0, 50.0) # 1 to 50 MB
        compute_load = self.rng.uniform(0.1, 5.0) # 0.1 to 5 GFLOPs
        return Task(data_size_mb=data_size, compute_load_gflops=compute_load)