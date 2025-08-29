# src/satgym/demand.py
import numpy as np
from typing import List, Dict

class DemandGenerator:
    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)

    def generate_demand_map(self, cell_ids: List[str]) -> Dict[str, float]:
        """

        Generates a demand value for each cell, with some hotspots.
        """
        demand_map = {}
        # Tạo 1-3 điểm nóng ngẫu nhiên
        num_hotspots = self.rng.integers(1, 4)
        hotspot_indices = self.rng.choice(len(cell_ids), num_hotspots, replace=False)
        
        for i, cell_id in enumerate(cell_ids):
            # Nhu cầu cơ bản
            base_demand = self.rng.uniform(0.1, 0.5) # Gbps
            
            # Nếu là điểm nóng, tăng nhu cầu lên
            if i in hotspot_indices:
                base_demand *= self.rng.uniform(5, 10)
            
            demand_map[cell_id] = base_demand
        
        return demand_map