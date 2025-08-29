import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, Optional

from satgym import STARPERF_PATH
from satgym.backend import StarPerfBackend
from satgym.demand import DemandGenerator

logger = logging.getLogger(__name__)

class BeamHoppingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.config = {
            "constellation_name": "Starlink",
            "simulation_steps": 200,
            "num_beams": 8, # Số búp sóng của vệ tinh
            "beam_capacity_gbps": 2.0, # Năng lực của mỗi búp sóng
        }
        self.config.update(kwargs)
        
        logger.info("--- Initializing SatGym-BeamHopping-v0 ---")
        
        # Khởi tạo backend (sử dụng logic tính dT an toàn)
        self.backend = self._initialize_backend()
        self.demand_generator = DemandGenerator()

        # Chọn một vệ tinh để theo dõi trong suốt episode
        self.satellite_to_track = self.backend.sat_id_map[1] # Theo dõi vệ tinh ID 1
        
        # --- Định nghĩa Action & Observation Space ---
        # Lấy một danh sách ô mẫu để xác định kích thước không gian
        self.visible_cells = self.backend.get_visible_ground_cells(self.satellite_to_track, 1)
        self.num_cells = len(self.visible_cells)
        self.cell_map = {cell_id: i for i, cell_id in enumerate(self.visible_cells)}

        # Action: Gán mỗi búp sóng cho một chỉ số ô
        self.action_space = spaces.MultiDiscrete([self.num_cells] * self.config["num_beams"])
        
        # Observation: Vector nhu cầu của các ô có thể nhìn thấy
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.num_cells,), dtype=np.float32)

        self.time_step = 0
        logger.info(f"--- Initialization Complete. Tracking {self.num_cells} cells. ---")

    def _initialize_backend(self):
        # ... (Tái sử dụng logic tính dT từ các môi trường trước)
        temp_config = self.config.copy(); temp_config['dT'] = 1000
        temp_backend = StarPerfBackend(starperf_path=STARPERF_PATH, config=temp_config)
        orbit_cycle = temp_backend.shell.orbit_cycle
        target_steps = self.config['simulation_steps']
        final_dT = orbit_cycle // (target_steps - 1) if target_steps > 1 else orbit_cycle
        final_config = self.config.copy(); final_config['dT'] = final_dT
        return StarPerfBackend(starperf_path=STARPERF_PATH, config=final_config)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.time_step = 1
        return self._get_observation(), self._get_info()

    def step(self, action):
        # 1. Lấy trạng thái hiện tại
        demand_map = self._get_demand_map()

        # 2. Tính toán năng lực được phân bổ cho mỗi ô
        cell_capacity = np.zeros(self.num_cells)
        for beam_idx, cell_idx in enumerate(action):
            if cell_idx < self.num_cells: # Đảm bảo action hợp lệ
                cell_capacity[cell_idx] += self.config["beam_capacity_gbps"]
        
        # 3. Tính toán phần thưởng (tổng thông lượng phục vụ)
        demand_vector = np.array(list(demand_map.values()))
        served_throughput_vector = np.minimum(demand_vector, cell_capacity)
        reward = np.sum(served_throughput_vector)

        self.time_step += 1
        terminated = self.time_step >= self.config["simulation_steps"]
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()
        
    def _get_demand_map(self) -> Dict[str, float]:
        """Lấy bản đồ nhu cầu cho các ô có thể nhìn thấy hiện tại."""
        self.visible_cells = self.backend.get_visible_ground_cells(self.satellite_to_track, self.time_step)
        # Cập nhật lại map nếu số lượng ô thay đổi
        if len(self.visible_cells) != self.num_cells:
            self.num_cells = len(self.visible_cells)
            self.cell_map = {cell_id: i for i, cell_id in enumerate(self.visible_cells)}
        
        return self.demand_generator.generate_demand_map(self.visible_cells)

    def _get_observation(self):
        """Tạo vector observation từ bản đồ nhu cầu."""
        demand_map = self._get_demand_map()
        
        # Sắp xếp nhu cầu theo đúng thứ tự của cell_map
        obs = np.zeros(self.num_cells, dtype=np.float32)
        for cell_id, demand in demand_map.items():
            if cell_id in self.cell_map:
                obs[self.cell_map[cell_id]] = demand
        
        return obs

    def _get_info(self):
        return {"time_step": self.time_step, "num_visible_cells": self.num_cells}

    def close(self):
        pass