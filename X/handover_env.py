import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, Optional

from satgym.backend import StarPerfBackend
from satgym.entities import MobileUser

logger = logging.getLogger(__name__)

class HandoverEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.config = {
            "constellation_name": "Starlink",
            "simulation_steps": 1000, # Chuyến bay dài hơn
            "num_candidate_satellites": 4, # Số vệ tinh ứng viên
            "handover_penalty": -1.0, # Phạt cho mỗi lần handover
            "downtime_penalty": -50.0, # Phạt nặng khi mất kết nối
        }
        self.config.update(kwargs)
        
        logger.info("--- Initializing SatGym-Handover-v0 ---")
        
        # Khởi tạo backend (cần logic tính dT giống RoutingEnv)
        self.backend = self._initialize_backend()
        
        # Tạo người dùng di động (ví dụ: chuyến bay New York -> London)
        self.user = MobileUser(
            start_pos=(40.7, -74.0), # (lat, lon) New York
            end_pos=(51.5, -0.1),   # London
            total_steps=self.config["simulation_steps"],
            name="Flight-A001"
        )

        # --- Định nghĩa Action & Observation Space ---
        k = self.config["num_candidate_satellites"]
        self.action_space = spaces.Discrete(k + 1) # 0=Stay, 1..k = Handover to candidate k
        # Observation: [chất lượng_hiện_tại, q_ứng_viên_1, ..., q_ứng_viên_k]
        self.observation_space = spaces.Box(low=0, high=1, shape=(k + 1,), dtype=np.float32)

        self.current_satellite = None
        self.time_step = 0
        
        logger.info("--- Initialization Complete ---")

    def _initialize_backend(self):
        # Tái sử dụng logic tính dT từ RoutingEnv
        temp_config = self.config.copy()
        temp_config['dT'] = 1000 
        temp_backend = StarPerfBackend(temp_config)
        orbit_cycle = temp_backend.shell.orbit_cycle
        
        target_steps = self.config['simulation_steps']
        final_dT = orbit_cycle // (target_steps - 1) if target_steps > 1 else orbit_cycle
        logger.info(f"Calculated dT={final_dT} to achieve ~{target_steps} steps.")
        
        final_config = self.config.copy()
        final_config['dT'] = final_dT
        return StarPerfBackend(final_config)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        logger.info("Resetting HandoverEnv...")
        
        self.time_step = 1
        self.user.reset()
        
        # Tìm vệ tinh tốt nhất ban đầu để kết nối
        best_sats = self.backend.find_k_best_satellites(self.user, self.time_step, 1)
        if not best_sats:
            # Trường hợp hiếm: không có vệ tinh nào trong tầm nhìn
            logger.error("No satellites in view at the start of the episode. Cannot reset.")
            # Cần xử lý tốt hơn, nhưng tạm thời reset lại
            return self.reset(seed=seed)

        self.current_satellite = best_sats[0]
        
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        self.time_step += 1
        self.user.move()
        
        # Lấy danh sách các vệ tinh ứng viên tại thời điểm HIỆN TẠI
        k = self.config["num_candidate_satellites"]
        candidate_sats = self.backend.find_k_best_satellites(self.user, self.time_step, k)
        
        reward = 0
        # Thực thi hành động
        if action == 0: # Giữ nguyên kết nối
            # Không có hình phạt handover
            pass
        elif action > 0 and (action - 1) < len(candidate_sats): # Handover
            self.current_satellite = candidate_sats[action - 1]
            reward += self.config["handover_penalty"]
        else: # Hành động không hợp lệ
            # Coi như mất kết nối
            self.current_satellite = None
        
        # Tính toán phần thưởng dựa trên kết nối mới (hoặc cũ)
        if self.current_satellite:
            link_quality = self.backend.get_link_quality(self.user, self.current_satellite, self.time_step)
            if link_quality > 0:
                reward += link_quality # Thưởng bằng chất lượng liên kết
            else:
                reward += self.config["downtime_penalty"] # Phạt nếu mất kết nối
        else:
            reward += self.config["downtime_penalty"]

        terminated = self.time_step >= self.config["simulation_steps"]
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()
        
    def _get_observation(self):
        k = self.config["num_candidate_satellites"]
        
        # Lấy chất lượng liên kết hiện tại
        if self.current_satellite:
            current_quality = self.backend.get_link_quality(self.user, self.current_satellite, self.time_step)
        else:
            current_quality = 0.0

        # Lấy chất lượng của các vệ tinh ứng viên
        candidate_sats = self.backend.find_k_best_satellites(self.user, self.time_step, k)
        candidate_qualities = [self.backend.get_link_quality(self.user, sat, self.time_step) for sat in candidate_sats]
        
        # Pad với 0 nếu không đủ ứng viên
        padding = [0.0] * (k - len(candidate_qualities))
        
        obs = np.array([current_quality] + candidate_qualities + padding, dtype=np.float32)
        return obs

    def _get_info(self):
        return {
            "time_step": self.time_step,
            "user_lat": self.user.latitude,
            "user_lon": self.user.longitude,
            "current_sat_id": self.current_satellite.id if self.current_satellite else -1,
        }

    def close(self):
        pass