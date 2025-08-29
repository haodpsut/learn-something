# src/satgym/envs/routing_env.py

import random
import logging
from typing import Dict, Optional
from satgym import STARPERF_PATH

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- Import các thành phần của SatGym ---
from satgym.backend import StarPerfBackend
from satgym.backend import User

logger = logging.getLogger(__name__)

class RoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, **kwargs):
        super().__init__()

        config = {
            "constellation_name": "Starlink",
            "simulation_steps": 100,
            "max_hops": 50,
            "reward_success": 100.0,
            "reward_failure": -100.0,
            "reward_per_hop": -1.0,
            "distance_reward_factor": 1000.0
        }
        config.update(kwargs)
        self.config = config

        logger.info("--- Initializing SatGym-Routing-v0 Environment ---")

        # --- LOGIC MỚI SẠCH SẼ HƠN ---
        # 1. Tính toán dT cần thiết
        # Chúng ta vẫn cần chạy tạm thời để lấy orbit_cycle, nhưng logic này có thể được cải thiện sau.
        # Hiện tại, giữ nguyên để đảm bảo hoạt động.
        temp_config = self.config.copy()
        temp_config['dT'] = 1000
        temp_backend = StarPerfBackend(starperf_path=STARPERF_PATH, config=temp_config)


        orbit_cycle = temp_backend.shell.orbit_cycle

        target_steps = self.config['simulation_steps']
        self.config['dT'] = orbit_cycle // (target_steps - 1) if target_steps > 1 else orbit_cycle
        logger.info(f"Orbit cycle is {orbit_cycle}s. Calculated dT={self.config['dT']} to achieve ~{target_steps} steps.")

        # 2. Khởi tạo Backend chính thức
        self.backend = StarPerfBackend(starperf_path=STARPERF_PATH, config=self.config)

        self.max_simulation_steps = self.backend.simulation_steps

        # --- Phần còn lại của __init__ (giữ nguyên) ---
        self.MAX_NEIGHBORS = 6
        self.action_space = spaces.Discrete(self.MAX_NEIGHBORS)
        obs_shape = 3 + self.MAX_NEIGHBORS * 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_shape,), dtype=np.float32)

        self._setup_ground_stations()
        logger.info(f"--- Initialization Complete for {self.backend.num_satellites} satellites ---")

    def _setup_ground_stations(self):
        self.ground_stations = [
            User(51.5, -0.1, "London"), User(40.7, -74.0, "NewYork"),
            User(1.35, 103.8, "Singapore"), User(-33.8, 151.2, "Sydney")
        ]


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.time_step = 1
        self.hop_count = 0
        self.source_user, self.target_user = self.np_random.choice(self.ground_stations, 2, replace=False)
        self.start_sat = self.backend.find_nearest_satellite(self.source_user, self.time_step)
        self.target_sat = self.backend.find_nearest_satellite(self.target_user, self.time_step)
        self.current_sat = self.start_sat
        logger.info(f"Resetting episode: {self.source_user.user_name} -> {self.target_user.user_name} | Path: Sat-{self.start_sat.id} -> Sat-{self.target_sat.id}")
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        self.hop_count += 1
        safe_time_step = min(self.time_step, self.max_simulation_steps)

        # --- REWARD SHAPING: BƯỚC 1 ---
        # Lấy vị trí và tính khoảng cách đến đích TRƯỚC khi di chuyển
        pos_current_before = self.backend.get_satellite_position(self.current_sat, safe_time_step)
        pos_target = self.backend.get_satellite_position(self.target_sat, safe_time_step)
        distance_before = np.linalg.norm(pos_target - pos_current_before)

        neighbors = self.backend.get_neighbors(self.current_sat.id, safe_time_step)

        terminated = False
        reward = self.config["reward_per_hop"] # Bắt đầu với chi phí mỗi bước

        if action >= len(neighbors):
            logger.warning(f"Invalid action {action} for {len(neighbors)} neighbors at t={self.time_step}. Packet dropped.")
            reward += self.config["reward_failure"] # Cộng thêm hình phạt thất bại
            terminated = True
        else:
            self.current_sat = neighbors[action]
            # --- REWARD SHAPING: BƯỚC 2 ---
            # Tính khoảng cách đến đích SAU khi di chuyển và tính reward
            pos_current_after = self.backend.get_satellite_position(self.current_sat, safe_time_step)
            distance_after = np.linalg.norm(pos_target - pos_current_after)

            # Phần thưởng là sự cải thiện về khoảng cách, đã được chuẩn hóa
            shaping_reward = (distance_before - distance_after) / self.config["distance_reward_factor"]
            reward += shaping_reward

        # --- Xử lý các điều kiện kết thúc ---
        if not terminated and self.current_sat.id == self.target_sat.id:
            logger.info(f"Target satellite {self.target_sat.id} reached in {self.hop_count} hops!")
            reward += self.config["reward_success"]
            terminated = True
        elif not terminated and self.hop_count >= self.config["max_hops"]:
            logger.warning(f"Max hops {self.config['max_hops']} exceeded. Terminating episode.")
            reward += self.config["reward_failure"]
            terminated = True

        self.time_step += 1
        truncated = self.time_step > self.max_simulation_steps
        if truncated:
            logger.warning(f"Simulation time {self.max_simulation_steps} steps exceeded. Truncating episode.")
            terminated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        # ... (Hàm này giữ nguyên) ...
        pos_current = self.backend.get_satellite_position(self.current_sat, self.time_step)
        pos_target = self.backend.get_satellite_position(self.target_sat, self.time_step)
        direction_to_target = pos_target - pos_current
        norm = np.linalg.norm(direction_to_target)
        if norm > 0: direction_to_target /= norm
        safe_time_step = min(self.time_step, self.max_simulation_steps)
        neighbors = self.backend.get_neighbors(self.current_sat.id, safe_time_step)
        neighbor_features = []
        for i in range(self.MAX_NEIGHBORS):
            if i < len(neighbors):
                neighbor_sat = neighbors[i]
                pos_neighbor = self.backend.get_satellite_position(neighbor_sat, self.time_step)
                direction_from_neighbor_to_target = pos_target - pos_neighbor
                norm_n = np.linalg.norm(direction_from_neighbor_to_target)
                if norm_n > 0: direction_from_neighbor_to_target /= norm_n
                neighbor_features.extend([1.0] + direction_from_neighbor_to_target.tolist())
            else:
                neighbor_features.extend([0.0, 0.0, 0.0, 0.0])
        observation = np.concatenate([direction_to_target, np.array(neighbor_features)]).astype(np.float32)
        if not self.observation_space.contains(observation):
            observation = np.clip(observation, -1.0, 1.0)
        return observation

    def _get_info(self):
        # ... (Hàm này giữ nguyên) ...
        return {"time_step": self.time_step, "hop_count": self.hop_count, "current_sat_id": self.current_sat.id, "target_sat_id": self.target_sat.id}

    def close(self):
        logger.info("Closing environment.")