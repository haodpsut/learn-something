import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, Optional, List

from satgym.backend import StarPerfBackend, User
from satgym.traffic import TrafficGenerator
from satgym import STARPERF_PATH

logger = logging.getLogger(__name__)

class ResourceAllocationEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, **kwargs):
        super().__init__()

        self.config = {
            "constellation_name": "Starlink",
            "simulation_steps": 100,
            "k_shortest_paths": 5,
            "reward_success_factor": 1.0, # Nhân với lượng băng thông được cấp
            "reward_failure": -0.1, # Phạt nhẹ khi từ chối
        }
        self.config.update(kwargs)

        logger.info("--- Initializing SatGym-ResourceAllocation-v0 ---")

        # Khởi tạo backend (cần logic tính dT giống RoutingEnv)
        # Tạm thời đơn giản hóa để kiểm tra
        backend_config = self.config.copy()
        backend_config['dT'] = 57 # Tạm thời hard-code, có thể cải thiện sau
        self.backend = StarPerfBackend(starperf_path=STARPERF_PATH, config=backend_config)

        self._setup_ground_stations()
        self.traffic_generator = TrafficGenerator(self.ground_stations)

        # --- NÂNG CẤP OBSERVATION SPACE ---
        self.action_space = spaces.Discrete(self.config["k_shortest_paths"])

        # Observation: 3 (yêu cầu) + K * 2 (features cho mỗi đường đi)
        # Feature mỗi đường đi: [số hop (chuẩn hóa), băng thông bottleneck (chuẩn hóa)]
        num_demand_features = 3
        num_path_features = 2
        obs_shape = num_demand_features + self.config["k_shortest_paths"] * num_path_features
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_shape,), dtype=np.float32)

        self.current_demand = None
        self.candidate_paths = []

        logger.info("--- Initialization Complete ---")


    def _setup_ground_stations(self):
        # ... (giữ nguyên)
        self.ground_stations = [
            User(51.5, -0.1, "London"), User(40.7, -74.0, "NewYork"),
            User(1.35, 103.8, "Singapore"), User(-33.8, 151.2, "Sydney")
        ]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        logger.info("Resetting ResourceAllocationEnv...")

        self.time_step = 1
        # Reset trạng thái băng thông trong backend
        self.backend.reset_bandwidth_state()

        # Lấy yêu cầu lưu lượng đầu tiên
        self._get_next_demand()

        return self._get_observation(), self._get_info()

    def _get_next_demand(self):
        """Lấy một yêu cầu mới và tính toán các đường đi ứng viên."""
        demands = self.traffic_generator.generate_demands(self.time_step)
        self.current_demand = demands[0]

        source_user = self.current_demand["source_user"]
        target_user = self.current_demand["target_user"]

        start_sat = self.backend.find_nearest_satellite(source_user, self.time_step)
        target_sat = self.backend.find_nearest_satellite(target_user, self.time_step)

        graph = self.backend.get_network_graph(self.time_step)
        self.candidate_paths = self.backend.find_k_shortest_paths(
            graph, start_sat.id, target_sat.id, self.config["k_shortest_paths"], weight=None # Theo số hop
        )
        # Nếu không tìm đủ K đường đi, lặp lại đường đi cuối cùng để action space luôn hợp lệ
        if len(self.candidate_paths) < self.config["k_shortest_paths"]:
            if not self.candidate_paths: # Không có đường đi nào
                # Xử lý trường hợp không thể kết nối
                self.candidate_paths = [[] for _ in range(self.config["k_shortest_paths"])]
            else:
                last_path = self.candidate_paths[-1]
                self.candidate_paths.extend([last_path] * (self.config["k_shortest_paths"] - len(self.candidate_paths)))

    def _get_observation(self):
        if self.current_demand is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        source_user = self.current_demand["source_user"]
        target_user = self.current_demand["target_user"]
        demand_gbps = self.current_demand["demand_gbps"]

        # 1. Feature của Yêu cầu (chuẩn hóa)
        source_id_norm = self.ground_stations.index(source_user) / len(self.ground_stations)
        target_id_norm = self.ground_stations.index(target_user) / len(self.ground_stations)
        demand_norm = demand_gbps / self.backend.link_config["isl_capacity_gbps"]

        demand_features = [source_id_norm, target_id_norm, demand_norm]

        # 2. Feature của các Đường đi Ứng viên
        path_features = []
        for path in self.candidate_paths:
            if not path:
                # Trường hợp không có đường đi
                hops_norm = 1.0 # Số hop tối đa
                bottleneck_norm = 0.0 # Không có băng thông
            else:
                # Chuẩn hóa số hop (giả định max_hops là giới hạn trên)
                hops_norm = (len(path) - 1) / self.config.get("max_hops", 50)

                # Lấy và chuẩn hóa băng thông bottleneck
                bottleneck_gbps = self.backend.get_path_bottleneck_bandwidth(path, self.time_step)
                bottleneck_norm = bottleneck_gbps / self.backend.link_config["isl_capacity_gbps"]

            path_features.extend([hops_norm, bottleneck_norm])

        # Kết hợp thành một vector observation
        observation = np.concatenate([
            np.array(demand_features),
            np.array(path_features)
        ]).astype(np.float32)

        return np.clip(observation, 0.0, 1.0)



    def _get_info(self):
        return {
            "time_step": self.time_step,
            "current_demand": self.current_demand,
            "candidate_paths": self.candidate_paths
        }

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        path_to_try = self.candidate_paths[action]
        demand_gbps = self.current_demand["demand_gbps"]

        if not path_to_try: # Nếu không có đường đi
            success = False
        else:
            success = self.backend.allocate_path(path_to_try, demand_gbps, self.time_step)

        if success:
            reward = demand_gbps * self.config["reward_success_factor"]
        else:
            reward = self.config["reward_failure"]

        self.time_step += 1
        terminated = False
        truncated = self.time_step >= self.backend.simulation_steps

        if not truncated:
            self._get_next_demand()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def close(self):
        pass