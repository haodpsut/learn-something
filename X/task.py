import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, Optional
from satgym import STARPERF_PATH

from satgym.backend import StarPerfBackend
from satgym.entities import MobileUser
from satgym.workload import TaskGenerator, Task

logger = logging.getLogger(__name__)

class TaskOffloadingEnv(gym.Env):
    """
    An environment for simulating the task offloading decision process.
    The agent decides where to execute a computational task to minimize latency or energy.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, **kwargs):
        super().__init__()

        self.config = {
            "constellation_name": "Starlink",
            "simulation_steps": 100,
            "cost_objective": "latency", # "latency" hoặc "energy"
        }
        self.config.update(kwargs)

        logger.info("--- Initializing SatGym-TaskOffloading-v0 ---")

        # Khởi tạo backend (sử dụng logic tính dT an toàn)
        self.backend = self._initialize_backend()

        # Tạo người dùng di động (vị trí cố định cho bài toán này)
        self.user = MobileUser(
            start_pos=(51.5, -0.1), # London
            end_pos=(51.5, -0.1),   # Không di chuyển
            total_steps=self.config["simulation_steps"],
            name="StaticTerminal-01"
        )

        # Tạo bộ phát tác vụ
        self.task_generator = TaskGenerator()
        self.current_task: Optional[Task] = None
        self.serving_satellite = None

        # --- Định nghĩa Action & Observation Space ---
        self.action_space = spaces.Discrete(3) # 0: Local, 1: Edge (Satellite), 2: Cloud (Ground)

        # Observation: [data_size (norm), compute_load (norm), link_quality_to_sat, sat_cpu_load (norm)]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

        self.time_step = 0
        logger.info("--- Initialization Complete ---")

    def _initialize_backend(self):
        # Tái sử dụng logic tính dT an toàn
        temp_config = self.config.copy()
        temp_config['dT'] = 1000
        temp_backend = StarPerfBackend(starperf_path=STARPERF_PATH, config=temp_config)

        orbit_cycle = temp_backend.shell.orbit_cycle

        target_steps = self.config['simulation_steps']
        final_dT = orbit_cycle // (target_steps - 1) if target_steps > 1 else orbit_cycle
        logger.info(f"Calculated dT={final_dT} to achieve ~{target_steps} steps.")

        final_config = self.config.copy()
        final_config['dT'] = final_dT

        final_backend = StarPerfBackend(starperf_path=STARPERF_PATH, config=final_config)
        return final_backend

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        logger.info("Resetting TaskOffloadingEnv...")

        self.time_step = 1
        self.user.reset()
        # Reset tải CPU của tất cả vệ tinh (để mỗi episode bắt đầu như mới)
        for sat in self.backend.sat_id_map.values():
            sat.cpu_load = 0.0

        # Lấy tác vụ đầu tiên
        self._get_next_task()

        return self._get_observation(), self._get_info()

    def _get_next_task(self):
        """Lấy tác vụ mới và xác định vệ tinh phục vụ."""
        self.current_task = self.task_generator.generate_task()
        # Trong bài toán này, chúng ta giả định chỉ có 1 vệ tinh tốt nhất phục vụ user
        best_sats = self.backend.find_k_best_satellites(self.user, self.time_step, 1)
        if not best_sats:
            logger.warning(f"No satellite in view for user at step {self.time_step}")
            self.serving_satellite = None
        else:
            self.serving_satellite = best_sats[0]

    def _get_observation(self):
        if self.current_task is None or self.serving_satellite is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Chuẩn hóa các giá trị để nằm trong khoảng [0, 1]
        # Giả định các giá trị max để chuẩn hóa
        max_data_size = 50.0 # MB
        max_compute_load = 5.0 # GFLOPs

        data_size_norm = self.current_task.data_size_mb / max_data_size
        compute_load_norm = self.current_task.compute_load_gflops / max_compute_load

        link_quality = self.backend.get_link_quality(self.user, self.serving_satellite, self.time_step)
        sat_cpu_load = self.serving_satellite.cpu_load

        obs = np.array([
            data_size_norm,
            compute_load_norm,
            link_quality,
            sat_cpu_load
        ], dtype=np.float32)

        return np.clip(obs, 0, 1)

    def _get_info(self):
        return {
            "time_step": self.time_step,
            "task_data_size_mb": self.current_task.data_size_mb if self.current_task else 0,
            "task_compute_load_gflops": self.current_task.compute_load_gflops if self.current_task else 0,
            "serving_sat_id": self.serving_satellite.id if self.serving_satellite else -1,
        }

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("Invalid action")

        if self.serving_satellite is None:
            # Không có vệ tinh nào để offload, buộc phải xử lý local
            action = 0

        # Tính toán chi phí cho tất cả các lựa chọn
        costs = self.backend.get_offloading_costs(self.user, self.serving_satellite, self.current_task, self.time_step)

        decision_map = {0: 'local', 1: 'edge', 2: 'cloud'}
        chosen_decision = decision_map[action]

        # Phần thưởng là chi phí âm (latency hoặc energy)
        cost = costs[chosen_decision][self.config['cost_objective']]
        reward = -cost

        # Cập nhật trạng thái (ví dụ: tải CPU của vệ tinh nếu offload lên edge)
        if chosen_decision == 'edge':
            # Mô hình đơn giản: tải CPU tăng lên trong suốt thời gian tính toán
            # Một mô hình phức tạp hơn sẽ cần một event scheduler.
            self.serving_satellite.cpu_load = 1.0 # Giả định vệ tinh bị chiếm dụng hoàn toàn

        self.time_step += 1

        # Trong môi trường này, mỗi bước là một quyết định, nên nó không "terminated"
        # mà chỉ "truncated" khi hết thời gian.
        terminated = False
        truncated = self.time_step >= self.config["simulation_steps"]

        # Lấy tác vụ tiếp theo cho bước sau
        if not truncated:
            self._get_next_task()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def close(self):
        pass