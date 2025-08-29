import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

from satgym.backend import StarPerfBackend
from satgym.entities import MobileUser
from satgym.workload import TaskGenerator

logger = logging.getLogger(__name__)

class TaskOffloadingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, **kwargs):
        super().__init__()
        logger.info("--- Initializing SatGym-TaskOffloading-v0 ---")
        
        # --- Định nghĩa Action & Observation Space ---
        self.action_space = spaces.Discrete(3) # 0: Local, 1: Edge, 2: Cloud
        
        # Observation: [data_size, compute_load, link_quality, sat_cpu_load]
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.observation_space.sample()
        info = {}
        return observation, info

    def step(self, action):
        observation = self.observation_space.sample()
        reward = -1.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info