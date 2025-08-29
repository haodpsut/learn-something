import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, Optional

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import to_parallel

from satgym import STARPERF_PATH
from satgym.backend import StarPerfBackend, User, Satellite

logger = logging.getLogger(__name__)

# --- Hàm helper để tạo môi trường theo chuẩn PettingZoo ---
def env(**kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = MultiAgentRoutingEnv(**kwargs)
    # OrderEnforcingWrapper raises an error if order of operations is violated
    env = wrappers.OrderEnforcingWrapper(env)
    return env
    
def raw_env(**kwargs):
    """
    To support the AEC API, the raw_env() function just uses the environment class wrapt in AECFromLegacy.
    This is so that people can convert the environment to parallel variants of the API as well.
    """
    return MultiAgentRoutingEnv(**kwargs)

class MultiAgentRoutingEnv(AECEnv):
    """
    A multi-agent environment for decentralized satellite routing.
    Each satellite is an independent agent.
    The environment conforms to the PettingZoo AECEnv API.
    """
    metadata = {
        "render_modes": ["human"],
        "name": "SatGym-MultiAgentRouting-v0",
        "is_parallelizable": False
    }

    def __init__(self, **kwargs):
        super().__init__()
        
        self.config = {
            "constellation_name": "Starlink",
            "simulation_steps": 100,
            "max_hops": 50,
            "reward_success": 100.0,
            "reward_failure": -100.0,
            "reward_per_hop": -1.0,
            "distance_reward_factor": 1000.0
        }
        self.config.update(kwargs)
        
        logger.info("--- Initializing SatGym-MultiAgentRouting-v0 ---")
        
        self.backend = self._initialize_backend()
        self.max_simulation_steps = self.backend.simulation_steps
        
        # --- Định nghĩa Agent ---
        self.possible_agents = [f"satellite_{i}" for i in range(1, self.backend.num_satellites + 1)]
        self.agent_id_map = {name: i for i, name in enumerate(self.possible_agents, 1)}
        self.agent_name_map = {i: name for name, i in self.agent_id_map.items()}

        # --- Định nghĩa Action/Observation Space cho mỗi Agent ---
        self.MAX_NEIGHBORS = 6
        obs_shape = 3 + self.MAX_NEIGHBORS * 4
        
        self._action_spaces = {agent: spaces.Discrete(self.MAX_NEIGHBORS) for agent in self.possible_agents}
        self._observation_spaces = {agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_shape,), dtype=np.float32) for agent in self.possible_agents}
        
        self._setup_ground_stations()
        self.packet = {} # Stores info about the packet being routed

    def _initialize_backend(self):
        # Tái sử dụng logic tính dT an toàn
        temp_config = self.config.copy(); temp_config['dT'] = 1000 
        temp_backend = StarPerfBackend(starperf_path=STARPERF_PATH, config=temp_config)
        orbit_cycle = temp_backend.shell.orbit_cycle
        target_steps = self.config['simulation_steps']
        final_dT = orbit_cycle // (target_steps - 1) if target_steps > 1 else orbit_cycle
        logger.info(f"Calculated dT={final_dT} to achieve ~{target_steps} steps.")
        final_config = self.config.copy(); final_config['dT'] = final_dT
        return StarPerfBackend(starperf_path=STARPERF_PATH, config=final_config)

    def _setup_ground_stations(self):
        self.ground_stations = [
            User(51.5, -0.1, "London"), User(40.7, -74.0, "NewYork"),
            User(1.35, 103.8, "Singapore"), User(-33.8, 151.2, "Sydney")
        ]

    # PettingZoo API requires observation_space and action_space to be methods
    def observation_space(self, agent): return self._observation_spaces[agent]
    def action_space(self, agent): return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.time_step = 1
        self.hop_count = 0
        
        source_user, target_user = np.random.choice(self.ground_stations, 2, replace=False)
        start_sat = self.backend.find_nearest_satellite(source_user, self.time_step)
        target_sat = self.backend.find_nearest_satellite(target_user, self.time_step)
        
        self.packet = { "current_sat_id": start_sat.id, "target_sat_id": target_sat.id }
        
        # Khởi tạo agent selector và đặt agent đầu tiên
        self._agent_selector = agent_selector.agent_selector(self.agents) # Sửa lỗi API
        self.agent_selection = self._agent_selector.reset()
        self.agent_selection = self.agent_name_map[start_sat.id]

    def step(self, action: int):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent_name = self.agent_selection
        agent_id = self.agent_id_map[agent_name]
        
        # Đặt lại reward tích lũy cho agent sắp hành động
        self._cumulative_rewards[agent_name] = 0
        self.rewards = {agent: 0 for agent in self.agents}
        
        self.hop_count += 1
        safe_time_step = min(self.time_step, self.max_simulation_steps)
        
        neighbors = self.backend.get_neighbors(agent_id, safe_time_step)
        
        if action >= len(neighbors):
            self._terminate_episode(self.config["reward_failure"])
        else:
            next_sat = neighbors[action]
            self.packet["current_sat_id"] = next_sat.id
            
            # Chỉ agent hành động nhận reward per hop
            self.rewards[agent_name] = self.config["reward_per_hop"]
            
            if next_sat.id == self.packet["target_sat_id"]:
                self._terminate_episode(self.config["reward_success"])
            elif self.hop_count >= self.config["max_hops"]:
                self._terminate_episode(self.config["reward_failure"])

        # Chuyển lượt cho agent tiếp theo (vệ tinh đang giữ gói tin)
        if not self.terminations[agent_name] and not self.truncations[agent_name]:
            self.time_step += 1
            if self.time_step > self.max_simulation_steps:
                self._truncate_episode()
            else:
                self.agent_selection = self.agent_name_map[self.packet['current_sat_id']]
        
        # Tích lũy reward (AECEnv yêu cầu)
        self._accumulate_rewards()

    def observe(self, agent: str):
        """Returns the observation for the specified agent."""
        agent_id = self.agent_id_map[agent]
        current_sat = self.backend.sat_id_map[agent_id]
        target_sat = self.backend.sat_id_map[self.packet["target_sat_id"]]

        # Tái sử dụng logic từ RoutingEnv._get_observation
        pos_current = self.backend.get_satellite_position(current_sat, self.time_step)
        pos_target = self.backend.get_satellite_position(target_sat, self.time_step)
        
        direction_to_target = pos_target - pos_current
        norm = np.linalg.norm(direction_to_target)
        if norm > 0: direction_to_target /= norm
        
        safe_time_step = min(self.time_step, self.max_simulation_steps)
        neighbors = self.backend.get_neighbors(agent_id, safe_time_step)
        
        neighbor_features = []
        for i in range(self.MAX_NEIGHBORS):
            if i < len(neighbors):
                neighbor_sat = neighbors[i]
                pos_neighbor = self.backend.get_satellite_position(neighbor_sat, self.time_step)
                direction = pos_target - pos_neighbor
                norm_n = np.linalg.norm(direction)
                if norm_n > 0: direction /= norm_n
                neighbor_features.extend([1.0] + direction.tolist())
            else:
                neighbor_features.extend([0.0, 0.0, 0.0, 0.0])

        observation = np.concatenate([direction_to_target, np.array(neighbor_features)]).astype(np.float32)
        return np.clip(observation, -1.0, 1.0)
    
    def _terminate_episode(self, final_reward):
        # Gán reward toàn cục cho tất cả agent
        for ag in self.agents:
            self.rewards[ag] = final_reward
            self.terminations[ag] = True
    
    def _truncate_episode(self):
        for ag in self.agents:
            self.truncations[ag] = True
            
    def close(self):
        pass