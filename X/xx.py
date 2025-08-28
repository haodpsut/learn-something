import sys
import os
from pathlib import Path
import random
import contextlib
import logging
from typing import List, Dict, Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- Cấu hình logging chuyên nghiệp ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- TÍCH HỢP STARPERF ---
STARPERF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "deps" / "StarPerf_Simulator"
if str(STARPERF_PATH.resolve()) not in sys.path:
    sys.path.insert(0, str(STARPERF_PATH.resolve()))

try:
    from src.constellation_generation.by_XML import constellation_configuration
    from src.XML_constellation.constellation_connectivity import connectivity_mode_plugin_manager
    from src.XML_constellation.constellation_entity.user import user as User
    from src.XML_constellation.constellation_entity.satellite import satellite as Satellite
    from src.XML_constellation.constellation_evaluation.exists_ISL.delay import distance_between_satellite_and_user
except ImportError as e:
    logger.critical(f"Failed to import StarPerf modules. Ensure StarPerf_Simulator is cloned in deps/. Error: {e}")
    raise

class StarPerfBackend:
    """
    Provides a clean API to interact with the StarPerf simulation backend.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.starperf_dir = STARPERF_PATH
        
        # --- KHÔI PHỤC CÁC DÒNG BỊ THIẾU ---
        self.constellation = self._initialize_constellation()
        self.shell = self.constellation.shells[0]
        self.num_satellites = self.shell.number_of_satellites
        self.sat_id_map: Dict[int, Satellite] = {sat.id: sat for orbit in self.shell.orbits for sat in orbit.satellites}
        # ------------------------------------

        # --- ĐẶT KHỐI DEBUG Ở ĐÂY (SAU KHI ĐÃ CÓ self.sat_id_map) ---
        print("\n--- DEBUG: DATA STRUCTURE VALIDATION ---")
        sample_sat = self.sat_id_map[1] # Lấy vệ tinh có ID là 1
        print(f"Configuration dT: {self.config['dT']}")
        print(f"Length of sample_sat.longitude: {len(sample_sat.longitude)}")
        print(f"Length of sample_sat.latitude: {len(sample_sat.latitude)}")
        if isinstance(sample_sat.altitude, list):
            print(f"Length of sample_sat.altitude: {len(sample_sat.altitude)}")
        else:
            print(f"Type of sample_sat.altitude: {type(sample_sat.altitude)}")
        
        if hasattr(sample_sat, 'ISL'):
             print(f"Length of sample_sat.ISL: {len(sample_sat.ISL)}")
        print("--- END DEBUG ---\n")
        # ----------------------------------------

    @contextlib.contextmanager
    def _as_current_dir(self):
        prev_cwd = Path.cwd()
        os.chdir(self.starperf_dir)
        try: yield
        finally: os.chdir(prev_cwd)
    
    def _initialize_constellation(self):
        with self._as_current_dir():
            logger.info(f"Loading constellation structure for {self.config['constellation_name']}...")
            constellation = constellation_configuration.constellation_configuration(
                dT=self.config['dT'], constellation_name=self.config['constellation_name']
            )
            
            logger.info("Building in-memory satellite connectivity...")
            connection_manager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
            connection_manager.execute_connection_policy(constellation=constellation, dT=self.config['dT'])
            
            '''
            # --- THAY THẾ BẰNG KHỐI DEBUG MỚI NÀY ---
            print("\n--- DEBUG: POST-CONNECTION ANALYSIS ---")
            shell = constellation.shells[0]
            sample_sat = shell.orbits[0].satellites[0]

            print("\nAttributes of a sample satellite:")
            print(dir(sample_sat))

            if hasattr(sample_sat, 'ISL'):
                print("\nFound 'ISL' attribute in satellite object.")
                print("Type of ISL:", type(sample_sat.ISL))
                
                # Kiểm tra xem ISL có phải là list hoặc tuple không
                if isinstance(sample_sat.ISL, (list, tuple)):
                    print("Length of ISL list:", len(sample_sat.ISL))
                    
                    # Nếu list không rỗng, in ra phần tử đầu tiên
                    if sample_sat.ISL:
                        first_element = sample_sat.ISL[0]
                        print("First element of ISL list is of type:", type(first_element))
                        
                        # In ra các thuộc tính của phần tử đầu tiên để xem nó là gì
                        print("Attributes of the first element:", dir(first_element))
            else:
                print("\n'ISL' attribute NOT found in satellite object.")

            print("--- END DEBUG ---\n")
            '''

            
        return constellation

    # ... bên trong class StarPerfBackend ...
    def get_neighbors(self, sat_id: int, time_step: int) -> List[Satellite]:
        """
        Returns a list of neighbor satellite objects for a given satellite at a time step.
        This is derived from the 'ISL' list within each satellite object.
        """
        current_sat = self.sat_id_map[sat_id]
        
        if not hasattr(current_sat, 'ISL'):
            return []

        neighbors = []
        for isl_link in current_sat.ISL:
            # SỬA LỖI Ở ĐÂY:
            # isl_link.satellite1 và satellite2 là các số nguyên (ID).
            # So sánh chúng với ID của vệ tinh hiện tại.
            
            neighbor_id = -1
            if isl_link.satellite1 == current_sat.id:
                neighbor_id = isl_link.satellite2
            elif isl_link.satellite2 == current_sat.id:
                neighbor_id = isl_link.satellite1
            
            if neighbor_id != -1:
                # Lấy đối tượng satellite hoàn chỉnh từ ID bằng map của chúng ta
                neighbors.append(self.sat_id_map[neighbor_id])
        
        return neighbors

    # --- HÀM MỚI CHO SCALABILITY ---
    def get_satellite_position(self, sat: Satellite, time_step: int) -> np.ndarray:
        """
        Returns the satellite's position as a NumPy array [lon, lat, alt].
        This abstracts away the raw data structure of StarPerf.
        """
        # Đảm bảo time_step hợp lệ
        safe_time_step = min(time_step, self.config['dT']) - 1 # Chuyển sang chỉ số 0-based
        
        # Xử lý trường hợp altitude là scalar hoặc list
        alt = sat.altitude[safe_time_step] if isinstance(sat.altitude, list) else sat.altitude
        
        return np.array([
            sat.longitude[safe_time_step], 
            sat.latitude[safe_time_step], 
            alt
        ])

    def find_nearest_satellite(self, user: User, time_step: int) -> Satellite:
        return min(
            (sat for orbit in self.shell.orbits for sat in orbit.satellites),
            key=lambda sat: distance_between_satellite_and_user(user, sat, time_step)
        )

class RoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, **kwargs):
        super().__init__()
        self.config = {
            "constellation_name": "Starlink", "dT": 1000, "max_hops": 30,
            "reward_success": 100.0, "reward_failure": -100.0, "reward_per_hop": -1.0,
        }
        self.config.update(kwargs)
        
        logger.info("--- Initializing SatGym-Routing-v0 Environment ---")
        self.backend = StarPerfBackend(self.config)
        
        self.MAX_NEIGHBORS = 6
        self.action_space = spaces.Discrete(self.MAX_NEIGHBORS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

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
        logger.info(f"Resetting episode: {self.source_user.user_name} -> {self.target_user.user_name} | "
                    f"Path: Sat-{self.start_sat.id} -> Sat-{self.target_sat.id}")
        return self._get_observation(), self._get_info()
    
    def step(self, action: int):
        self.hop_count += 1
        
        # Đảm bảo time_step hợp lệ TRƯỚC khi sử dụng
        safe_time_step = min(self.time_step, self.config['dT'])
        neighbors = self.backend.get_neighbors(self.current_sat.id, safe_time_step)
        
        terminated = False
        if action >= len(neighbors):
            logger.warning(f"Invalid action {action} for {len(neighbors)} neighbors at t={self.time_step}. Packet dropped.")
            reward = self.config["reward_failure"]
            terminated = True
        else:
            self.current_sat = neighbors[action]
            reward = self.config["reward_per_hop"]
        
        if not terminated and self.current_sat.id == self.target_sat.id:
            logger.info(f"Target satellite {self.target_sat.id} reached in {self.hop_count} hops!")
            reward = self.config["reward_success"]
            terminated = True
        elif not terminated and self.hop_count >= self.config["max_hops"]:
            logger.warning(f"Max hops {self.config['max_hops']} exceeded. Terminating episode.")
            reward = self.config["reward_failure"]
            terminated = True

        # Tăng time_step LÊN SAU KHI đã sử dụng nó
        self.time_step += 1
        
        truncated = self.time_step > self.config['dT']
        if truncated:
            logger.warning(f"Simulation time {self.config['dT']}s exceeded. Truncating episode.")
            # Khi bị truncate, thường không phạt nặng như khi thất bại
            # terminated phải là False khi truncated là True
            terminated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self):
        # Đảm bảo time_step dùng để truy cập luôn nằm trong phạm vi
        # Dùng self.time_step hiện tại, nhưng không vượt quá dT
        safe_time_step = min(self.time_step, self.config['dT'])
        
        pos_current = self.backend.get_satellite_position(self.current_sat, safe_time_step)
        pos_target = self.backend.get_satellite_position(self.target_sat, safe_time_step)
        
        direction_vector = pos_target - pos_current
        norm = np.linalg.norm(direction_vector)
        if norm > 0:
            direction_vector /= norm
            
        return direction_vector.astype(np.float32)



    def _get_info(self):
        return {
            "time_step": self.time_step, "hop_count": self.hop_count,
            "current_sat_id": self.current_sat.id, "target_sat_id": self.target_sat.id
        }

    def close(self):
        logger.info("Closing environment.")