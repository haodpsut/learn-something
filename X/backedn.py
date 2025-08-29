# src/satgym/backend.py

import sys
import os
from pathlib import Path
import contextlib
import logging
from typing import List, Dict, Any
from skyfield.api import load, wgs84 # Đảm bảo có import này

import networkx as nx
import numpy as np
import h5py # <-- DÒNG CẦN THÊM

from .workload import Task # Import Task để sử dụng trong type hint

# --- THÊM IMPORT BỊ THIẾU ---
from .entities import MobileUser
from .workload import Task # Đảm bảo Task cũng được import


# --- Cấu hình logging ---
logger = logging.getLogger(__name__)

# --- TÍCH HỢP STARPERF ---
#STARPERF_PATH = Path(__file__).resolve().parent.parent.parent / "deps" / "StarPerf_Simulator"

#if str(STARPERF_PATH.resolve()) not in sys.path:
#    sys.path.insert(0, str(STARPERF_PATH.resolve()))

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
    This class is the single source of truth for the simulation state.
    """
    def __init__(self, starperf_path: Path, config: Dict[str, Any]):
        self.starperf_dir = starperf_path # Gán từ tham số
        self.config = config
        
        self.link_config = {
            "isl_capacity_gbps": 10.0,
            "uplink_bandwidth_gbps": 1.0, # Băng thông đường lên User -> Sat
        }
        
        
                # --- THÊM CẤU HÌNH CHO TÀI NGUYÊN TÍNH TOÁN ---
        self.compute_config = {
            "satellite_cpu_capacity_ghz": 10.0, # Năng lực tính toán của mỗi vệ tinh
            "ground_cpu_capacity_ghz": 1000.0, # Năng lực của Cloud (rất lớn)
            "cycles_per_flop": 1e-9 # Giả định: 1 GFLOP cần 1 chu kỳ CPU (GHz)
        }

        self.ts = load.timescale()
        self.constellation = self._initialize_constellation()
        self.shell = self.constellation.shells[0]
        self.num_satellites = self.shell.number_of_satellites
        self.sat_id_map: Dict[int, Satellite] = {sat.id: sat for orbit in self.shell.orbits for sat in orbit.satellites}


        # --- "TRANG TRÍ" CÁC VỆ TINH VỚI THUỘC TÍNH MỚI ---
        for sat in self.sat_id_map.values():
            sat.cpu_capacity_ghz = self.compute_config["satellite_cpu_capacity_ghz"]
            sat.cpu_load = 0.0 # Tải ban đầu là 0
        # ---------------------------------------------

        
        sample_sat = self.sat_id_map[1]
        self.simulation_steps = len(sample_sat.longitude)
        logger.info(f"Initialized with {self.simulation_steps} simulation time steps.")

        self._graph_cache: Dict[int, nx.Graph] = {}

        # --- THÊM QUẢN LÝ TRẠNG THÁI BĂNG THÔNG ---
        # key: (time_step, sat1_id, sat2_id), value: used_bandwidth_gbps
        self.link_bandwidth_state: Dict[tuple, float] = {}



    # --- CÁC HÀM HELPER MỚI CHO TASK OFFLOADING ---
    
    def calculate_transmission_time(self, data_size_mb: float, bandwidth_gbps: float) -> float:
        """Calculates time to transmit data in seconds."""
        if bandwidth_gbps <= 0:
            return float('inf')
        data_size_gb = data_size_mb / 1000.0
        return data_size_gb / bandwidth_gbps

    def calculate_computation_time(self, task: Task, cpu_capacity_ghz: float) -> float:
        """Calculates time to compute a task in seconds."""
        if cpu_capacity_ghz <= 0:
            return float('inf')
        # Thời gian = (Số phép tính) / (Số phép tính mỗi giây)
        # Số phép tính = task.compute_load_gflops
        # Số phép tính/giây = cpu_capacity_ghz / cycles_per_flop (theo giả định của chúng ta)
        compute_time = task.compute_load_gflops / (cpu_capacity_ghz / self.compute_config["cycles_per_flop"])
        return compute_time

    def get_offloading_costs(self, user: MobileUser, serving_sat: Satellite, task: Task, time_step: int) -> dict:
        """
        Calculates the latency and energy cost for all three offloading decisions.
        Returns a dictionary with the costs.
        """
        costs = {}
        
        # --- 1. Chi phí thực thi tại chỗ (Local) ---
        latency_local = self.calculate_computation_time(task, user.cpu_capacity_ghz)
        energy_local = latency_local * user.power_compute_watt
        costs['local'] = {'latency': latency_local, 'energy': energy_local}
        
        # --- 2. Chi phí Offload lên Vệ tinh (Edge) ---
        link_quality = self.get_link_quality(user, serving_sat, time_step)
        uplink_bandwidth = self.link_config["uplink_bandwidth_gbps"] * link_quality
        
        time_uplink = self.calculate_transmission_time(task.data_size_mb, uplink_bandwidth)
        time_compute_sat = self.calculate_computation_time(task, serving_sat.cpu_capacity_ghz)
        # Bỏ qua thời gian downlink và queue cho đơn giản
        latency_edge = time_uplink + time_compute_sat
        
        energy_edge = (time_uplink * user.power_transmit_watt) # Năng lượng user dùng để gửi
        costs['edge'] = {'latency': latency_edge, 'energy': energy_edge}

        # --- 3. Chi phí Offload xuống Cloud (Mặt đất) ---
        # Đây là mô hình rất đơn giản
        # Giả định độ trễ từ vệ tinh xuống mặt đất và xử lý là một hằng số
        latency_sat_to_cloud = 0.050 # 50ms
        latency_cloud = time_uplink + latency_sat_to_cloud
        energy_cloud = energy_edge # Năng lượng user bỏ ra là như nhau
        costs['cloud'] = {'latency': latency_cloud, 'energy': energy_cloud}
        
        return costs



    def get_elevation_angle(self, user: User, satellite: Satellite, time_step: int) -> float:
        """
        Calculates the elevation angle of a satellite relative to a ground user.
        """
        # --- SỬA LỖI Ở ĐÂY ---
        # Sử dụng self.ts đã được khởi tạo
        sat_skyfield_obj = satellite.true_satellite 
        time = self.ts.utc(2023, 10, 1, 0, 0, (time_step - 1) * self.config['dT'])
        # --------------------

        user_location = wgs84.latlon(user.latitude, user.longitude)
        difference = sat_skyfield_obj - user_location
        topocentric = difference.at(time)
        alt, az, distance = topocentric.altaz()
        
        return alt.degrees

    def get_link_quality(self, user: User, satellite: Satellite, time_step: int) -> float:
        """
        Calculates a normalized link quality score (0 to 1) for a user-satellite link.
        A simple model based on elevation angle. Returns 0 if satellite is below horizon.
        """
        elevation = self.get_elevation_angle(user, satellite, time_step)

        # Nếu vệ tinh dưới đường chân trời, chất lượng liên kết là 0
        if elevation < 0:
            return 0.0
        
        # Mô hình đơn giản: chất lượng tỉ lệ thuận với sin(góc ngẩng)
        # sin(0) = 0, sin(90) = 1. Đây là một cách chuẩn hóa tự nhiên.
        quality = np.sin(np.radians(elevation))
        
        return quality

    def find_k_best_satellites(self, user: User, time_step: int, k: int) -> List[Satellite]:
        """
        Finds the K satellites with the best link quality for a given user.
        """
        # Lấy tất cả vệ tinh
        all_sats = list(self.sat_id_map.values())
        
        # Sắp xếp các vệ tinh dựa trên chất lượng liên kết (giảm dần)
        # Dùng lambda function để tính quality cho mỗi vệ tinh
        sorted_sats = sorted(
            all_sats,
            key=lambda sat: self.get_link_quality(user, sat, time_step),
            reverse=True
        )
        
        # Trả về K vệ tinh hàng đầu
        return sorted_sats[:k]



    def get_path_bottleneck_bandwidth(self, path: List[int], time_step: int) -> float:
        """Calculates the minimum available bandwidth along a path."""
        if not path:
            return 0.0
        
        bottleneck = float('inf')
        for i in range(len(path) - 1):
            u, v = sorted((path[i], path[i+1]))
            link_key = (time_step, u, v)
            
            used_bw = self.link_bandwidth_state.get(link_key, 0.0)
            available_bw = self.link_config["isl_capacity_gbps"] - used_bw
            bottleneck = min(bottleneck, available_bw)
            
        return bottleneck


    @contextlib.contextmanager
    def _as_current_dir(self):
        prev_cwd = Path.cwd()
        os.chdir(self.starperf_dir)
        try: yield
        finally: os.chdir(prev_cwd)

    def _initialize_constellation(self):
        h5_filepath = self.starperf_dir / "data" / "XML_constellation" / f"{self.config['constellation_name']}.h5"
        
        with self._as_current_dir():
            logger.info(f"Loading constellation structure for {self.config['constellation_name']}...")
            
            # Bước 1: Gọi hàm của StarPerf. Nó sẽ tạo file HDF5 chỉ có cấu trúc /position/shellX
            constellation = constellation_configuration.constellation_configuration(
                dT=self.config['dT'], constellation_name=self.config['constellation_name']
            )
            
            # --- GIẢI PHÁP NÂNG CAO ---
            # Bước 2: Sửa chữa file HDF5 để nó có cấu trúc mà các hàm sau mong đợi.
            logger.info(f"Augmenting HDF5 file structure at {h5_filepath}...")
            try:
                with h5py.File(h5_filepath, 'a') as file: # Mở ở chế độ append ('a')
                    if 'delay' not in file:
                        delay_group = file.create_group('delay')
                        
                        # Sao chép cấu trúc subgroup 'shellX' từ 'position' sang 'delay'
                        if 'position' in file:
                            position_group = file['position']
                            for shell_name in position_group.keys():
                                if shell_name not in delay_group:
                                    delay_group.create_group(shell_name)
                                    logger.info(f"Created missing subgroup: /delay/{shell_name}")
            except Exception as e:
                logger.error(f"Failed to augment HDF5 file: {e}", exc_info=True)
                raise
            # --------------------------

            logger.info("Building in-memory satellite connectivity and writing to HDF5...")
            
            # Bước 3: Bây giờ, hàm kết nối sẽ tìm thấy cấu trúc /delay/shellX mà nó cần.
            connection_manager = connectivity_mode_plugin_manager.connectivity_mode_plugin_manager()
            connection_manager.execute_connection_policy(constellation=constellation, dT=self.config['dT'])
            
        return constellation



    

    def reset_bandwidth_state(self):
        """Clears all bandwidth allocations."""
        self.link_bandwidth_state.clear()
        logger.info("Backend bandwidth state has been reset.")

    def get_remaining_bandwidth(self, time_step: int) -> np.ndarray:
        """
        Returns a flattened vector of remaining bandwidth on all potential links.
        This is a core part of the observation for ResourceAllocationEnv.
        """
        # Đây là một hàm phức tạp, chúng ta sẽ hiện thực hóa nó sau.
        # Tạm thời trả về một vector giả.
        num_sats = self.num_satellites
        num_links = num_sats * (num_sats - 1) // 2
        return np.full(num_links, self.link_config["isl_capacity_gbps"])

    def allocate_path(self, path: List[int], demand_gbps: float, time_step: int) -> bool:
        """
        Tries to allocate bandwidth for a demand along a specific path.
        Returns True if successful, False otherwise.
        This function implements the "physics" of resource allocation.
        """
        # 1. Kiểm tra xem có đủ tài nguyên không
        for i in range(len(path) - 1):
            u, v = sorted((path[i], path[i+1])) # Sắp xếp để key luôn nhất quán
            link_key = (time_step, u, v)
            
            used_bw = self.link_bandwidth_state.get(link_key, 0.0)
            available_bw = self.link_config["isl_capacity_gbps"] - used_bw
            
            if demand_gbps > available_bw:
                logger.warning(f"Allocation failed on link ({u}-{v}) at t={time_step}. "
                             f"Demand: {demand_gbps:.2f}, Available: {available_bw:.2f}")
                return False # Không đủ băng thông trên một liên kết

        # 2. Nếu kiểm tra thành công, thực hiện cấp phát
        for i in range(len(path) - 1):
            u, v = sorted((path[i], path[i+1]))
            link_key = (time_step, u, v)
            
            self.link_bandwidth_state[link_key] = self.link_bandwidth_state.get(link_key, 0.0) + demand_gbps

        logger.info(f"Successfully allocated {demand_gbps:.2f} Gbps on path {path} at t={time_step}.")
        return True




    # --- THÊM LẠI HÀM BỊ THIẾU VÀO ĐÂY ---
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
            neighbor_id = -1
            if isl_link.satellite1 == current_sat.id:
                neighbor_id = isl_link.satellite2
            elif isl_link.satellite2 == current_sat.id:
                neighbor_id = isl_link.satellite1
            
            if neighbor_id != -1:
                # Lấy đối tượng satellite hoàn chỉnh từ ID bằng map của chúng ta
                neighbors.append(self.sat_id_map[neighbor_id])
        
        return neighbors
    # ------------------------------------


    
        # --- HÀM HELPER MỚI ---
    def _distance_between_sats(self, sat1: Satellite, sat2: Satellite, time_step: int) -> float:
        """
        Calculates the great-circle distance in kilometers between two satellites.
        This is a dedicated version for sat-to-sat distance calculation.
        """
        # Lấy vị trí tại đúng time_step
        safe_idx = min(time_step, self.simulation_steps) - 1
        lon1, lat1 = sat1.longitude[safe_idx], sat1.latitude[safe_idx]
        lon2, lat2 = sat2.longitude[safe_idx], sat2.latitude[safe_idx]

        # Chuyển đổi độ sang radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Công thức Haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c  # Bán kính Trái Đất ~ 6371 km
        return km



    def get_network_graph(self, time_step: int) -> nx.Graph:
        if time_step in self._graph_cache:
            return self._graph_cache[time_step]
        
        graph = self._build_network_graph(time_step)
        self._graph_cache[time_step] = graph
        return graph

    
    def _build_network_graph(self, time_step: int) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.sat_id_map.keys())
        
        for sat_id, current_sat in self.sat_id_map.items():
            if not hasattr(current_sat, 'ISL'): continue
            
            for isl_link in current_sat.ISL:
                neighbor_id = -1
                if isl_link.satellite1 == sat_id: neighbor_id = isl_link.satellite2
                elif isl_link.satellite2 == sat_id: neighbor_id = isl_link.satellite1
                
                if neighbor_id != -1 and sat_id < neighbor_id:
                    neighbor_sat = self.sat_id_map[neighbor_id]
                    
                    # --- SỬA LỖI Ở ĐÂY ---
                    # Sử dụng hàm mới, chuyên dụng của chúng ta
                    dist = self._distance_between_sats(current_sat, neighbor_sat, time_step)
                    # ----------------------
                    
                    C = 299792.458
                    delay = dist / C
                    capacity = self.link_config["isl_capacity_gbps"]
                    
                    G.add_edge(sat_id, neighbor_id, delay=delay, capacity=capacity)
        return G


    def find_k_shortest_paths(self, graph: nx.Graph, source_id: int, target_id: int, k: int, weight: str = 'delay') -> List[List[int]]:
        if not graph.has_node(source_id) or not graph.has_node(target_id): return []
        paths_generator = nx.shortest_simple_paths(graph, source=source_id, target=target_id, weight=weight)
        k_shortest_paths = []
        try:
            for i, path in enumerate(paths_generator):
                if i >= k: break
                k_shortest_paths.append(path)
        except nx.NetworkXNoPath: return []
        return k_shortest_paths

    def get_satellite_position(self, sat: Satellite, time_step: int) -> np.ndarray:
        safe_time_step_index = min(time_step, self.simulation_steps) - 1
        alt = sat.altitude[safe_time_step_index] if isinstance(sat.altitude, list) else sat.altitude
        return np.array([sat.longitude[safe_time_step_index], sat.latitude[safe_time_step_index], alt])

    def find_nearest_satellite(self, user: User, time_step: int) -> Satellite:
        return min(
            (sat for orbit in self.shell.orbits for sat in orbit.satellites),
            key=lambda sat: distance_between_satellite_and_user(user, sat, time_step)
        )