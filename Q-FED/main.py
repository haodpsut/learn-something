# main.py

import matplotlib.pyplot as plt
import config
import geometry
import os # Thêm thư viện os để làm việc với thư mục

import matplotlib.pyplot as plt
import numpy as np # Cần numpy cho poisson
import config
import geometry
import channel # Import module mới
import os


import numpy as np

def visualize_deployment(satellite_pos, to_positions, all_users, filename="deployment.png"):
    """
    Vẽ sơ đồ phân bố mạng và lưu thành file ảnh.
    
    Args:
        satellite_pos (tuple): Tọa độ vệ tinh.
        to_positions (list): Danh sách tọa độ các TO.
        all_users (list): Danh sách tọa độ tất cả người dùng.
        filename (str): Tên file để lưu hình ảnh.
    """
    plt.figure(figsize=(10, 10))
    
    # Vẽ các trạm mặt đất (TOs)
    to_x = [pos[0] for pos in to_positions]
    to_y = [pos[1] for pos in to_positions]
    plt.scatter(to_x, to_y, c='red', marker='s', s=100, label='Terrestrial Operators (TOs)')
    
    # Vẽ người dùng
    user_x = [pos[0] for pos in all_users]
    user_y = [pos[1] for pos in all_users]
    plt.scatter(user_x, user_y, c='blue', marker='.', s=10, label='Users')
    
    # Vẽ hình chiếu của vệ tinh
    sat_proj_x, sat_proj_y, _ = satellite_pos
    plt.scatter(sat_proj_x, sat_proj_y, c='green', marker='*', s=200, label=f'Satellite Projection (z={config.SAT_ALTITUDE/1000}km)')
    
    # Thiết lập đồ thị
    plt.title('Network Deployment Scenario')
    plt.xlabel('X-coordinate (meters)')
    plt.ylabel('Y-coordinate (meters)')
    plt.xlim(0, config.AREA_WIDTH)
    plt.ylim(0, config.AREA_HEIGHT)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # --- THAY ĐỔI QUAN TRỌNG ---
    # Tạo thư mục results nếu chưa có
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # Lưu hình ảnh
    filepath = os.path.join('results', filename)
    plt.savefig(filepath, dpi=300) # dpi=300 cho chất lượng cao
    plt.close() # Đóng figure để giải phóng bộ nhớ
    
    print(f"Deployment visualization saved to '{filepath}'")


if __name__ == '__main__':
    # --- Bước 1: Tạo kịch bản mạng ---
    
    current_satellite_pos = geometry.get_satellite_position(time_t=1)
    to_positions = geometry.deploy_terrestrial_operators(config.NUM_TO)
    
    all_users_by_to = {} # Dùng dictionary để quản lý người dùng của từng TO
    all_users_list = []
    
    for i, to_pos in enumerate(to_positions):
        num_users_this_to = np.random.poisson(config.USERS_PER_TO)
        users = geometry.deploy_users_around_to(to_pos, num_users_this_to, config.TO_CELL_RADIUS)
        all_users_by_to[i] = users # Lưu người dùng theo index của TO
        all_users_list.extend(users)
        
    # --- Bước 2: Trực quan hóa (tùy chọn) ---
    print(f"Deployed {len(to_positions)} TOs.")
    print(f"Deployed {len(all_users_list)} users.")
    print(f"Satellite position at t=1s: {current_satellite_pos}")
    
    visualize_deployment(current_satellite_pos, to_positions, all_users_list)
    
    # --- Bước 3: Kiểm tra Module Kênh truyền ---
    print("\n--- Channel Gain Calculation Test ---")
    
    # Chọn TO đầu tiên và người dùng đầu tiên của nó để kiểm tra
    if config.NUM_TO > 0 and len(all_users_by_to.get(0, [])) > 0:
        first_to_pos = to_positions[0]
        first_user_pos = all_users_by_to[0][0]
        
        # 1. Tính độ lợi kênh từ Vệ tinh tới người dùng này
        sat_to_user_gain = channel.get_satellite_channel_gain(current_satellite_pos, first_user_pos)
        print(f"Satellite -> User Channel Gain: {sat_to_user_gain:.3e} (linear) or {channel.linear_to_db(sat_to_user_gain):.2f} dB")

        # 2. Tính độ lợi kênh từ TO của nó tới người dùng này
        terra_to_user_gain = channel.get_terrestrial_channel_gain(first_to_pos, first_user_pos)
        print(f"Terrestrial TO -> User Channel Gain: {terra_to_user_gain:.3e} (linear) or {channel.linear_to_db(terra_to_user_gain):.2f} dB")
        
        # Tính SINR (ví dụ đơn giản)
        # Tín hiệu mong muốn từ TO mặt đất
        signal_power = config.TERRA_TRANS_POWER_W * terra_to_user_gain
        # Nhiễu chỉ từ tạp âm (chưa có nhiễu từ các nguồn khác)
        noise_power = config.NOISE_POWER_W
        
        sinr = signal_power / noise_power
        throughput = np.log2(1 + sinr) # Shannon capacity (bps/Hz)
        
        print(f"\nSimple SINR Calculation (Terrestrial Link):")
        print(f"Signal Power: {channel.linear_to_db(signal_power) + 30:.2f} dBm")
        print(f"Noise Power: {channel.linear_to_db(noise_power) + 30:.2f} dBm")
        print(f"SINR: {channel.linear_to_db(sinr):.2f} dB")
        print(f"Achievable Rate (bps/Hz): {throughput:.2f} bps/Hz")
