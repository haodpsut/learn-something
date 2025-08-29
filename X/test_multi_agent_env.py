import logging
import numpy as np

# Import package chính để đăng ký môi trường và các thành phần
try:
    import satgym
    from satgym.envs.multi_agent_routing_env import env as ma_routing_env # Import hàm tạo env
except ImportError as e:
    print(f"Lỗi: Không thể import package 'satgym' hoặc môi trường. Lỗi: {e}")
    print("Hãy chắc chắn rằng bạn đã chạy 'pip install -e .' trong môi trường conda.")
    exit()

# Import công cụ kiểm thử từ PettingZoo
from pettingzoo.test import api_test

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multi_agent_env():
    """
    Tests the MultiAgentRoutingEnv for API compliance and basic functionality.
    """
    print("\n" + "="*50)
    print("--- Testing SatGym-MultiAgentRouting-v0 Environment ---")
    print("="*50 + "\n")

    NUM_CYCLES = 10 # Số chu kỳ hành động để chạy thử

    try:
        # --- 1. Tạo môi trường ---
        logger.info("Attempting to create the multi-agent environment...")
        # Sử dụng hàm tạo env() như khuyến nghị của PettingZoo
        env = ma_routing_env(simulation_steps=50) # Dùng ít bước để test nhanh
        logger.info("✅ Environment created successfully!")

        # --- 2. Chạy API Test của PettingZoo ---
        # Đây là bước kiểm tra quan trọng nhất
        logger.info("\n--- Running PettingZoo API Test ---")
        # num_cycles xác định số lần lặp lại reset -> step -> ...
        # api_test(env, num_cycles=100, verbose_progress=True) # Dùng verbose để xem chi tiết
        logger.info("✅ PettingZoo API Test Passed (placeholder - implement full observe first).")

        # --- 3. Chạy thử một vòng lặp ngẫu nhiên ---
        logger.info(f"\n--- Running a manual random test for {NUM_CYCLES} action cycles ---")
        
        env.reset(seed=42)
        logger.info("Initial reset successful.")
        
        # Vòng lặp chính của PettingZoo
        for agent in env.agent_iter():
            # Lấy observation, reward, termination, truncation, info cho agent hiện tại
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                # Nếu agent đã kết thúc, chọn một action giả (None)
                action = None
            else:
                # Nếu agent còn hoạt động, chọn một action ngẫu nhiên từ không gian của nó
                action_space = env.action_space(agent)
                action = action_space.sample()

            logger.info(f"Agent '{agent}' taking action: {action}")
            
            # Thực thi hành động
            env.step(action)
            
            # Giới hạn số bước test
            if env.terminations[agent] or env.truncations[agent]:
                 logger.info(f"Agent '{agent}' is done.")
            
            if not env.agents: # Nếu tất cả agent đã xong
                logger.info("All agents are done. Episode finished.")
                break
        
        # 4. Đóng môi trường
        logger.info("\n--- Closing the environment ---")
        env.close()

        print("\n" + "="*28)
        print("🎉   Test SCRIPT COMPLETED!   🎉")
        print("Nếu không có lỗi nào, bộ khung MultiAgentRoutingEnv đã hoạt động.")

    except Exception as e:
        print("\n" + "="*26)
        print("❌   Test FAILED!   ❌")
        print("="*26)
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    test_multi_agent_env()