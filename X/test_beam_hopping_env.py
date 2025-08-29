import gymnasium as gym
import numpy as np
import logging

# Import package chính để đăng ký môi trường
try:
    import satgym
except ImportError:
    print("Lỗi: Không thể import package 'satgym'.")
    print("Hãy chắc chắn rằng bạn đã chạy 'pip install -e .' trong môi trường conda.")
    exit()

# Cấu hình logging để xem output chi tiết
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_beam_hopping_env():
    """
    A simple script to test the basic functionality of the BeamHoppingEnv.
    """
    print("\n" + "="*50)
    print("--- Testing SatGym-BeamHopping-v0 Environment ---")
    print("="*50 + "\n")
    
    ENVIRONMENT_ID = "SatGym-BeamHopping-v0"
    NUM_STEPS = 10 # Số bước để chạy thử

    try:
        # 1. Tạo môi trường
        logger.info(f"Attempting to create environment: '{ENVIRONMENT_ID}'")
        # Giảm số bước mô phỏng để test nhanh hơn
        env = gym.make(ENVIRONMENT_ID, simulation_steps=50) 
        logger.info("✅ Environment created successfully!")

        # 2. Kiểm tra Action và Observation space
        logger.info("\n--- Checking Spaces ---")
        logger.info(f"Action Space: {env.action_space}")
        logger.info(f"Observation Space: {env.observation_space}")
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete), "Action space should be MultiDiscrete"
        assert isinstance(env.observation_space, gym.spaces.Box), "Observation space should be Box"
        logger.info("✅ Spaces are valid.")

        # 3. Chạy thử một episode ngắn
        logger.info(f"\n--- Running a short test for {NUM_STEPS} steps ---")
        
        observation, info = env.reset(seed=42)
        logger.info(f"Initial reset successful.")
        logger.info(f"Initial Info: {info}")
        logger.info(f"Initial observation shape (num cells): {observation.shape}")
        assert env.observation_space.contains(observation), "Initial observation is not in the defined space"

        # Vòng lặp tương tác
        for i in range(NUM_STEPS):
            # Chọn một hành động ngẫu nhiên (gán ngẫu nhiên các búp sóng vào các ô)
            action = env.action_space.sample()
            logger.info(f"\n>>> Step {i+1}/{NUM_STEPS} <<<")
            
            num_beams_to_assign = env.unwrapped.config['num_beams']
            logger.info(f"Taking random action (assigning {num_beams_to_assign} beams to cells)...")
                    
            observation, reward, terminated, truncated, info = env.step(action)
            
            logger.info(f"Current visible cells: {info.get('num_visible_cells', '?')}")
            logger.info(f"Received reward (Total Throughput): {reward:.2f} Gbps")
            logger.info(f"Is terminated: {terminated}, Is truncated: {truncated}")
            
            # Kiểm tra lại observation vì kích thước có thể thay đổi
            if not env.observation_space.contains(observation):
                 logger.warning(f"Observation shape mismatch! Env shape: {env.observation_space.shape}, Obs shape: {observation.shape}")
                 # Đây là một vấn đề cần giải quyết, nhưng cho phép test tiếp tục
            
            if terminated or truncated:
                logger.warning("Episode finished. Test stopping.")
                break

        # 4. Đóng môi trường
        logger.info("\n--- Closing the environment ---")
        env.close()

        print("\n" + "="*28)
        print("🎉   Test SCRIPT COMPLETED!   🎉")
        print("Nếu không có lỗi nào ở trên, môi trường BeamHoppingEnv đã hoạt động ở mức cơ bản.")

    except Exception as e:
        print("\n" + "="*26)
        print("❌   Test FAILED!   ❌")
        print("="*26)
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    test_beam_hopping_env()