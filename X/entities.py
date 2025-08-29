# src/satgym/entities.py
from .backend import User
import numpy as np

class MobileUser(User):
    """Represents a moving user, e.g., on an airplane."""
    def __init__(self, start_pos: tuple, end_pos: tuple, total_steps: int, name: str):
        super().__init__(latitude=start_pos[0], longitude=start_pos[1], user_name=name)
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.total_steps = total_steps
        self.current_step = 0

        # --- THUỘC TÍNH MỚI CHO TASK OFFLOADING ---
        self.cpu_capacity_ghz = 1.0  # Năng lực tính toán (ví dụ: 1 GHz)
        self.power_transmit_watt = 0.5 # Công suất phát (ví dụ: 0.5W)
        self.power_compute_watt = 0.8 # Công suất khi tính toán (ví dụ: 0.8W)
        # ----------------------------------------


    def move(self):
        """Updates the user's position for the next time step using linear interpolation."""
        self.current_step += 1
        fraction = min(self.current_step / self.total_steps, 1.0)
        current_pos = self.start_pos + fraction * (self.end_pos - self.start_pos)
        self.latitude, self.longitude = current_pos[0], current_pos[1]

    def reset(self):
        """Resets the user to the starting position."""
        self.current_step = 0
        self.latitude, self.longitude = self.start_pos[0], self.start_pos[1]