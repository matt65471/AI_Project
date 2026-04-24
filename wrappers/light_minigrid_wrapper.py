import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MiniGridLightWrapper(gym.ObservationWrapper):
    """
    Optimized for CPU:
    - Uses raw symbolic grid (7x7x3) instead of rendering pixels.
    - No frame stacking (the LSTM in DRQN handles time).
    - Output: (3, 7, 7) uint8
    """
    def __init__(self, env):
        super().__init__(env)
        # MiniGrid symbolic obs is (7, 7, 3). We transpose to (3, 7, 7) for PyTorch Conv2d.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, 7, 7),
            dtype=np.uint8
        )

    def observation(self, obs):
        # obs['image'] is (7, 7, 3). PyTorch expects (C, H, W) -> (3, 7, 7)
        image = obs["image"]
        return np.transpose(image, (2, 0, 1))

def make_minigrid_env(env_id="MiniGrid-MemoryS7-v0", render_mode=None):
    """
    Returns a lightweight environment for CPU training.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = MiniGridLightWrapper(env)
    return env