import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MiniGridLightWrapper(gym.ObservationWrapper):
    """
    Optimized for CPU:
    - Uses raw symbolic grid instead of rendering pixels
    - No frame stacking (LSTM handles time)
    - Output: (3, H, W) uint8
    """
    def __init__(self, env):
        super().__init__(env)

        # Get observation size from env
        obs_shape = env.observation_space["image"].shape  # (H, W, 3)
        h, w, c = obs_shape

        self.obs_h = h
        self.obs_w = w

        # PyTorch expects (C, H, W)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c, h, w),
            dtype=np.uint8
        )

    def observation(self, obs):
        # obs['image'] is (H, W, 3) → transpose to (3, H, W)
        image = obs["image"]
        return np.transpose(image, (2, 0, 1)).astype(np.uint8)


def make_minigrid_env(env_id="MiniGrid-MemoryS7-v0", render_mode=None):
    """
    Returns a lightweight environment for CPU training.
    Uses symbolic observations instead of rendered pixels.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = MiniGridLightWrapper(env)
    return env