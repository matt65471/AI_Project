import minigrid
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces
from collections import deque
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


class MiniGridWrapper(gym.Wrapper):
    """
    Wraps MiniGrid MemoryEnv to match the same interface as the Atari wrapper:
    - Extracts RGB image observation from the dict obs
    - Converts to grayscale
    - Resizes to 84x84
    - Stacks 4 frames
    Output: (4, 84, 84) uint8 — same shape as Breakout
    """

    def __init__(self, env, frame_stack=4, screen_size=(84, 84)):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.screen_size = screen_size
        self._frames = deque(maxlen=frame_stack)

        h, w = screen_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, h, w),
            dtype=np.uint8,
        )

    def _process(self, obs):
        # obs is (H, W, 3) RGB uint8
        if isinstance(obs, dict):
            img = obs["image"]
        else:
            img = obs
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        h, w = self.screen_size
        resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        return resized  # (84, 84) uint8

    def _get_obs(self):
        return np.stack(self._frames, axis=0)  # (4, 84, 84)

    def _extract_image(self, obs):
        # MiniGrid returns a dict — pull the image key
        if isinstance(obs, dict):
            return obs["image"]
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed = self._process(obs)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(processed)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self._process(obs)

        #cv2.imshow("What the AI sees", processed)
        #cv2.waitKey(1)

        self._frames.append(processed)
        return self._get_obs(), reward, terminated, truncated, info


def make_minigrid_env(
    env_id="MiniGrid-MemoryS7-v0",
    render_mode=None,
    frame_stack=4,
    screen_size=(84, 84),
):
    """
    Makes MiniGrid MemoryEnv wrapped to match Atari (4, 84, 84) uint8 format.
    Uses RGBImgObsWrapper to convert symbolic obs to RGB pixels.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)     # (H, W, 3) RGB pixels
    env = MiniGridWrapper(
        env,
        frame_stack=frame_stack,
        screen_size=screen_size,
    )
    return env