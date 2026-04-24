"""Environment preprocessing for DQN on either Atari Pong or MiniGrid Memory."""

from __future__ import annotations

from collections import deque
from typing import Literal

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

import ale_py

gym.register_envs(ale_py)


_RGB_TO_Y = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} via the sign function (Section 5.1 of paper)."""

    def reward(self, reward: float) -> float:  # type: ignore[override]
        return float(np.sign(reward))


class NoopResetWrapper(gym.Wrapper):
    """Execute a random number of no-op actions at the start of each episode."""

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeWrapper(gym.Wrapper):
    """Treat life loss as terminal, while only resetting on real game over."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self._real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._real_done = terminated or truncated
        lives = info.get("lives", self.lives)
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self._real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = info.get("lives", 0)
        return obs, info


class DQNAtariPreprocessWrapper(gym.Wrapper):
    """Max over two frames, grayscale, resize, and frame-stack for Atari."""

    def __init__(
        self,
        env: gym.Env,
        *,
        frame_stack: int = 4,
        screen_size: tuple[int, int] = (84, 84),
    ) -> None:
        super().__init__(env)
        self.frame_stack = frame_stack
        self.screen_size = screen_size
        self._prev_raw: np.ndarray | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)

        h, w = screen_size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(frame_stack, h, w), dtype=np.uint8
        )

    def _process_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        y = np.tensordot(rgb, _RGB_TO_Y, axes=([2], [0]))
        y = np.clip(y, 0, 255).astype(np.uint8)
        h, w = self.screen_size
        return cv2.resize(y, (w, h), interpolation=cv2.INTER_AREA)

    def _get_obs(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        assert obs.ndim == 3 and obs.shape[-1] == 3
        self._prev_raw = obs
        processed = self._process_rgb(obs)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(processed)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert obs.ndim == 3 and obs.shape[-1] == 3
        assert self._prev_raw is not None
        fused = np.maximum(obs, self._prev_raw)
        self._prev_raw = obs
        processed = self._process_rgb(fused)
        self._frames.append(processed)
        return self._get_obs(), reward, terminated, truncated, info


class DQNMinigridPreprocessWrapper(gym.Wrapper):
    """
    1. Luminance (grayscale) from RGB.
    2. Resize to ``screen_size`` (default 84×84).
    3. Stack the last ``frame_stack`` processed frames (default 4).

    Expects observations shaped (H, W, 3) uint8 RGB from MiniGrid wrappers.
    Output observation shape: ``(frame_stack, screen_size[0], screen_size[1])`` uint8.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        frame_stack: int = 4,
        screen_size: tuple[int, int] = (84, 84),
    ) -> None:
        super().__init__(env)
        self.frame_stack = frame_stack
        self.screen_size = screen_size
        self._frames: deque[np.ndarray] = deque(maxlen=frame_stack)

        h, w = screen_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, h, w),
            dtype=np.uint8,
        )

    def _process_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """RGB (H,W,3) -> grayscale resized uint8 (h,w)."""
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        y = np.tensordot(rgb, _RGB_TO_Y, axes=([2], [0]))
        y = np.clip(y, 0, 255).astype(np.uint8)
        h, w = self.screen_size
        return cv2.resize(y, (w, h), interpolation=cv2.INTER_AREA)

    def _get_obs(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        assert obs.ndim == 3 and obs.shape[-1] == 3, (
            f"Expected RGB (H,W,3), got shape {obs.shape}; use obs_type='rgb' when making the env."
        )
        processed = self._process_rgb(obs)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(processed)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        assert obs.ndim == 3 and obs.shape[-1] == 3
        processed = self._process_rgb(obs)
        self._frames.append(processed)
        return self._get_obs(), reward, terminated, truncated, info


def make_minigrid_dqn_env(
    env_id: str = "MiniGrid-MemoryS9-v0",
    *,
    render_mode: str | None = None,
    frame_stack: int = 4,
    screen_size: tuple[int, int] = (84, 84),
    clip_reward: bool = True,
) -> gym.Env:
    """Build a preprocessed MiniGrid Memory environment for DQN.

    Wrappers applied in order (innermost first):
      1. MiniGrid env
      2. RGBImgPartialObsWrapper - agent-view RGB image observations
      3. ImgObsWrapper - convert dict observation -> raw image array
      4. DQNMinigridPreprocessWrapper - grayscale, resize, frame stack
      5. ClipRewardWrapper (optional)
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = DQNMinigridPreprocessWrapper(env, frame_stack=frame_stack, screen_size=screen_size)
    if clip_reward:
        env = ClipRewardWrapper(env)
    return env


def make_atari_dqn_env(
    env_id: str = "ALE/Pong-v5",
    *,
    render_mode: str | None = None,
    frame_stack: int = 4,
    screen_size: tuple[int, int] = (84, 84),
    noop_max: int = 30,
    episodic_life: bool = True,
    clip_reward: bool = True,
) -> gym.Env:
    """Build a preprocessed Atari environment for DQN."""
    env = gym.make(env_id, render_mode=render_mode, obs_type="rgb")
    if noop_max > 0:
        env = NoopResetWrapper(env, noop_max=noop_max)
    if episodic_life:
        env = EpisodicLifeWrapper(env)
    env = DQNAtariPreprocessWrapper(env, frame_stack=frame_stack, screen_size=screen_size)
    if clip_reward:
        env = ClipRewardWrapper(env)
    return env


def make_dqn_env(
    domain: Literal["atari", "minigrid"],
    env_id: str,
    *,
    render_mode: str | None = None,
    clip_reward: bool = True,
) -> gym.Env:
    """Create either an Atari or MiniGrid DQN environment from one switch."""
    if domain == "atari":
        return make_atari_dqn_env(env_id, render_mode=render_mode, clip_reward=clip_reward)
    return make_minigrid_dqn_env(env_id, render_mode=render_mode, clip_reward=clip_reward)
