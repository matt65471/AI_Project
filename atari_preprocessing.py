"""DQN-style Atari preprocessing (Mnih et al.): max over two RGB frames, luminance, 84×84, stack m frames."""

from __future__ import annotations

from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ale-py ≥ 0.11 requires explicit registration with Gymnasium.
gym.register_envs(ale_py)


# ITU-R BT.601 luma from RGB (same idea as extracting "Y" from RGB)
_RGB_TO_Y = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class NoopResetWrapper(gym.Wrapper):
    """Execute a random number of no-op actions at the start of each episode.

    The paper uses up to 30 no-ops so the agent cannot memorize a fixed
    initial state.  Action 0 is assumed to be NOOP.
    """

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
    """Treat the loss of a life as a terminal state (but do not reset the game).

    This improves value estimation for games with multiple lives by giving
    the agent a denser terminal signal.  The actual game only resets when
    *all* lives are gone.
    """

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
            # Step with NOOP to advance from the life-lost state.
            obs, _, _, _, info = self.env.step(0)
        self.lives = info.get("lives", 0)
        return obs, info


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} via the sign function (Section 5.1 of paper)."""

    def reward(self, reward: float) -> float:  # type: ignore[override]
        return float(np.sign(reward))


class DQNAtariPreprocessWrapper(gym.Wrapper):
    """
    1. Per-pixel max over current and previous raw RGB frame (reduces sprite flicker).
    2. Luminance (grayscale) from RGB.
    3. Resize to ``screen_size`` (default 84×84).
    4. Stack the last ``frame_stack`` processed frames (default 4).

    Expects observations shaped (H, W, 3) uint8 RGB from the ALE env (``obs_type="rgb"``).
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
        self._prev_raw: np.ndarray | None = None
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
        self._prev_raw = obs
        fused = np.maximum(obs, obs)
        processed = self._process_rgb(fused)
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
    """Build a fully preprocessed DQN Atari environment.

    Wrappers applied in order (innermost first):
      1. ALE gym env (RGB, frameskip=4 via ALE default)
      2. NoopResetWrapper  – random start position
      3. EpisodicLifeWrapper – life loss = terminal signal
      4. DQNAtariPreprocessWrapper – max-pool, grayscale, resize, frame-stack
      5. ClipRewardWrapper – reward in {-1, 0, +1}
    """
    env = gym.make(env_id, render_mode=render_mode, obs_type="rgb")
    if noop_max > 0:
        env = NoopResetWrapper(env, noop_max=noop_max)
    if episodic_life:
        env = EpisodicLifeWrapper(env)
    env = DQNAtariPreprocessWrapper(env, frame_stack=frame_stack, screen_size=screen_size)
    if clip_reward:
        env = ClipRewardWrapper(env)
    return env
