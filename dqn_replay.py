"""Uniform experience replay for DQN (Mnih et al., 2015)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """
    Ring buffer of ``(s, a, r, s', done)``. Default capacity matches the Nature paper (1e6).
    Observations are stored as ``uint8`` ``(frame_stack, H, W)`` (e.g. 4×84×84).
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        obs_shape: tuple[int, ...] = (4, 84, 84),
    ) -> None:
        self.capacity = int(capacity)
        self.obs_shape = obs_shape
        self._obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self._next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self._actions = np.zeros((self.capacity,), dtype=np.int64)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._idx = 0
        self.size = 0

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        np.copyto(self._obs[self._idx], obs)
        np.copyto(self._next_obs[self._idx], next_obs)
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._dones[self._idx] = float(done)
        self._idx = (self._idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError(f"buffer size {self.size} < batch_size {batch_size}")
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self._obs[idx], device=device, dtype=torch.float32).div_(255.0)
        next_obs = torch.as_tensor(self._next_obs[idx], device=device, dtype=torch.float32).div_(
            255.0
        )
        actions = torch.as_tensor(self._actions[idx], device=device, dtype=torch.long)
        rewards = torch.as_tensor(self._rewards[idx], device=device, dtype=torch.float32)
        dones = torch.as_tensor(self._dones[idx], device=device, dtype=torch.float32)
        return obs, actions, rewards, next_obs, dones
