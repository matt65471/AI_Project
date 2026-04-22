"""Uniform experience replay memory (Mnih et al., 2015 – Algorithm 1, line 5).

Stores transitions (s, a, r, s', done) in a fixed-size circular buffer and
samples uniform random mini-batches for training.

Memory layout
-------------
Observations are stored as uint8 to keep RAM usage manageable.  At 1 M
transitions with 4-frame 84×84 stacks the uncompressed size is:
  1e6 × 2 (s + s') × 4 × 84 × 84 ≈ 56 GB
Instead we store each *frame* once and reconstruct stacks on sampling.
This "frame buffer" approach reduces that to ~28 GB for the raw frames, but
for simplicity this implementation stores full (s, s') stacks which is more
straightforward and fine for shorter training runs (up to ~100 K steps).
Set ``capacity`` lower if memory is limited.
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    """Circular replay buffer storing uint8 stacked-frame transitions.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of a single observation (C, H, W), e.g. (4, 84, 84).
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self._pos = 0
        self._size = 0

        # Pre-allocate numpy arrays for efficiency.
        self._obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self._actions  = np.zeros(capacity, dtype=np.int64)
        self._rewards  = np.zeros(capacity, dtype=np.float32)
        self._dones    = np.zeros(capacity, dtype=np.bool_)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        i = self._pos
        self._obs[i]      = obs
        self._next_obs[i] = next_obs
        self._actions[i]  = action
        self._rewards[i]  = reward
        self._dones[i]    = done
        self._pos  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Return a dict of randomly sampled arrays, each of length ``batch_size``."""
        if self._size < batch_size:
            raise ValueError(
                f"Not enough transitions: have {self._size}, need {batch_size}."
            )
        idxs = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs":      self._obs[idxs],
            "actions":  self._actions[idxs],
            "rewards":  self._rewards[idxs],
            "next_obs": self._next_obs[idxs],
            "dones":    self._dones[idxs],
        }

    def __len__(self) -> int:
        return self._size
