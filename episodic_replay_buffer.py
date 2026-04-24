"""Episodic replay buffer for recurrent DQN.

Stores full episodes and samples fixed-length contiguous subsequences so the
LSTM can be trained on temporally coherent rollouts.

Sampling strategy (simple, per Hausknecht & Stone 2015):
  - Pick a random episode (weighted by length so every transition is
    equally likely to appear).
  - Pick a random start index within that episode.
  - Return seq_len consecutive transitions.  If the episode ends before
    seq_len is reached, the tail is zero-padded and a boolean mask marks
    the valid timesteps so the loss can ignore the padding.

Capacity is measured in *transitions*, matching the Nature DQN buffer; when
the buffer would exceed capacity the oldest whole episodes are evicted.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np


class _Episode:
    __slots__ = ("obs", "actions", "rewards", "next_obs", "dones", "length")

    def __init__(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.next_obs = next_obs
        self.dones = dones
        self.length = obs.shape[0]


class EpisodicReplayBuffer:
    """Fixed-capacity episode buffer; samples fixed-length subsequences."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape

        self._episodes: Deque[_Episode] = deque()
        self._size = 0

        self._cur_obs: list[np.ndarray] = []
        self._cur_actions: list[int] = []
        self._cur_rewards: list[float] = []
        self._cur_next_obs: list[np.ndarray] = []
        self._cur_dones: list[bool] = []

    def __len__(self) -> int:
        return self._size

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Append one transition to the episode currently being recorded.

        When ``done`` is True the episode is finalised and added to the buffer,
        evicting the oldest episodes if capacity would be exceeded.
        """
        self._cur_obs.append(obs)
        self._cur_actions.append(action)
        self._cur_rewards.append(reward)
        self._cur_next_obs.append(next_obs)
        self._cur_dones.append(done)

        if done:
            ep = _Episode(
                obs=np.stack(self._cur_obs).astype(np.uint8, copy=False),
                actions=np.asarray(self._cur_actions, dtype=np.int64),
                rewards=np.asarray(self._cur_rewards, dtype=np.float32),
                next_obs=np.stack(self._cur_next_obs).astype(np.uint8, copy=False),
                dones=np.asarray(self._cur_dones, dtype=np.bool_),
            )
            self._episodes.append(ep)
            self._size += ep.length

            while self._size > self.capacity and self._episodes:
                dropped = self._episodes.popleft()
                self._size -= dropped.length

            self._cur_obs.clear()
            self._cur_actions.clear()
            self._cur_rewards.clear()
            self._cur_next_obs.clear()
            self._cur_dones.clear()

    def sample(self, batch_size: int, seq_len: int) -> dict[str, np.ndarray]:
        """Return a batch of padded subsequences and a validity mask.

        Shapes (B = batch_size, T = seq_len, * = obs_shape):
            obs      (B, T, *)   uint8
            actions  (B, T)      int64
            rewards  (B, T)      float32
            next_obs (B, T, *)   uint8
            dones    (B, T)      bool
            mask     (B, T)      float32  (1.0 = valid, 0.0 = padding)
        """
        if not self._episodes:
            raise ValueError("Replay buffer is empty.")

        lengths = np.array([ep.length for ep in self._episodes], dtype=np.int64)
        probs = lengths / lengths.sum()
        ep_idxs = np.random.choice(len(self._episodes), size=batch_size, p=probs)

        obs_batch      = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=np.uint8)
        next_obs_batch = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=np.uint8)
        act_batch      = np.zeros((batch_size, seq_len), dtype=np.int64)
        rew_batch      = np.zeros((batch_size, seq_len), dtype=np.float32)
        done_batch     = np.zeros((batch_size, seq_len), dtype=np.bool_)
        mask_batch     = np.zeros((batch_size, seq_len), dtype=np.float32)

        for b, ep_idx in enumerate(ep_idxs):
            ep = self._episodes[int(ep_idx)]
            if ep.length <= seq_len:
                start = 0
                n = ep.length
            else:
                start = np.random.randint(0, ep.length - seq_len + 1)
                n = seq_len

            end = start + n
            obs_batch[b, :n]      = ep.obs[start:end]
            next_obs_batch[b, :n] = ep.next_obs[start:end]
            act_batch[b, :n]      = ep.actions[start:end]
            rew_batch[b, :n]      = ep.rewards[start:end]
            done_batch[b, :n]     = ep.dones[start:end]
            mask_batch[b, :n]     = 1.0

        return {
            "obs":      obs_batch,
            "actions":  act_batch,
            "rewards":  rew_batch,
            "next_obs": next_obs_batch,
            "dones":    done_batch,
            "mask":     mask_batch,
        }
