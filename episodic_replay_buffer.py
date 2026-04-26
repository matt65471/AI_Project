"""Episodic replay buffer for recurrent DQN, with optional prioritized sampling.

Stores full episodes and samples fixed-length contiguous subsequences so the
LSTM can be trained on temporally coherent rollouts.

Two sampling modes:

  * Uniform (default): picks an episode with probability proportional to its
    length, then a random subsequence within it.  Equivalent to uniform
    transition-level sampling.

  * Prioritized: each episode carries a priority p_i, and is sampled with
    probability proportional to p_i^alpha.  After each update the agent
    reports per-sequence TD errors and the priorities are refreshed so that
    high-error (i.e. surprising / informative) episodes are revisited more
    often.  Importance-sampling weights w_i = (N * P(i))^(-beta) are returned
    so the loss can be debiased.

This is a per-episode flavour of Prioritized Experience Replay (Schaul et al.
2016), simpler than per-transition PER but well suited to sparse-reward
recurrent tasks where the rare successful episodes are the ones we want to
revisit.

Capacity is measured in transitions; oldest whole episodes are evicted when
the buffer would exceed capacity.
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
    """Fixed-capacity episode buffer.  Optionally prioritized."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        *,
        prioritized: bool = False,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape

        self.prioritized = prioritized
        self.alpha = alpha
        self.eps = eps

        self._episodes: Deque[_Episode] = deque()
        self._priorities: Deque[float] = deque()
        self._max_priority: float = 1.0
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
        """Append one transition; finalise episode on done=True."""
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
            # New episodes start at the current max priority so they are
            # guaranteed to be sampled at least once before being deprioritised.
            self._priorities.append(self._max_priority)
            self._size += ep.length

            while self._size > self.capacity and self._episodes:
                dropped = self._episodes.popleft()
                self._priorities.popleft()
                self._size -= dropped.length

            self._cur_obs.clear()
            self._cur_actions.clear()
            self._cur_rewards.clear()
            self._cur_next_obs.clear()
            self._cur_dones.clear()

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        *,
        beta: float = 0.4,
    ) -> dict[str, np.ndarray]:
        """Return a batch of padded subsequences plus mask, indices, IS weights.

        Shapes (B = batch_size, T = seq_len, * = obs_shape):
            obs        (B, T, *)  uint8
            actions    (B, T)     int64
            rewards    (B, T)     float32
            next_obs   (B, T, *)  uint8
            dones      (B, T)     bool
            mask       (B, T)     float32  (1.0 = valid, 0.0 = padding)
            indices    (B,)       int64    episode indices (for priority update)
            is_weights (B,)       float32  importance-sampling weights
        """
        if not self._episodes:
            raise ValueError("Replay buffer is empty.")

        n_eps = len(self._episodes)

        if self.prioritized:
            priorities = np.fromiter(self._priorities, dtype=np.float64, count=n_eps)
            scaled = priorities ** self.alpha
            probs = scaled / scaled.sum()
            ep_idxs = np.random.choice(n_eps, size=batch_size, p=probs)
            sample_probs = probs[ep_idxs]
            is_weights = (n_eps * sample_probs) ** (-beta)
            is_weights = is_weights / is_weights.max()
            is_weights = is_weights.astype(np.float32)
        else:
            lengths = np.array(
                [ep.length for ep in self._episodes], dtype=np.int64
            )
            probs = lengths / lengths.sum()
            ep_idxs = np.random.choice(n_eps, size=batch_size, p=probs)
            is_weights = np.ones(batch_size, dtype=np.float32)

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
                # Bias starts toward 0 so the LSTM sees the cue more often.
                # 50% of the time start at the beginning of the episode.
                if np.random.rand() < 0.5:
                    start = 0
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
            "obs":        obs_batch,
            "actions":    act_batch,
            "rewards":    rew_batch,
            "next_obs":   next_obs_batch,
            "dones":      done_batch,
            "mask":       mask_batch,
            "indices":    ep_idxs.astype(np.int64),
            "is_weights": is_weights,
        }

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """Refresh priorities for the episodes at the given indices.

        ``priorities`` are typically |TD error| values computed by the agent
        after the gradient step.  A small constant ``eps`` is added so no
        episode ever has zero probability of being sampled again.
        """
        if not self.prioritized:
            return
        n_eps = len(self._priorities)
        for idx, p in zip(indices, priorities):
            i = int(idx)
            if 0 <= i < n_eps:
                new_p = float(p) + self.eps
                self._priorities[i] = new_p
                if new_p > self._max_priority:
                    self._max_priority = new_p
