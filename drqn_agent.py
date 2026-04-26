"""Recurrent DQN (DRQN) agent.

Same training loop skeleton as :class:`DQNAgent` (ε-greedy, replay, target
network, RMSProp, Huber loss) but:

  - the network carries an LSTM between the convolutional features and the
    Q head, so the agent has internal memory that persists across timesteps;
  - replay stores whole episodes, and updates are computed over sampled
    fixed-length subsequences with a validity mask;
  - during acting, the hidden state is carried from step to step within an
    episode and reset via :meth:`reset_hidden` at episode boundaries.

Defaults follow Hausknecht & Stone (2015): LSTM hidden size 512, sequence
length 10, zero-initialised hidden state at the start of each training
subsequence.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from drqn_model import DRQNNetwork
from episodic_replay_buffer import EpisodicReplayBuffer


class DRQNAgent:
    """DRQN agent with per-sequence training and per-episode hidden state."""

    REPLAY_CAPACITY    = 1_000_000
    BATCH_SIZE         = 32
    REPLAY_START_SIZE  = 50_000
    TARGET_UPDATE_FREQ = 10_000
    GAMMA              = 0.99
    EPS_START          = 1.0
    EPS_END            = 0.1
    EPS_ANNEAL_STEPS   = 1_000_000
    LR                 = 0.00025
    RMS_ALPHA          = 0.95
    RMS_EPS            = 0.01

    LSTM_HIDDEN_SIZE   = 128
    SEQ_LEN            = 16

    # Prioritized Experience Replay defaults (Schaul et al. 2016).
    PER_ALPHA          = 0.6   # how strongly priorities bias sampling
    PER_BETA_START     = 0.4   # IS-weight exponent at start of training
    PER_BETA_END       = 1.0   # IS-weight exponent fully on by end of anneal
    PER_BETA_ANNEAL    = 1_000_000  # env steps over which beta anneals

    def __init__(
        self,
        n_actions: int,
        obs_shape: tuple[int, ...],
        *,
        device: str | torch.device | None = None,
        replay_capacity: int | None = None,
        batch_size: int | None = None,
        replay_start_size: int | None = None,
        target_update_freq: int | None = None,
        gamma: float | None = None,
        eps_start: float | None = None,
        eps_end: float | None = None,
        eps_anneal_steps: int | None = None,
        lr: float | None = None,
        lstm_hidden_size: int | None = None,
        seq_len: int | None = None,
        prioritized: bool = False,
        per_alpha: float | None = None,
        per_beta_start: float | None = None,
        per_beta_end: float | None = None,
        per_beta_anneal: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.obs_shape = obs_shape

        self.replay_capacity    = replay_capacity    or self.REPLAY_CAPACITY
        self.batch_size         = batch_size         or self.BATCH_SIZE
        self.replay_start_size  = replay_start_size  or self.REPLAY_START_SIZE
        self.target_update_freq = target_update_freq or self.TARGET_UPDATE_FREQ
        self.gamma              = gamma              if gamma is not None else self.GAMMA
        self.eps_start          = eps_start          if eps_start is not None else self.EPS_START
        self.eps_end            = eps_end            if eps_end is not None else self.EPS_END
        self.eps_anneal_steps   = eps_anneal_steps   or self.EPS_ANNEAL_STEPS
        self.lr                 = lr                 or self.LR
        self.lstm_hidden_size   = lstm_hidden_size   or self.LSTM_HIDDEN_SIZE
        self.seq_len            = seq_len            or self.SEQ_LEN

        self.prioritized        = prioritized
        self.per_alpha          = per_alpha          if per_alpha is not None else self.PER_ALPHA
        self.per_beta_start     = per_beta_start     if per_beta_start is not None else self.PER_BETA_START
        self.per_beta_end       = per_beta_end       if per_beta_end is not None else self.PER_BETA_END
        self.per_beta_anneal    = per_beta_anneal    or self.PER_BETA_ANNEAL

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        frame_stack = obs_shape[0]
        self.online_net = DRQNNetwork(
            n_actions,
            frame_stack=frame_stack,
            lstm_hidden_size=self.lstm_hidden_size,
        ).to(self.device)
        self.target_net = DRQNNetwork(
            n_actions,
            frame_stack=frame_stack,
            lstm_hidden_size=self.lstm_hidden_size,
        ).to(self.device)
        self._sync_target()
        self.target_net.eval()

        self.optimizer = torch.optim.RMSprop(
            self.online_net.parameters(),
            lr=self.lr,
            alpha=self.RMS_ALPHA,
            eps=self.RMS_EPS,
        )

        self.memory = EpisodicReplayBuffer(
            self.replay_capacity,
            obs_shape,
            prioritized=self.prioritized,
            alpha=self.per_alpha,
        )

        self.steps_done = 0
        self.updates_done = 0

        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_hidden(self) -> None:
        """Clear the acting LSTM state - call at the start of each episode."""
        self._hidden = None

    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy action selection that also advances the LSTM hidden state."""
        eps = self.current_epsilon()

        with torch.no_grad():
            state = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            q, self._hidden = self.online_net(state, self._hidden)

        if random.random() < eps:
            return random.randrange(self.n_actions)
        return int(q.reshape(-1).argmax().item())

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.push(obs, action, reward, next_obs, done)
        self.steps_done += 1

    def update(self) -> float | None:
        """One gradient step on a batch of sampled subsequences.

        With prioritized replay enabled, importance-sampling weights are applied
        to the per-sequence loss, and per-episode priorities are refreshed
        from the magnitude of the TD error after the step.
        """
        if len(self.memory) < self.replay_start_size:
            return None

        beta = self._current_per_beta()
        batch = self.memory.sample(self.batch_size, self.seq_len, beta=beta)

        obs        = torch.from_numpy(batch["obs"]).to(self.device)
        actions    = torch.from_numpy(batch["actions"]).to(self.device)
        rewards    = torch.from_numpy(batch["rewards"]).to(self.device)
        next_obs   = torch.from_numpy(batch["next_obs"]).to(self.device)
        dones      = torch.from_numpy(batch["dones"]).to(self.device)
        mask       = torch.from_numpy(batch["mask"]).to(self.device)
        is_weights = torch.from_numpy(batch["is_weights"]).to(self.device)

        q_all, _ = self.online_net(obs)
        q_values = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # Double DQN: pick action with online net, evaluate it with target net.
            next_actions = self.online_net(next_obs)[0].argmax(dim=2)
            next_q_all, _ = self.target_net(next_obs)
            next_q = next_q_all.gather(2, next_actions.unsqueeze(-1)).squeeze(-1)
            targets = rewards + self.gamma * next_q * (~dones)

        elementwise = nn.functional.smooth_l1_loss(
            q_values, targets, reduction="none"
        )
        # Per-sequence loss weighted by IS weights (PER); reduces to plain
        # masked Huber when prioritized=False (all weights equal 1).
        per_seq_valid = mask.sum(dim=1).clamp(min=1.0)
        per_seq_loss  = (elementwise * mask).sum(dim=1) / per_seq_valid
        loss = (per_seq_loss * is_weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates_done += 1

        if self.updates_done % self.target_update_freq == 0:
            self._sync_target()

        if self.prioritized:
            with torch.no_grad():
                td_abs = (q_values - targets).abs() * mask
                # Per-episode priority = mean |TD error| over valid timesteps.
                ep_priorities = (td_abs.sum(dim=1) / per_seq_valid).cpu().numpy()
            self.memory.update_priorities(batch["indices"], ep_priorities)

        return loss.item()

    def _current_per_beta(self) -> float:
        """Linearly anneal beta from per_beta_start to per_beta_end."""
        frac = min(1.0, self.steps_done / self.per_beta_anneal)
        return self.per_beta_start + (self.per_beta_end - self.per_beta_start) * frac

    def current_epsilon(self) -> float:
        return max(
            self.eps_end,
            self.eps_start
            - (self.eps_start - self.eps_end) * self.steps_done / self.eps_anneal_steps,
        )

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "updates_done": self.updates_done,
                "lstm_hidden_size": self.lstm_hidden_size,
                "seq_len": self.seq_len,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done   = ckpt.get("steps_done", 0)
        self.updates_done = ckpt.get("updates_done", 0)
