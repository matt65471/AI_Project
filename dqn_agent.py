"""DQN agent (Mnih et al., 2015).

Implements:
  - ε-greedy exploration with linear annealing (Table 1)
  - Experience replay (Algorithm 1)
  - Separate target network updated every C steps (Algorithm 1)
  - RMSProp optimiser (Extended Data Table 1)
  - Huber loss (gradient clipping equivalent, Section 5.1)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from dqn_model import DQNNetwork
from replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    pass


class DQNAgent:
    """DQN agent faithful to the Nature paper hyper-parameters.

    Paper hyper-parameters (Extended Data Table 1):
      replay memory size      N = 1 000 000
      minibatch size              32
      replay start size       50 000  (random exploration before learning)
      agent history length     m = 4  (frame stack – handled by the env)
      target update frequency  C = 10 000
      discount factor          γ = 0.99
      ε initial value              1.0
      ε final value                0.1
      ε annealed over         1 000 000 frames
      learning rate          α = 0.00025
      RMSProp momentum            0.95
      RMSProp squared-gradient    0.95
      RMSProp min squared grad    0.01
      max no-op steps             30
    """

    # Paper hyper-parameters
    REPLAY_CAPACITY   = 1_000_000
    BATCH_SIZE        = 32
    REPLAY_START_SIZE = 50_000
    TARGET_UPDATE_FREQ = 10_000   # steps
    GAMMA             = 0.99
    EPS_START         = 1.0
    EPS_END           = 0.1
    EPS_ANNEAL_STEPS  = 1_000_000
    LR                = 0.00025
    RMS_ALPHA         = 0.95      # squared-gradient decay
    RMS_EPS           = 0.01      # min squared gradient (paper calls it ε_min)

    def __init__(
        self,
        n_actions: int,
        obs_shape: tuple[int, ...],
        *,
        device: str | torch.device | None = None,
        # Allow overriding any paper hyper-parameter for ablation studies.
        replay_capacity: int | None = None,
        batch_size: int | None = None,
        replay_start_size: int | None = None,
        target_update_freq: int | None = None,
        gamma: float | None = None,
        eps_start: float | None = None,
        eps_end: float | None = None,
        eps_anneal_steps: int | None = None,
        lr: float | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.obs_shape = obs_shape

        # Hyper-parameters (allow override)
        self.replay_capacity    = replay_capacity   or self.REPLAY_CAPACITY
        self.batch_size         = batch_size        or self.BATCH_SIZE
        self.replay_start_size  = replay_start_size or self.REPLAY_START_SIZE
        self.target_update_freq = target_update_freq or self.TARGET_UPDATE_FREQ
        self.gamma              = gamma             if gamma is not None else self.GAMMA
        self.eps_start          = eps_start         if eps_start is not None else self.EPS_START
        self.eps_end            = eps_end           if eps_end is not None else self.EPS_END
        self.eps_anneal_steps   = eps_anneal_steps  or self.EPS_ANNEAL_STEPS
        self.lr                 = lr                or self.LR

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Networks
        frame_stack = obs_shape[0]
        self.online_net = DQNNetwork(n_actions, frame_stack=frame_stack).to(self.device)
        self.target_net = DQNNetwork(n_actions, frame_stack=frame_stack).to(self.device)
        self._sync_target()
        self.target_net.eval()

        # Optimiser: RMSProp as specified in the paper.
        # PyTorch's RMSprop: alpha ↔ smoothing constant (squared-gradient decay),
        #                    eps  ↔ numerical stabiliser (we use RMS_EPS = 0.01).
        self.optimizer = torch.optim.RMSprop(
            self.online_net.parameters(),
            lr=self.lr,
            alpha=self.RMS_ALPHA,
            eps=self.RMS_EPS,
        )

        # Replay memory
        self.memory = ReplayBuffer(self.replay_capacity, obs_shape)

        # Training counters
        self.steps_done = 0      # total environment steps
        self.updates_done = 0    # gradient updates

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy action selection.

        ε is linearly annealed from EPS_START to EPS_END over EPS_ANNEAL_STEPS.
        After that it stays at EPS_END.
        """
        eps = max(
            self.eps_end,
            self.eps_start
            - (self.eps_start - self.eps_end) * self.steps_done / self.eps_anneal_steps,
        )

        if random.random() < eps:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            q_values = self.online_net(state)
            return int(q_values.argmax(dim=1).item())

    def store(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into replay memory and increment step counter."""
        self.memory.push(obs, action, reward, next_obs, done)
        self.steps_done += 1

    def update(self) -> float | None:
        """Sample a minibatch and perform one gradient-descent step.

        Returns the scalar loss if an update was performed, else None.
        """
        if len(self.memory) < self.replay_start_size:
            return None

        batch = self.memory.sample(self.batch_size)

        obs      = torch.from_numpy(batch["obs"]).to(self.device)
        actions  = torch.from_numpy(batch["actions"]).to(self.device)
        rewards  = torch.from_numpy(batch["rewards"]).to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).to(self.device)
        dones    = torch.from_numpy(batch["dones"]).to(self.device)

        # Current Q-values for the actions that were taken.
        q_values = self.online_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (no gradient through target network).
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1).values
            # Terminal states have no future reward.
            targets = rewards + self.gamma * next_q * (~dones)

        # Huber loss (≡ smooth L1) – equivalent to the gradient clipping
        # described in Section 5.1 of the paper.
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates_done += 1

        # Periodically copy online weights to target network.
        if self.updates_done % self.target_update_freq == 0:
            self._sync_target()

        return loss.item()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_hidden(self) -> None:
        """No-op for the feed-forward DQN; present for interface parity with DRQN."""
        return None

    def current_epsilon(self) -> float:
        return max(
            self.eps_end,
            self.eps_start
            - (self.eps_start - self.eps_end) * self.steps_done / self.eps_anneal_steps,
        )

    def _sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

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
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done  = ckpt.get("steps_done", 0)
        self.updates_done = ckpt.get("updates_done", 0)
