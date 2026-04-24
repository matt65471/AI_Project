"""Recurrent DQN (DRQN) network.

Architecture (input: B × T × C × 84 × 84 uint8 frames):
  CNN (same as DQN - Mnih et al. 2015)
    Conv1  32 filters  8×8  stride 4  ReLU
    Conv2  64 filters  4×4  stride 2  ReLU
    Conv3  64 filters  3×3  stride 1  ReLU
  Flatten                             -> 3136
  LSTM   hidden_size units  (default 512)
  FC     n_actions  linear

The convolutional features at each timestep are fed into the LSTM, and the
per-timestep LSTM output is projected to Q-values.  Following Hausknecht & Stone
(2015), DRQN uses an LSTM in place of the penultimate fully-connected layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DRQNNetwork(nn.Module):
    """Recurrent Q-network: CNN -> LSTM -> FC."""

    def __init__(
        self,
        n_actions: int,
        frame_stack: int = 4,
        lstm_hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.lstm_hidden_size = lstm_hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, frame_stack, 84, 84)
            conv_out_size = int(self.conv(dummy).flatten(start_dim=1).shape[1])
        self.conv_out_size = conv_out_size

        self.lstm = nn.LSTM(
            input_size=conv_out_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Linear(lstm_hidden_size, n_actions)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:      (B, C, H, W)  - single timestep, or
                    (B, T, C, H, W) - sequence of T timesteps.
                    uint8 tensors are normalised to [0, 1].
            hidden: (h, c) each shaped (1, B, lstm_hidden_size).  If None, a
                    zero hidden state is used (standard DRQN zero-init).
        Returns:
            q:      (B, T, n_actions)  Q-values per timestep.
                    If input was (B, C, H, W), T is inserted as 1 for the
                    caller's convenience; squeeze if needed.
            hidden: final LSTM hidden state (h, c).
        """
        squeezed_time = False
        if x.dim() == 4:
            x = x.unsqueeze(1)
            squeezed_time = True

        B, T = x.shape[0], x.shape[1]

        flat = x.reshape(B * T, *x.shape[2:])
        if flat.dtype == torch.uint8:
            flat = flat.float() / 255.0

        feats = self.conv(flat).flatten(start_dim=1)
        feats = feats.view(B, T, self.conv_out_size)

        out, hidden = self.lstm(feats, hidden)
        q = self.head(out)

        if squeezed_time:
            q = q.squeeze(1)
        return q, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return h, c
