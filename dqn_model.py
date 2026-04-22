"""DQN convolutional network (Mnih et al., 2015 – Table 1 / Extended Data Table 1).

Architecture (input: 4 × 84 × 84 uint8 frames, normalised to [0, 1]):
  Conv1  32 filters  8×8  stride 4  ReLU
  Conv2  64 filters  4×4  stride 2  ReLU
  Conv3  64 filters  3×3  stride 1  ReLU
  FC     512 units              ReLU
  Out    n_actions  linear

Output dimensions after each conv (input 84×84):
  Conv1 → (32, 20, 20)   floor((84-8)/4)+1 = 20
  Conv2 → (64,  9,  9)   floor((20-4)/2)+1 =  9
  Conv3 → (64,  7,  7)   floor(( 9-3)/1)+1 =  7
  Flatten → 64×7×7 = 3 136
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Q-network mapping a stacked-frame observation to per-action Q-values."""

    def __init__(self, n_actions: int, frame_stack: int = 4) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # 64 * 7 * 7 = 3136 for an 84×84 input
        conv_out_size = 64 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: uint8 or float tensor of shape (B, C, 84, 84).
               Automatically normalised to [0, 1] if dtype is uint8.
        Returns:
            Q-values of shape (B, n_actions).
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return self.fc(self.conv(x).flatten(start_dim=1))
